#!/usr/bin/env bash
# Launch mode-2 (parametric_flux_from_phase1) source-kinematics centre test
# non-interactively with nohup, using multiple CPU cores on the host.
#
# Equivalent to:
#   python scripts/plot_source_kinematic_modes.py \
#     --settings settings/runners/kinms_mock_parametric_truth_kinematics.json \
#     --run-phase2-centre \
#     --force-phase1
#
# Phase-1: free lens centre in a ±0.05" box around lens_mass_model,
# regularization fixed at 6000. Phase-2 centre tests for all three SB modes
# use the phase-1 lens centre. Re-run with --force-phase1 so cached
# phase1_bundle.npz is not reused.
#
# Usage:
#   bash scripts/launch_mode2_phase2_centre.sh
#   bash scripts/launch_mode2_phase2_centre.sh --cores 16
#   LENSKIN_CORES=16 bash scripts/launch_mode2_phase2_centre.sh
#   bash scripts/launch_mode2_phase2_centre.sh --foreground
#   LENSKIN_DRY_RUN=1 bash scripts/launch_mode2_phase2_centre.sh
#
# Parallelism:
#   LENSKIN_CORES   Autofit/Nautilus worker count (default: auto = nproc-1).
#                   Also exported so Python search.resolve_number_of_cores sees it.
#   LENSKIN_THREADS Per-process BLAS/OpenMP/JAX threads (default: 1 when
#                   LENSKIN_CORES>1, else all CPUs). Override if needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
PID_DIR="${LOG_DIR}/pids"
JOB_NAME="mode2_phase2_centre"
SETTINGS="settings/runners/kinms_mock_parametric_truth_kinematics.json"
CMD="python scripts/plot_source_kinematic_modes.py --settings ${SETTINGS} --run-phase2-centre --force-phase1"

foreground=0
cores_arg=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --foreground|-f)
      foreground=1
      shift
      ;;
    --cores)
      cores_arg="${2:?--cores requires an integer or 'auto'}"
      shift 2
      ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

resolve_cores() {
  local requested="${cores_arg:-${LENSKIN_CORES:-auto}}"
  local available
  available="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
  if [ "${requested}" = "auto" ]; then
    if [ "${available}" -gt 1 ]; then
      echo $((available - 1))
    else
      echo 1
    fi
  else
    echo "${requested}"
  fi
}

activate_conda() {
  if [ -f "${HOME}/packages/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/packages/miniforge3/etc/profile.d/conda.sh"
  elif [ -n "${CONDA_EXE:-}" ]; then
    # shellcheck source=/dev/null
    source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
  elif [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh. Set CONDA_EXE or install miniforge." >&2
    exit 1
  fi
  conda activate "${LENSKIN_CONDA_ENV:-autolens}"
}

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

if [ ! -f "${SETTINGS}" ]; then
  echo "ERROR: settings not found: ${SETTINGS}" >&2
  exit 1
fi

CORES="$(resolve_cores)"
export LENSKIN_CORES="${CORES}"

# Multiprocessing pool + BLAS: keep one thread per worker to avoid oversubscription.
# For a single-process path (scipy centre opt), raise LENSKIN_THREADS if useful.
if [ -n "${LENSKIN_THREADS:-}" ]; then
  THREADS="${LENSKIN_THREADS}"
elif [ "${CORES}" -gt 1 ]; then
  THREADS=1
else
  THREADS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
fi
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
# JAX / XLA CPU parallelism (NUFFT path)
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=true}"

activate_conda

stamp="$(date +%Y%m%d_%H%M%S)"
out="${LOG_DIR}/${JOB_NAME}_${stamp}.out"
err="${LOG_DIR}/${JOB_NAME}_${stamp}.err"
pidfile="${PID_DIR}/${JOB_NAME}_${stamp}.pid"

echo "Repo:     ${REPO_ROOT}"
echo "Conda:    ${CONDA_DEFAULT_ENV:-}"
echo "Host:     $(hostname)"
echo "Cores:    LENSKIN_CORES=${LENSKIN_CORES}  (BLAS/OMP threads=${THREADS})"
echo "Command:  ${CMD}"

if [ "${LENSKIN_DRY_RUN:-0}" = "1" ]; then
  echo "[dry-run] would write stdout -> ${out}"
  exit 0
fi

if [ "${foreground}" = "1" ]; then
  echo "Running in foreground..."
  echo "  stdout: ${out}"
  echo "  stderr: ${err}"
  bash -c "${CMD}" >"${out}" 2>"${err}"
  echo "Finished (exit $?)"
  exit 0
fi

# Environment (LENSKIN_CORES, OMP_*, …) is inherited by nohup.
nohup bash -c "${CMD}" >"${out}" 2>"${err}" </dev/null &
pid=$!
echo "${pid}" >"${pidfile}"
disown "${pid}" 2>/dev/null || true

echo "Started ${JOB_NAME} (pid ${pid})"
echo "  stdout:  ${out}"
echo "  stderr:  ${err}"
echo "  pidfile: ${pidfile}"
echo "Monitor with: tail -f ${out}"
