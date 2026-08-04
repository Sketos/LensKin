#!/usr/bin/env bash
# Launch LensKin KinMS-mock jobs non-interactively (no Slurm).
#
# Runs each job in the background with nohup, logging under logs/.
#
# Usage (from anywhere):
#   bash scripts/launch_jobs.sh
#   bash scripts/launch_jobs.sh parametric_flux pixelized
#   bash scripts/launch_jobs.sh --all
#   bash scripts/launch_jobs.sh --list
#   bash scripts/launch_jobs.sh --settings settings/runners/foo.json
#   bash scripts/launch_jobs.sh --foreground parametric_flux   # block until done
#
# Default suite (no args):
#   parametric_flux  pixelized  parametric_truth_kin  pixelized_truth_kin
#
# Environment:
#   LENSKIN_CONDA_ENV   conda env (default: autolens)
#   LENSKIN_CORES       Autofit/Nautilus workers (default: auto = nproc-1)
#   LENSKIN_THREADS     BLAS/OpenMP threads per process (default: 1 if cores>1)
#   LENSKIN_MAX_JOBS    max concurrent background jobs (default: 0 = unlimited)
#   LENSKIN_DRY_RUN=1   print commands without starting
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
PID_DIR="${LOG_DIR}/pids"

DEFAULT_SUITE=(
  parametric_flux
  pixelized
  parametric_truth_kin
  pixelized_truth_kin
)

resolve_cores() {
  local requested="${LENSKIN_CORES:-auto}"
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

configure_parallelism() {
  local cores threads
  cores="$(resolve_cores)"
  export LENSKIN_CORES="${cores}"
  if [ -n "${LENSKIN_THREADS:-}" ]; then
    threads="${LENSKIN_THREADS}"
  elif [ "${cores}" -gt 1 ]; then
    threads=1
  else
    threads="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
  fi
  export OMP_NUM_THREADS="${threads}"
  export MKL_NUM_THREADS="${threads}"
  export OPENBLAS_NUM_THREADS="${threads}"
  export NUMEXPR_NUM_THREADS="${threads}"
  export VECLIB_MAXIMUM_THREADS="${threads}"
  echo "Parallelism: LENSKIN_CORES=${LENSKIN_CORES}, BLAS/OMP threads=${threads}"
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

job_command() {
  # Prints the python/bash command for a named job (cwd = repo root).
  case "$1" in
    parametric_flux)
      echo "python scripts/run_fit.py --settings settings/runners/kinms_mock_parametric_flux.json"
      ;;
    pixelized)
      echo "python scripts/run_fit.py --settings settings/runners/kinms_mock_pixelized.json"
      ;;
    parametric)
      echo "python scripts/run_fit.py --settings settings/runners/kinms_mock_parametric.json"
      ;;
    parametric_highres)
      echo "python scripts/run_fit.py --settings settings/runners/kinms_mock_parametric_highres.json"
      ;;
    parametric_bbox512)
      echo "python scripts/run_fit.py --settings settings/runners/kinms_mock_parametric_bbox512.json"
      ;;
    parametric_1024)
      echo "python scripts/run_fit.py --settings settings/runners/kinms_mock_parametric_1024.json"
      ;;
    parametric_mock40)
      echo "python scripts/run_fit.py --settings settings/runners/kinms_mock_parametric_mock40.json"
      ;;
    parametric_truth_kin)
      echo "python scripts/test_phase2_parametric.py --settings settings/runners/kinms_mock_parametric_truth_kinematics.json"
      ;;
    pixelized_truth_kin)
      echo "python scripts/test_phase2_pixelization.py --settings settings/runners/kinms_mock_pixelized_truth_kinematics.json"
      ;;
    *)
      return 1
      ;;
  esac
}

list_jobs() {
  echo "Available job names:"
  local name
  for name in \
    parametric_flux pixelized parametric parametric_highres \
    parametric_bbox512 parametric_1024 parametric_mock40 \
    parametric_truth_kin pixelized_truth_kin
  do
    printf "  %-22s  %s\n" "${name}" "$(job_command "${name}")"
  done
  echo
  echo "Default suite: ${DEFAULT_SUITE[*]}"
}

wait_for_slot() {
  local max_jobs="${LENSKIN_MAX_JOBS:-0}"
  if [ "${max_jobs}" -le 0 ]; then
    return 0
  fi
  while true; do
    local running=0
    local pidfile
    shopt -s nullglob
    for pidfile in "${PID_DIR}"/*.pid; do
      local pid
      pid="$(cat "${pidfile}")"
      if kill -0 "${pid}" 2>/dev/null; then
        running=$((running + 1))
      else
        rm -f "${pidfile}"
      fi
    done
    shopt -u nullglob
    if [ "${running}" -lt "${max_jobs}" ]; then
      return 0
    fi
    sleep 30
  done
}

launch_one() {
  local job_name="$1"
  local cmd="$2"
  local foreground="${3:-0}"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local out="${LOG_DIR}/${job_name}_${stamp}.out"
  local err="${LOG_DIR}/${job_name}_${stamp}.err"
  local pidfile="${PID_DIR}/${job_name}_${stamp}.pid"

  mkdir -p "${LOG_DIR}" "${PID_DIR}"

  if [ "${LENSKIN_DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] ${job_name}: ${cmd}"
    echo "  stdout -> ${out}"
    return 0
  fi

  if [ "${foreground}" = "1" ]; then
    echo "Running ${job_name} in foreground..."
    echo "  command: ${cmd}"
    echo "  stdout:  ${out}"
    echo "  stderr:  ${err}"
    # shellcheck disable=SC2086
    bash -c "${cmd}" >"${out}" 2>"${err}"
    echo "Finished ${job_name} (exit $?)"
    return 0
  fi

  wait_for_slot

  # Detach fully from the terminal / SSH session.
  nohup bash -c "${cmd}" >"${out}" 2>"${err}" </dev/null &
  local pid=$!
  echo "${pid}" >"${pidfile}"
  disown "${pid}" 2>/dev/null || true

  echo "Started ${job_name} (pid ${pid})"
  echo "  command: ${cmd}"
  echo "  stdout:  ${out}"
  echo "  stderr:  ${err}"
  echo "  pidfile: ${pidfile}"
}

launch_named() {
  local name="$1"
  local foreground="${2:-0}"
  local cmd
  if ! cmd="$(job_command "${name}")"; then
    echo "ERROR: unknown job '${name}'. Use --list." >&2
    return 1
  fi
  # For fit jobs, skip if settings file is missing.
  if [[ "${cmd}" == *"--settings "* ]]; then
    local settings
    settings="$(echo "${cmd}" | sed -n 's/.*--settings //p' | awk '{print $1}')"
    if [ ! -f "${settings}" ]; then
      echo "Skipping ${name}: missing ${settings}"
      return 0
    fi
  fi
  launch_one "${name}" "${cmd}" "${foreground}"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_jobs.sh [job_name ...]
  bash scripts/launch_jobs.sh --all
  bash scripts/launch_jobs.sh --list
  bash scripts/launch_jobs.sh --foreground [job_name ...]
  bash scripts/launch_jobs.sh --settings path/to/settings.json [--name JOB]

No arguments → default KinMS mock suite in the background (nohup).
EOF
}

main() {
  cd "${REPO_ROOT}"

  local foreground=0
  local -a jobs=()

  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --list)
        list_jobs
        exit 0
        ;;
      --foreground)
        foreground=1
        shift
        ;;
      --all)
        jobs=(
          parametric_flux pixelized parametric parametric_highres
          parametric_bbox512 parametric_1024 parametric_mock40
          parametric_truth_kin pixelized_truth_kin
        )
        shift
        ;;
      --settings)
        shift
        local settings="${1:?--settings requires a path}"
        shift
        local job_name
        job_name="$(basename "${settings}" .json)"
        if [ "${1:-}" = "--name" ]; then
          shift
          job_name="${1:?--name requires a value}"
          shift
        fi
        activate_conda
        configure_parallelism
        launch_one "${job_name}" "python scripts/run_fit.py --settings ${settings}" "${foreground}"
        return 0
        ;;
      -*)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
      *)
        jobs+=("$1")
        shift
        ;;
    esac
  done

  if [ "${#jobs[@]}" -eq 0 ]; then
    jobs=("${DEFAULT_SUITE[@]}")
  fi

  activate_conda
  configure_parallelism
  echo "Repo:  ${REPO_ROOT}"
  echo "Conda: ${CONDA_DEFAULT_ENV:-}"
  echo "Host:  $(hostname)"
  echo "Date:  $(date)"
  echo

  local name
  for name in "${jobs[@]}"; do
    launch_named "${name}" "${foreground}"
  done

  if [ "${foreground}" = "0" ] && [ "${LENSKIN_DRY_RUN:-0}" != "1" ]; then
    echo
    echo "Jobs are running in the background. Monitor with:"
    echo "  tail -f logs/<job>_*.out"
    echo "  ls ${PID_DIR}"
  fi
}

main "$@"
