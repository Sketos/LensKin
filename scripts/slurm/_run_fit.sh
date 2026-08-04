#!/usr/bin/env bash
# Shared Slurm job body: activate conda and run a LensKin fit.
set -euo pipefail

SETTINGS="${1:?Usage: _run_fit.sh <settings.json>}"

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
  echo "Could not find conda.sh. Set CONDA_EXE or activate your env before submitting." >&2
  exit 1
fi

conda activate "${LENSKIN_CONDA_ENV:-autolens}"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Settings: ${SETTINGS}"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

python scripts/run_fit.py --settings "${SETTINGS}"
