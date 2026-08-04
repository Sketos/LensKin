#!/usr/bin/env bash
# Submit a Slurm job for settings/runners/kinms_mock_parametric_1024.json
#
# KinMS source cube: 1024x1024 over 5.0" (image plane stays 40x40).
#
# Usage (from anywhere):
#   bash scripts/slurm/submit_kinms_mock_parametric_1024.sh
#
# Optional environment variables:
#   LENSKIN_CONDA_ENV  conda environment name (default: autolens)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
SETTINGS="settings/runners/kinms_mock_parametric_1024.json"
JOB_NAME="kinms_mock_parametric_1024"

mkdir -p "${LOG_DIR}"

JOB_ID="$(
  sbatch --parsable \
    --job-name="${JOB_NAME}" \
    --partition=huge \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --chdir="${REPO_ROOT}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --export=ALL,LENSKIN_CONDA_ENV="${LENSKIN_CONDA_ENV:-autolens}" \
    --wrap="bash scripts/slurm/_run_fit.sh ${SETTINGS}"
)"

echo "Submitted job ${JOB_ID}"
echo "stdout: ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "stderr: ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.err"
