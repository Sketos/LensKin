#!/usr/bin/env bash
# Submit a Slurm job for settings/runners/kinms_mock_parametric_bbox512.json
#
# KinMS source plane: 512x512 over x,y in [-0.5, 0.5] arcsec (1"x1" bbox).
# Image plane stays 40x40 over 5.0".
#
# Usage (from anywhere):
#   bash scripts/slurm/submit_kinms_mock_parametric_bbox512.sh
#
# Optional environment variables:
#   LENSKIN_CONDA_ENV  conda environment name (default: autolens)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
SETTINGS="settings/runners/kinms_mock_parametric_bbox512.json"
JOB_NAME="kinms_mock_parametric_bbox512"

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
