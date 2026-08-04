#!/usr/bin/env bash
# Parametric phase-2 truth-kinematics test (phase 1 lens + scipy centre opt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
SETTINGS="settings/runners/kinms_mock_parametric_truth_kinematics.json"
JOB_NAME="kinms_mock_parametric_truth_kin"

mkdir -p "${LOG_DIR}"

JOB_ID="$(
  sbatch --parsable \
    --job-name="${JOB_NAME}" \
    --partition=huge \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --chdir="${REPO_ROOT}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --export=ALL,LENSKIN_CONDA_ENV="${LENSKIN_CONDA_ENV:-autolens}" \
    --wrap="bash -lc 'source ~/packages/miniforge3/etc/profile.d/conda.sh && conda activate \${LENSKIN_CONDA_ENV:-autolens} && python scripts/test_phase2_parametric.py --settings ${SETTINGS}'"
)"

echo "Submitted job ${JOB_ID}"
