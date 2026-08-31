#!/usr/bin/env bash
# Create the conda environment used to train and evaluate HybridDeepGaitSwin.
#
# Usage:
#   bash environment.sh
#   conda activate hid2026
#
# Does not hard-code a machine path. Uses `conda` on PATH, or CONDA_BIN if set.

set -euo pipefail

ENV_NAME="hid2026"
PYTHON_VERSION="3.9"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CONDA_BIN:-}" ]]; then
    :
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
else
    echo "[ERROR] conda not found on PATH. Install Miniconda/Anaconda, or set CONDA_BIN." >&2
    exit 1
fi

echo "============================================================"
echo "  Creating conda environment: ${ENV_NAME}"
echo "  conda: ${CONDA_BIN}"
echo "============================================================"

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Environment '${ENV_NAME}' already exists. Skipping creation."
else
    "${CONDA_BIN}" create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    echo "[INFO] Environment '${ENV_NAME}' created."
fi

# Resolve env python/pip without assuming a install prefix.
CONDA_BASE="$("${CONDA_BIN}" info --base)"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

if [[ ! -x "${PYTHON}" ]]; then
    echo "[ERROR] Expected interpreter not found: ${PYTHON}" >&2
    exit 1
fi

echo ""
echo "============================================================"
echo "  Installing PyTorch 2.5.1 + CUDA 12.1"
echo "============================================================"
"${PIP}" install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "============================================================"
echo "  Installing remaining packages from requirements.txt"
echo "============================================================"
"${PIP}" install -r "${REPO_ROOT}/requirements.txt"
"${PIP}" install tb-nightly

echo ""
echo "============================================================"
echo "  Environment setup complete."
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "============================================================"
