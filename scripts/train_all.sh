#!/usr/bin/env bash
# Train HybridDeepGaitSwin on all three datasets (60k iterations each).
#
# Usage:
#   export CASIAB_PKL=/path/to/casiab-pkl
#   export HID_PKL=/path/to/hid-pkl
#   export SUSTECH_PKL=/path/to/sustech-pkl
#   bash scripts/train_all.sh --gpu 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Usage: bash scripts/train_all.sh [--gpu N]"; exit 1 ;;
    esac
done

missing=0
for var in CASIAB_PKL HID_PKL SUSTECH_PKL; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] Set ${var} to the pretreated .pkl root for that dataset." >&2
        missing=1
    fi
done
if [[ "${missing}" -ne 0 ]]; then
    exit 1
fi

run_one() {
    local name="$1"
    local data_dir="$2"
    echo "===== START ${name} ====="
    bash "${REPO_ROOT}/train.sh" --dataset "${name}" --data_dir "${data_dir}" --gpu "${GPU}"
    echo "===== DONE ${name} ====="
}

echo "HybridDeepGaitSwin full training (60k iterations x 3 datasets)"
echo "GPU: ${GPU}"

run_one "casiab"  "${CASIAB_PKL}"
run_one "hid"     "${HID_PKL}"
run_one "sustech" "${SUSTECH_PKL}"

echo "All three datasets finished."
