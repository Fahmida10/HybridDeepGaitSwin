#!/usr/bin/env bash
# Convert raw gait images to the .pkl layout expected by OpenGait.
#
# Usage:
#   bash scripts/preprocess.sh --dataset casiab|hid|sustech --input <raw_dir> --output <pkl_dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

DATASET=""
INPUT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2";   shift 2 ;;
        --input)   INPUT_DIR="$2"; shift 2 ;;
        --output)  OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${DATASET}" || -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Usage: bash scripts/preprocess.sh --dataset casiab|hid|sustech --input <raw_dir> --output <pkl_dir>" >&2
    exit 1
fi

PYTHON_BIN="$(resolve_python)"
mkdir -p "${OUTPUT_DIR}"

case "${DATASET}" in
    hid)
        echo "[INFO] Preprocessing HID 2022 (train split)..."
        "${PYTHON_BIN}" "${OPENGAIT}/datasets/HID/pretreatment_HID.py" \
            --input_path "${INPUT_DIR}/train" \
            --output_path "${OUTPUT_DIR}"
        ;;
    sustech)
        echo "[INFO] Preprocessing SUSTech1K..."
        "${PYTHON_BIN}" "${OPENGAIT}/datasets/SUSTech1K/pretreatment_SUSTech1K.py" \
            --input_path "${INPUT_DIR}" \
            --output_path "${OUTPUT_DIR}" \
            --img_h 64 --img_w 64 --workers 8
        ;;
    casiab)
        echo "[INFO] Preprocessing CASIA-B..."
        "${PYTHON_BIN}" "${OPENGAIT}/datasets/pretreatment.py" \
            --input_path "${INPUT_DIR}" \
            --output_path "${OUTPUT_DIR}" \
            --img_h 64 --img_w 64 --workers 8
        ;;
    *)
        echo "[ERROR] Unknown dataset '${DATASET}'. Use: casiab | hid | sustech" >&2
        exit 1
        ;;
esac

echo "[INFO] Done -> ${OUTPUT_DIR}"
