#!/usr/bin/env bash
# Train HybridDeepGaitSwin on CASIA-B, HID, or SUSTech1K.
#
# Usage:
#   bash train.sh --dataset casiab|hid|sustech --data_dir <pretreated_pkl_root> [--gpu 0] [--resume <ckpt.pt>]
#
# Dataset roots are local to your machine. Pass them as --data_dir; do not commit paths.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${REPO_ROOT}/scripts/lib.sh"

GPU="0"
DATA_DIR=""
DATASET=""
RESUME_CKPT="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2";     shift 2 ;;
        --data_dir) DATA_DIR="$2";    shift 2 ;;
        --gpu)      GPU="$2";         shift 2 ;;
        --resume)   RESUME_CKPT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,8p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${DATASET}" || -z "${DATA_DIR}" ]]; then
    echo "Usage: bash train.sh --dataset casiab|hid|sustech --data_dir <pkl_root> [--gpu N] [--resume ckpt.pt]" >&2
    exit 1
fi

read -r CFG_REL OG_NAME SAVE_NAME TRAIN_PORT EVAL_PORT < <(dataset_meta "${DATASET}")
CFG_TEMPLATE="${REPO_ROOT}/${CFG_REL}"
CKPT_DIR="${OPENGAIT}/output/${OG_NAME}/HybridDeepGaitSwin/${SAVE_NAME}/checkpoints"
RESOLVED_CFG="${TMP_DIR}/${DATASET}_train_$$.yaml"

write_resolved_config "${CFG_TEMPLATE}" "${RESOLVED_CFG}" "${DATA_DIR}" "0" "${RESUME_CKPT}"

echo "============================================================"
echo "  Training HybridDeepGaitSwin"
echo "  Dataset : ${DATASET} (${OG_NAME})"
echo "  Data    : ${DATA_DIR}"
echo "  GPU     : ${GPU}"
echo "  Resume  : ${RESUME_CKPT}"
echo "  Config  : ${CFG_TEMPLATE}"
echo "  Output  : ${CKPT_DIR}"
echo "============================================================"

launch_opengait "${GPU}" "${TRAIN_PORT}" "${RESOLVED_CFG}" train

echo ""
echo "============================================================"
echo "  Training complete. Running evaluation..."
echo "============================================================"

LATEST_CKPT="$(latest_checkpoint "${CKPT_DIR}")"
if [[ -z "${LATEST_CKPT}" ]]; then
    echo "[ERROR] No checkpoint found in ${CKPT_DIR}" >&2
    rm -f "${RESOLVED_CFG}"
    exit 1
fi
echo "  Checkpoint: ${LATEST_CKPT}"

EVAL_CFG="${TMP_DIR}/${DATASET}_eval_$$.yaml"
EVAL_PARTITION=""
if [[ "${DATASET}" == "hid" ]]; then
    EVAL_PARTITION="./datasets/HID/HID-train-eval.json"
fi
write_resolved_config "${CFG_TEMPLATE}" "${EVAL_CFG}" "${DATA_DIR}" "${LATEST_CKPT}" "0" "${EVAL_PARTITION}"

launch_opengait "${GPU}" "${EVAL_PORT}" "${EVAL_CFG}" test

rm -f "${RESOLVED_CFG}" "${EVAL_CFG}"
echo ""
echo "  Done."
