#!/usr/bin/env bash
# Evaluate a trained HybridDeepGaitSwin checkpoint.
#
# Usage:
#   bash eval.sh --dataset casiab|hid|sustech --data_dir <pretreated_pkl_root> --ckpt <checkpoint.pt> [--gpu 0]
#
# HID evaluation swaps the partition to HID-train-eval.json:
# gallery = first sequence per subject, probe = remainder (evaluate_HID2022).
# That is an internal protocol, not an official challenge submission.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${REPO_ROOT}/scripts/lib.sh"

GPU="0"
DATA_DIR=""
DATASET=""
CKPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET="$2";  shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --ckpt)     CKPT="$2";     shift 2 ;;
        --gpu)      GPU="$2";      shift 2 ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${DATASET}" || -z "${DATA_DIR}" || -z "${CKPT}" ]]; then
    echo "Usage: bash eval.sh --dataset casiab|hid|sustech --data_dir <pkl_root> --ckpt <checkpoint.pt> [--gpu N]" >&2
    exit 1
fi

read -r CFG_REL OG_NAME SAVE_NAME TRAIN_PORT EVAL_PORT < <(dataset_meta "${DATASET}")
CFG_TEMPLATE="${REPO_ROOT}/${CFG_REL}"
EVAL_CFG="${TMP_DIR}/${DATASET}_eval_only_$$.yaml"

EVAL_PARTITION=""
if [[ "${DATASET}" == "hid" ]]; then
    EVAL_PARTITION="./datasets/HID/HID-train-eval.json"
fi
write_resolved_config "${CFG_TEMPLATE}" "${EVAL_CFG}" "${DATA_DIR}" "${CKPT}" "0" "${EVAL_PARTITION}"

echo "============================================================"
echo "  Evaluating HybridDeepGaitSwin"
echo "  Dataset    : ${DATASET} (${OG_NAME})"
echo "  Data       : ${DATA_DIR}"
echo "  Checkpoint : ${CKPT}"
echo "  GPU        : ${GPU}"
if [[ "${DATASET}" == "hid" ]]; then
    echo "  Protocol   : internal HID train-split Rank-1 (not a challenge submission)"
fi
echo "============================================================"

launch_opengait "${GPU}" "${EVAL_PORT}" "${EVAL_CFG}" test

rm -f "${EVAL_CFG}"
echo "  Done."
