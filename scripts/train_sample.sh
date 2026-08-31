#!/usr/bin/env bash
# Smoke-test HybridDeepGaitSwin on tiny sample partitions (200 iterations).
#
# Usage:
#   bash scripts/train_sample.sh --sample_root <dir> [--gpu 0]
#
# Expected layout under --sample_root:
#   CASIA-B/pkl/   CASIA-B/partition/CASIA-B-sample.json
#   HID/pkl/       HID/partition/HID-sample.json
#   HID/partition/HID-sample-train-eval.json
#   SUSTech1K/pkl/ SUSTech1K/partition/SUSTech1K-sample.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${SCRIPT_DIR}/lib.sh"

GPU="0"
SAMPLE_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --sample_root) SAMPLE_ROOT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${SAMPLE_ROOT}" ]]; then
    echo "Usage: bash scripts/train_sample.sh --sample_root <dir> [--gpu N]" >&2
    exit 1
fi

train_one() {
    local ds_name="$1"
    local port_train="$2"
    local port_eval="$3"
    local key
    case "${ds_name}" in
        CASIA-B) key="casiab" ;;
        HID) key="hid" ;;
        SUSTech1K) key="sustech" ;;
    esac

    local data_dir="${SAMPLE_ROOT}/${ds_name}/pkl"
    local partition="${SAMPLE_ROOT}/${ds_name}/partition/${ds_name}-sample.json"
    local cfg_template="${REPO_ROOT}/configs/sample/${ds_name}/hybrid.yaml"
    local save_name
    save_name="$(awk '/^trainer_cfg:/{f=1} f&&/save_name:/{print $2; exit}' "${cfg_template}")"
    local resolved_cfg="${TMP_DIR}/sample_${key}_train_$$.yaml"

    write_resolved_config "${cfg_template}" "${resolved_cfg}" "${data_dir}" "0" "0" "${partition}"

    local ckpt_dir="${OPENGAIT}/output/${ds_name}/HybridDeepGaitSwin/${save_name}/checkpoints"

    echo ""
    echo "============================================================"
    echo "  Sample training: HybridDeepGaitSwin | ${ds_name}"
    echo "  Data       : ${data_dir}"
    echo "  Partition  : ${partition}"
    echo "  GPU        : ${GPU}"
    echo "============================================================"

    launch_opengait "${GPU}" "${port_train}" "${resolved_cfg}" train

    local latest_ckpt
    latest_ckpt="$(latest_checkpoint "${ckpt_dir}")"
    if [[ -z "${latest_ckpt}" ]]; then
        echo "[ERROR] No checkpoint in ${ckpt_dir}" >&2
        rm -f "${resolved_cfg}"
        return 1
    fi

    local eval_cfg="${TMP_DIR}/sample_${key}_eval_$$.yaml"
    local eval_partition="${partition}"
    if [[ "${ds_name}" == "HID" ]]; then
        eval_partition="${SAMPLE_ROOT}/HID/partition/HID-sample-train-eval.json"
    fi
    write_resolved_config "${cfg_template}" "${eval_cfg}" "${data_dir}" "${latest_ckpt}" "0" "${eval_partition}"

    launch_opengait "${GPU}" "${port_eval}" "${eval_cfg}" test

    rm -f "${resolved_cfg}" "${eval_cfg}"
    echo "  [OK] ${ds_name} complete"
}

echo "============================================================"
echo "  HybridDeepGaitSwin sample training (all 3 datasets)"
echo "============================================================"

train_one "CASIA-B" 2980 2981
train_one "HID" 2982 2983
train_one "SUSTech1K" 2984 2985

echo ""
echo "  All three sample datasets trained and evaluated."
