#!/usr/bin/env bash
# Shared helpers for HybridDeepGaitSwin train / eval / preprocess.
# Sourced by scripts in this directory and by the root train.sh / eval.sh.

set -euo pipefail

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LIB_DIR}/.." && pwd)"
OPENGAIT="${REPO_ROOT}/opengait_toolkit"
TMP_DIR="${REPO_ROOT}/.tmp"

mkdir -p "${TMP_DIR}"

resolve_python() {
    if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
        echo "${PYTHON}"
        return
    fi
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "${CONDA_PREFIX}/bin/python"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    echo "[ERROR] No Python interpreter found. Activate the conda env (hid2026) or set PYTHON." >&2
    exit 1
}

sync_hybrid_model() {
    local dest="${OPENGAIT}/opengait/modeling/models/hybrid_gait.py"
    cp "${REPO_ROOT}/hybrid_gait.py" "${dest}"
}

dataset_meta() {
    # Prints: config_rel  open_gait_dataset_name  save_name  train_port  eval_port
    case "$1" in
        casiab)
            echo "configs/casiab.yaml CASIA-B HybridDeepGaitSwin_CASIAB 29714 29715"
            ;;
        hid)
            echo "configs/hid.yaml HID HybridDeepGaitSwin_HID 29710 29711"
            ;;
        sustech)
            echo "configs/sustech.yaml SUSTech1K HybridDeepGaitSwin_SUSTech1K 29712 29713"
            ;;
        *)
            echo "[ERROR] Unknown dataset '$1'. Use: casiab | hid | sustech" >&2
            exit 1
            ;;
    esac
}

write_resolved_config() {
    local template="$1"
    local out="$2"
    local data_root="$3"
    local eval_restore="$4"
    local train_restore="$5"
    local partition="${6:-}"
    local python_bin
    python_bin="$(resolve_python)"
    "${python_bin}" - "${template}" "${out}" "${data_root}" "${eval_restore}" "${train_restore}" "${partition}" <<'PY'
import sys
from pathlib import Path

src, dst, data_root, eval_restore, train_restore, partition = sys.argv[1:7]
text = Path(src).read_text()
text = text.replace("DATA_ROOT_PLACEHOLDER", data_root)
text = text.replace("EVAL_RESTORE_HINT", eval_restore)
text = text.replace("TRAIN_RESTORE_HINT", train_restore)
if partition:
    text = text.replace("PARTITION_PLACEHOLDER", partition)
    # Full HID eval uses the internal train-split protocol file.
    text = text.replace("./datasets/HID/HID.json", partition)
Path(dst).write_text(text)
PY
}

launch_opengait() {
    local gpu="$1"
    local port="$2"
    local cfg="$3"
    local phase="$4"
    local python_bin
    python_bin="$(resolve_python)"
    sync_hybrid_model
    (
        cd "${OPENGAIT}"
        if [[ "${phase}" == "train" ]]; then
            CUDA_VISIBLE_DEVICES="${gpu}" \
            "${python_bin}" -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port="${port}" \
                opengait/main.py \
                --cfgs "${cfg}" \
                --phase "${phase}" \
                --log_to_file
        else
            CUDA_VISIBLE_DEVICES="${gpu}" \
            "${python_bin}" -m torch.distributed.launch \
                --nproc_per_node=1 \
                --master_port="${port}" \
                opengait/main.py \
                --cfgs "${cfg}" \
                --phase "${phase}"
        fi
    )
}

latest_checkpoint() {
    local ckpt_dir="$1"
    ls -t "${ckpt_dir}"/*.pt 2>/dev/null | head -1 || true
}
