# HybridDeepGaitSwin

Final gait-recognition model for this dissertation: a dual-branch encoder that fuses DeepGaitV2 P3D features with a 3D Swin Transformer via **AttentionFusion**, then uses the same 16-part HPP head as DeepGaitV2.

Built on a trimmed [OpenGait](https://github.com/ShiqiYu/OpenGait) toolkit so the hybrid and the CNN baseline share partitions, samplers and Rank-1 evaluators.

The model definition is the root file **`hybrid_gait.py`**. Everything else lives in folders. Training copies that file into the toolkit before launch, so the root script is the source of truth.

---

## Headline Rank-1 (%)

| Metric | DeepGaitV2 | HybridDeepGaitSwin | Δ (pp) |
|---|---:|---:|---:|
| CASIA-B NM | 97.28 | **97.81** | +0.53 |
| CASIA-B BG | 93.52 | **95.24** | +1.72 |
| CASIA-B CL | 76.75 | **82.65** | +5.90 |
| HID (internal) | 90.78 | **93.15** | +2.37 |
| SUSTech1K Overall | 75.16 | **78.48** | +3.32 |

Full condition table: [`results/metrics.csv`](results/metrics.csv). Figures: [`figures/`](figures/). 

**HID protocol:** gallery = first sequence per subject, probe = remaining sequences (`evaluate_HID2022`). **Not** a competition submission.

Checkpoints are **not** stored here. They are written at runtime to `opengait_toolkit/output/` and ignored by git.

---

## Architecture

```
Input silhouette sequence  [N, 1, T, H, W]
        │
        ▼
Shared stem: layer0 → layer1 → layer2 (P3D)
        │
   ┌────┴────┐
   │         │
Deep branch  Swin branch
layer3–4     upsample to 30×20
(P3D)        → SwinTransformer3D → 1×1 projection
   │         │
   └────┬────┘  (spatial sizes aligned)
        ▼
AttentionFusion
  cat(Deep, Swin) → conv stack (squeeze 16)
  → Softmax over 2 branches
  → ω_d ⊙ Deep + ω_s ⊙ Swin
        ▼
Temporal max-pool → HPP (16 parts) → SeparateFCs → BNNecks
        ▼
Triplet loss (margin 0.2) + cross-entropy (scale 16)
```

AttentionFusion is **map-level**. Weights are not two global scalars and are not aligned to HPP parts. Softmax is taken over the branch axis after scores are reshaped to `[N, 2, C, S, H, W]`. HPP runs **after** fusion.

See `figures/architecture.png`, `figures/cnn_vs_hybrid.png` and `figures/attentionfusion.png`.

---

## Repository layout

```
HybridDeepGaitSwin/
├── hybrid_gait.py          ← canonical model
├── train.sh                ← full training + eval
├── eval.sh                 ← standalone evaluation
├── environment.sh
├── requirements.txt
├── configs/                ← dissertation experiment configurations.
├── scripts/                ← preprocess, train-all, sample smoke test
├── figures/                ← dissertation figures
├── results/metrics.csv     ← reported Rank-1 table
└── opengait_toolkit/       ← trimmed OpenGait (no weights)
```

---

## Environment

```bash
bash environment.sh
conda activate hid2026
```

`environment.sh` uses `conda` on `PATH` (or `CONDA_BIN`). It does not hard-code a home directory. PyTorch 2.5.1 + CUDA 12.1 is installed first, then `requirements.txt`.

Alternatively, activate any env that already has those packages and set `PYTHON` if `python` is not on `PATH`.

---

## Data

Obtain CASIA-B, HID 2022 and SUSTech1K under their own licences. This repository does not ship silhouettes or pretrained weights.

Preprocess to OpenGait `.pkl` sequences (pass **your** local directories):

```bash
bash scripts/preprocess.sh --dataset casiab  --input /path/to/CASIA-B-raw    --output /path/to/casiab-pkl
bash scripts/preprocess.sh --dataset hid     --input /path/to/HID2022-raw    --output /path/to/hid-pkl
bash scripts/preprocess.sh --dataset sustech --input /path/to/SUSTech1K-raw  --output /path/to/sustech-pkl
```

HID pretreatment reads `train/` under the raw input directory.

---

## Train and evaluate

```bash
conda activate hid2026

bash train.sh --dataset casiab  --data_dir /path/to/casiab-pkl  --gpu 0
bash train.sh --dataset hid     --data_dir /path/to/hid-pkl     --gpu 0
bash train.sh --dataset sustech --data_dir /path/to/sustech-pkl --gpu 0
```

Or all three, with roots supplied as environment variables (never committed):

```bash
export CASIAB_PKL=/path/to/casiab-pkl
export HID_PKL=/path/to/hid-pkl
export SUSTECH_PKL=/path/to/sustech-pkl
bash scripts/train_all.sh --gpu 0
```

Standalone evaluation (HID automatically uses `HID-train-eval.json`):

```bash
bash eval.sh --dataset hid --data_dir /path/to/hid-pkl \
  --ckpt opengait_toolkit/output/HID/HybridDeepGaitSwin/HybridDeepGaitSwin_HID/checkpoints/<name>.pt \
  --gpu 0
```

Smoke test on a tiny local sample tree:

```bash
bash scripts/train_sample.sh --sample_root /path/to/sample --gpu 0
```

---

## Training configuration (reported runs)

| Dataset | Iterations | Optimiser | Batch (P×K) | Swin depths | P3D layers | Identities | Frames |
|---|---:|---|---|:---:|:---:|---:|---:|
| CASIA-B | 60,000 | AdamW 3e-4, wd 0.02 | 2×2 | [1, 1] | [1, 1, 1, 1] | 74 | 30 |
| HID 2022 | 60,000 | AdamW 3e-4, wd 0.02 | 2×2 | [2, 2] | [1, 4, 4, 1] | 500 | 30 |
| SUSTech1K | 60,000 | AdamW 3e-4, wd 0.02 | 2×2 | [1, 1] | [1, 1, 1, 1] | 250 | 10 |

Shared: AttentionFusion squeeze 16; HPP 16 parts; triplet margin 0.2; CE scale 16; mixed precision.

DeepGaitV2 was trained with SGD. AdamW and the Hybrid's additional parameters are stated confounds.
---

## Reproducibility

- Root `hybrid_gait.py` is copied into the toolkit on every train/eval run.
- Configs use `DATA_ROOT_PLACEHOLDER` / restore-hint tokens; scripts fill them at runtime. No machine paths are stored in YAML.
- `.gitignore` excludes `*.pt`, `output/`, `logs/` and `.tmp/`.
- Reported numbers are in `results/metrics.csv` so they can be checked without rerunning 60k iterations.

---

## Citation

```bibtex
@article{wang2023deepgaitv2,
  title   = {DeepGaitV2: A Powerful Baseline for Gait Recognition},
  author  = {Wang, Jinkai and Nie, Shengzhe and He, Yunlong and Han, Jianfeng and Yu, Shiqi},
  journal = {arXiv preprint arXiv:2303.14613},
  year    = {2023}
}

@inproceedings{fan2023opengait,
  title     = {OpenGait: Revisiting Gait Recognition towards Better Practicality},
  author    = {Fan, Chao and Liang, Junhao and Shen, Chuanfu and Hou, Saihui and Huang, Yongzhen and Yu, Shiqi},
  booktitle = {CVPR},
  year      = {2023}
}
```
