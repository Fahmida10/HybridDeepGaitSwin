# Reported Rank-1 results

`metrics.csv` is the dissertation comparison of **HybridDeepGaitSwin** against **DeepGaitV2**.

All values are Rank-1 accuracy (%). Deltas are percentage points (hybrid minus DeepGaitV2).

| Metric | DeepGaitV2 | HybridDeepGaitSwin | Δ (pp) |
|---|---:|---:|---:|
| CASIA-B NM | 97.28 | **97.81** | +0.53 |
| CASIA-B BG | 93.52 | **95.24** | +1.72 |
| CASIA-B CL | 76.75 | **82.65** | +5.90 |
| HID Rank-1 (internal) | 90.78 | **93.15** | +2.37 |
| SUSTech1K Overall | 75.16 | **78.48** | +3.32 |

HID uses an internal protocol (gallery = first sequence per subject, probe = remainder). It is **not** an official HID 2022 challenge score.

Weights are not stored in this repository. Re-running `train.sh` writes checkpoints under `opengait_toolkit/output/` (gitignored).
