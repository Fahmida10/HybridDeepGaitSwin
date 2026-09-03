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

### SUSTech1K Rank-1 (%) by Condition

| Condition | SwinGait | DeepGaitV2 | HybridDeepGaitSwin | Δ vs Deep |
|---|---:|---:|---:|---:|
| Overall | 69.18 | 75.16 | **78.48** | **+3.32** |
| Normal | 73.93 | 79.09 | **80.36** | **+1.27** |
| Bag | 73.16 | 77.87 | **79.64** | **+1.77** |
| Clothing | 31.84 | 42.87 | **50.00** | **+7.13** |
| Carrying | 69.72 | 75.26 | **77.74** | **+2.48** |
| Umbrella | 68.11 | 75.88 | **78.82** | **+2.94** |
| Uniform | 67.82 | 74.70 | **77.28** | **+2.58** |
| Occlusion | 70.69 | 77.20 | **81.37** | **+4.17** |
| Night | 22.69 | 24.82 | **32.88** | **+8.06** |

**Δ vs Deep** = HybridDeepGaitSwin − DeepGaitV2 in percentage points. Higher Rank-1 is better.
