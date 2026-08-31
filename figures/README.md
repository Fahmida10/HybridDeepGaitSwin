# Figures

Dissertation figures for HybridDeepGaitSwin. Machine paths and lab-notebook labels are not used.

| File | What it shows |
|---|---|
| `architecture.png` | HybridDeepGaitSwin network diagram |
| `cnn_vs_hybrid.png` | DeepGaitV2 (CNN only) vs HybridDeepGaitSwin (P3D + Swin + AttentionFusion) |
| `attentionfusion.png` | Map-level Softmax mix vs two global scalars (not used) |
| `training_pipeline.png` | Preprocess → train → save (60k iterations on every dataset) |
| `evaluation_pipeline.png` | Checkpoint → embeddings → Rank-1, with dataset protocols |
| `headline_rank1.png` | Primary Rank-1 comparison |
| `casiab_detail.png` | CASIA-B NM/BG/CL and view-angle curves |
| `hid_sustech_training.png` | HID, SUSTech1K conditions, and training loss |

Regenerate the two pipeline diagrams:

```bash
python scripts/make_pipeline_figures.py
```
