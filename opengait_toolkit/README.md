# Trimmed OpenGait toolkit

This folder is a **minimal** OpenGait checkout used only as the training/evaluation harness.

The HybridDeepGaitSwin definition is **not** authored here. The root file `../hybrid_gait.py` is copied into `opengait/modeling/models/hybrid_gait.py` by `train.sh` and `eval.sh` before every run.

Do not store checkpoints in this tree. Runtime output goes to `output/` and is gitignored.
