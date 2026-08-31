#!/usr/bin/env python3
"""Draw path-free training and evaluation pipeline figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, title, body, facecolor, edgecolor):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=1.4,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color="#1a1a1a")
    ax.text(x + w / 2, y + 0.055, body, ha="center", va="bottom",
            fontsize=8.2, color="#333333", wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.2, color="#222222",
        shrinkA=0, shrinkB=0,
    ))


def training_pipeline():
    fig, ax = plt.subplots(figsize=(14.2, 5.6), dpi=160)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.set_title("HybridDeepGaitSwin — training pipeline", fontsize=15,
                 fontweight="bold", pad=8, color="#111111")

    steps = [
        (0.25, 2.55, 1.55, 1.55, "#dbeafe", "#1d4ed8",
         "1. Raw silhouettes", "CASIA-B · HID\nSUSTech1K"),
        (2.05, 2.55, 1.55, 1.55, "#dcfce7", "#15803d",
         "2. Pretreat", "binary silhouettes\nfixed 64×64"),
        (3.85, 2.55, 1.55, 1.55, "#ffedd5", "#c2410c",
         "3. Sequences", "data/*.pkl\nOpenGait layout"),
        (5.65, 2.55, 1.55, 1.55, "#ede9fe", "#6d28d9",
         "4. Sampler", "TripletSampler\nP×K = 2×2"),
        (7.45, 2.55, 1.70, 1.55, "#dbeafe", "#1e3a8a",
         "5. Forward", "HybridDeepGaitSwin\nAttentionFusion"),
        (9.40, 2.55, 1.50, 1.55, "#fce7f3", "#be185d",
         "6. Losses", "triplet 0.2\nCE scale 16"),
        (11.15, 2.55, 1.35, 1.55, "#ccfbf1", "#0f766e",
         "7. AdamW", "3e-4 · wd 0.02\ndifferential LR"),
        (12.70, 2.55, 1.25, 1.55, "#ffedd5", "#9a3412",
         "8. Save", "output/\nevery 10k"),
    ]
    for x, y, w, h, fc, ec, title, body in steps:
        _box(ax, x, y, w, h, title, body, fc, ec)
    for i in range(len(steps) - 1):
        x, y, w, h = steps[i][0], steps[i][1], steps[i][2], steps[i][3]
        nx = steps[i + 1][0]
        _arrow(ax, x + w + 0.02, y + h / 2, nx - 0.02, y + h / 2)

    band = FancyBboxPatch(
        (0.25, 0.35), 13.70, 1.70,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1, linestyle="--",
        facecolor="#f8fafc", edgecolor="#64748b",
    )
    ax.add_patch(band)
    ax.text(7.10, 1.78, "Training iterations (matched budget)", ha="center",
            va="top", fontsize=11, fontweight="bold", color="#1e293b")

    datasets = [
        (1.4, "CASIA-B", "60,000"),
        (7.1, "HID 2022", "60,000"),
        (12.5, "SUSTech1K", "60,000"),
    ]
    for x, name, iters in datasets:
        ax.text(x, 1.15, name, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#0f172a")
        ax.text(x, 0.70, f"{iters} iterations", ha="center", va="center",
                fontsize=10, color="#334155")

    fig.tight_layout()
    path = OUT / "training_pipeline.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def evaluation_pipeline():
    fig, ax = plt.subplots(figsize=(14.2, 7.4), dpi=160)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.set_title("HybridDeepGaitSwin — evaluation pipeline", fontsize=15,
                 fontweight="bold", pad=8, color="#111111")

    steps = [
        (0.40, 5.15, 2.30, 1.45, "#dbeafe", "#1d4ed8",
         "1. Load weights", "output/.../checkpoints/\n(runtime only)"),
        (3.15, 5.15, 2.30, 1.45, "#dbeafe", "#1e3a8a",
         "2. Inference", "full sequence\nHybridDeepGaitSwin"),
        (5.90, 5.15, 2.30, 1.45, "#ede9fe", "#6d28d9",
         "3. Embeddings", "HPP 16 parts\nSeparateFC"),
        (8.65, 5.15, 2.30, 1.45, "#ffedd5", "#c2410c",
         "4. Match", "mean part-wise\nEuclidean distance"),
        (11.40, 5.15, 2.40, 1.45, "#dcfce7", "#15803d",
         "5. Rank-1", "nearest gallery ID\nmust be correct"),
    ]
    for x, y, w, h, fc, ec, title, body in steps:
        _box(ax, x, y, w, h, title, body, fc, ec)
    for i in range(len(steps) - 1):
        x, y, w, h = steps[i][0], steps[i][1], steps[i][2], steps[i][3]
        nx = steps[i + 1][0]
        _arrow(ax, x + w + 0.04, y + h / 2, nx - 0.04, y + h / 2)

    panels = [
        (0.40, 0.40, 4.20, 4.20, "#ecfdf5", "#047857",
         "CASIA-B",
         "OpenGait indoor protocol\n\n"
         "Probe vs gallery by view\n\n"
         "Report\n  NM Rank-1\n  BG Rank-1\n  CL Rank-1"),
        (5.00, 0.40, 4.20, 4.20, "#ecfdf5", "#047857",
         "SUSTech1K",
         "Outdoor conditions\n\n"
         "Silhouette modality only\n\n"
         "Report Overall plus\n  Normal, Bag, Clothing\n  Carrying, Umbrella\n  Uniform, Occlusion, Night"),
        (9.60, 0.40, 4.20, 4.20, "#ecfdf5", "#047857",
         "HID 2022 (internal)",
         "Not a challenge submission\n\n"
         "gallery = first sequence\n  per training subject\n"
         "probe = remaining sequences\n\n"
         "Evaluator: evaluate_HID2022\n"
         "Partition: HID-train-eval.json"),
    ]
    for x, y, w, h, fc, ec, title, body in panels:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.5, facecolor=fc, edgecolor=ec,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
                fontsize=12.5, fontweight="bold", color="#064e3b")
        ax.text(x + 0.28, y + h - 0.70, body, ha="left", va="top",
                fontsize=9.2, color="#14532d", linespacing=1.35)

    fig.tight_layout()
    path = OUT / "evaluation_pipeline.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def relabel_result_figures():
    """Cover lab-style 'Summary N' titles on dissertation result figures."""
    from PIL import Image, ImageDraw, ImageFont

    jobs = [
        ("headline_rank1.png", "Headline Rank-1: HybridDeepGaitSwin vs DeepGaitV2"),
        ("casiab_detail.png", "CASIA-B Rank-1 by condition and view"),
        ("hid_sustech_training.png", "HID, SUSTech1K conditions, and training loss"),
    ]
    font_path = Path(plt.matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(str(font_path), 52)

    for name, title in jobs:
        path = OUT / name
        im = Image.open(path).convert("RGBA")
        draw = ImageDraw.Draw(im)
        # Cover the original matplotlib suptitle ("Summary N — ...") as well.
        bar_h = max(150, int(im.height * 0.072))
        draw.rectangle([0, 0, im.width, bar_h], fill=(255, 255, 255, 255))
        draw.text((im.width / 2, bar_h / 2), title, fill=(17, 24, 39, 255),
                  font=font, anchor="mm")
        im.save(path)
        print("relabelled", path)


if __name__ == "__main__":
    training_pipeline()
    evaluation_pipeline()
    # Result-figure titles are already cleaned in figures/. Relabel only
    # when regenerating from the original dissertation PNGs:
    #   python scripts/make_pipeline_figures.py --relabel
    import sys
    if "--relabel" in sys.argv:
        relabel_result_figures()
