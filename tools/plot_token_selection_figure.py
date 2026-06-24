"""
Generate §4.3 Token Selection figure (2-panel, subfigure).

(a) RoPE hierarchy: 4-bar chart showing EOS >> PAD >> Comma >> Image μ_k
(b) AbsPE manipulation: PixArt smiley PE experiment (imported PDF)

Usage:
    python tools/plot_token_selection_figure.py \
        --json outputs/flux_keyimportance_stats.json \
        --comma_json outputs/pad_vs_comma_ki/pad_vs_comma_ki.json \
        --out_dir paper/figures/generated

If --comma_json is not provided, uses placeholder value (52x uniform).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils.paper_palette import (
    C_RED, C_ORANGE, C_BLUE, C_GRAY,
    PAPER_RCPARAMS, FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT, FS_SMALL,
    FIG_HALF,
    clean_ax, save_fig,
)

plt.rcParams.update(PAPER_RCPARAMS)

EOS_POS = 17
N_TEXT = 512
N_TOTAL = 4608

C_COMMA = "#E8A87C"  # lighter warm tone for comma (distinct from PAD orange)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="outputs/flux_keyimportance_stats.json")
    parser.add_argument("--comma_json", default=None,
                        help="Path to pad_vs_comma_ki.json (optional)")
    parser.add_argument("--out_dir", default="./paper/figures/generated")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    imp = np.array(data["importance"])
    assert len(imp) == N_TOTAL

    uniform = 1.0 / N_TOTAL

    eos_val = imp[EOS_POS]
    pad_avg = imp[EOS_POS + 1:N_TEXT].mean()
    img_avg = imp[N_TEXT:].mean()

    eos_ratio = eos_val / uniform
    pad_ratio = pad_avg / uniform
    img_ratio = img_avg / uniform

    if args.comma_json and os.path.exists(args.comma_json):
        with open(args.comma_json) as f:
            comma_data = json.load(f)
        ki_baseline_1 = np.array(comma_data["ki_baseline"])
        ki_comma_1 = np.array(comma_data["ki_comma200"])
        # Method 1: Same-position cross-experiment comparison
        # Positions 18-114 are PAD in baseline, become comma in comma exp
        comma_end = comma_data.get("comma_end", 115)
        pad_same_pos = ki_baseline_1[18:comma_end].mean()
        comma_same_pos = ki_comma_1[18:comma_end].mean()
        comma_to_pad_ratio = comma_same_pos / pad_same_pos
        # Scale using N=100 PAD value
        comma_ratio = pad_ratio * comma_to_pad_ratio
        print(f"  Same-position ratio: {comma_to_pad_ratio:.3f}, "
              f"comma={comma_ratio:.0f}x uniform (−{(1-comma_to_pad_ratio)*100:.0f}%)")
    else:
        comma_ratio = 29.0
        print(f"  Using placeholder comma value: {comma_ratio:.0f}x")

    # ── Single panel (a): horizontal bar chart (top=EOS, bottom=Image), wide aspect ──
    fig, ax = plt.subplots(figsize=(FIG_HALF * 1.0, FIG_HALF * 0.45), dpi=300)

    categories = ["EOS", "PAD", "Comma", "Image"]
    ratios = [eos_ratio, pad_ratio, comma_ratio, img_ratio]
    # Red → Orange → warm transition → Blue
    colors = [C_RED, C_ORANGE, "#E8A87C", C_BLUE]

    y_pos = np.arange(len(categories))[::-1]  # EOS at top
    bars = ax.barh(y_pos, ratios, color=colors, height=0.6,
                   edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=FS_LABEL)
    ax.set_xlabel(r"Mean $\mu_k$ ($\times$uniform)", fontsize=FS_LABEL)
    ax.set_xscale("log")
    ax.set_xlim(1, eos_ratio * 5)

    for bar, ratio in zip(bars, ratios):
        label = f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×"
        x_offset = ratio * 1.5
        ax.text(x_offset, bar.get_y() + bar.get_height() / 2,
                label, ha="left", va="center",
                fontsize=FS_ANNOT, fontweight="bold",
                color=bar.get_facecolor())

    reduction = (1 - comma_ratio / pad_ratio) * 100

    clean_ax(ax)
    ax.grid(axis="x", alpha=0.1, zorder=0)

    save_fig(fig, args.out_dir, "fig_token_selection_bars.pdf")
    print(f"\n  EOS={eos_ratio:.0f}x  PAD={pad_ratio:.0f}x  "
          f"Comma={comma_ratio:.0f}x  Image={img_ratio:.1f}x")
    print(f"  Reduction PAD→Comma: −{reduction:.0f}%")


if __name__ == "__main__":
    main()
