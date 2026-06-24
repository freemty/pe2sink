"""
Generate paper-ready figures from experiment outputs.

Usage:
    python tools/plot_paper_figures.py [--exp_root PATH] [--out_dir PATH] [--sink_value_json PATH]

Data sources:
    exp00d (attn_sink_task/stats)  → fig_sink_time_log, fig_sink_layer, fig_flux_key_importance
    exp05a (sink_curves JSON)      → fig_norm_paradox, fig_cheating_strategy,
                                     fig_same_norm_diff_dist, fig_sink_value_info
"""
import glob
import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from src.utils.paper_palette import (
    C_RED, C_ORANGE, C_BLUE, C_GRAY, C_DEEP, C_BG,
    TOKEN_COLORS, TOKEN_LABELS, PAPER_RCPARAMS,
    FIG_FULL, FIG_HALF,
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT, FS_SMALL,
    clean_ax as _clean_ax, layer_ticks as _layer_ticks,
    plot_sink_vs_rand as _plot_sink_vs_rand,
    save_fig as _save_fig, check_keys as _check_keys,
)

plt.rcParams.update(PAPER_RCPARAMS)

# Token type classification for FLUX.1
# Token 17 = EOS, 49-73 = semantic text, 74-511 = PAD, 512+ = image
def token_type(tid):
    if tid == 17:
        return "eos"
    if 49 <= tid <= 73:
        return "text"
    if (0 <= tid <= 16) or (18 <= tid <= 48) or (74 <= tid <= 511):
        return "pad"
    return "image"

EPS = 1e-8  # Safe divisor for ratio computation

# RoPE axis boundary channels — shared by all band heatmap functions.
# Each entry: list of channel indices where the RoPE axis changes.
# Derived from each model's controller _get_rope_band_config().
ROPE_AXIS_SEGS = {
    "FLUX":    [16, 72],   # text_id(0-15)|H(16-71)|W(72-127)  axes_dim=(16,56,56)
    "Wan":     [44, 86],   # t|h at 44, h|w at 86  (128-dim, 3 axes)
    "Z-Image": [32, 80],   # t(0-31)|h(32-79)|w(80-127)  axes_dims=[32,48,48]
    "LTX":     [44, 86],   # pad(0-1)+t(2-43)|h(44-85)|w(86-127)  128-dim, 3×42
    "Cosmos":  [44, 86],   # same 3D RoPE layout as Wan
}


def parse_top100(filepath):
    tokens = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Type:") or line.startswith("====") or line.startswith("Step("):
                continue
            for pair in line.split(", "):
                parts = pair.split(":")
                if len(parts) == 2:
                    try:
                        tokens[int(parts[0])] = float(parts[1])
                    except ValueError:
                        pass
    return tokens


def plot_fig_sink_time(exp_root, out_dir):
    """Fig 2: Log-scale heatmap — max Key Importance across layers x steps."""
    stats_dir = os.path.join(exp_root, "attn_sink_task/stats/per_step")
    if not os.path.exists(stats_dir):
        return

    layer_dirs = sorted(
        [d for d in os.listdir(stats_dir) if d.startswith("layer_")],
        key=lambda x: int(x.split("_")[1]))

    layers, steps, data = [], None, []
    for ld in layer_dirs:
        layers.append(int(ld.split("_")[1]))
        lpath = os.path.join(stats_dir, ld)
        vals = {}
        for sf in os.listdir(lpath):
            if "keyimportance_softmax_stats" not in sf:
                continue
            step = int(sf.split("_")[3])
            with open(os.path.join(lpath, sf)) as f:
                for line in f:
                    if line.strip().startswith("max:"):
                        vals[step] = float(line.strip().split("max:")[1])
                        break
        if steps is None:
            steps = sorted(vals.keys())
        data.append([vals.get(s, 1e-6) for s in steps])

    arr = np.clip(np.array(data), 1e-4, None)

    fig, ax = plt.subplots(figsize=(FIG_HALF, 2.2))
    im = ax.imshow(arr, aspect="auto", cmap="magma",
                   norm=mcolors.LogNorm(vmin=arr[arr > 0].min(), vmax=arr.max()),
                   interpolation="nearest")
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Layer Index")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_yticks(range(0, len(layers), 2))
    ax.set_yticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=FS_SMALL)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Max Key Importance (log)")
    _save_fig(fig, out_dir, "fig_sink_time_log.pdf")


def plot_fig_sink_layer(exp_root, out_dir):
    """Fig 3: Line plot — sink concentration across layers. RED accent."""
    stats_dir = os.path.join(exp_root, "attn_sink_task/stats/over_steps")
    if not os.path.exists(stats_dir):
        return
    layer_files = sorted(
        [f for f in os.listdir(stats_dir) if f.startswith("layer_") and f.endswith("_softmax.txt")],
        key=lambda x: int(x.split("_")[1]))
    layers, vals = [], []
    for lf in layer_files:
        layers.append(int(lf.split("_")[1]))
        tokens = parse_top100(os.path.join(stats_dir, lf))
        vals.append(max(tokens.values()) if tokens else 0)

    fig, ax = plt.subplots(figsize=(FIG_HALF, 2.2))
    ax.plot(range(len(layers)), vals, "o-", color=C_RED, markersize=1.8, linewidth=0.9)
    ax.fill_between(range(len(layers)), vals, alpha=0.10, color=C_RED)
    ax.set_xticks(range(0, len(layers), 2))
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=FS_SMALL)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-1 Token Key Importance")
    _clean_ax(ax)
    _save_fig(fig, out_dir, "fig_sink_layer.pdf")


def plot_fig_key_importance_bar(exp_root, out_dir, flux_image=None):
    """Fig 3(a): Top-30 Key Importance bar chart (pure bar, no image).

    Simple ranked bar chart colored by token type.
    Paired with 3D spatial chart as subfigure (b).
    """
    gf = None
    for candidate in [
        os.path.join(exp_root, "attn_sink_task/stats/over_steps/"
                     "GLOBAL_over_steps_avg_top100_softmax.txt"),
        "outputs/flux_GLOBAL_top100_softmax.txt",
    ]:
        if os.path.exists(candidate):
            gf = candidate
            break
    if gf is None:
        print("  Skipping fig_flux_key_importance: no GLOBAL txt found")
        return

    tokens = parse_top100(gf)
    top = sorted(tokens.items(), key=lambda x: -x[1])[:30]
    ids = [t[0] for t in top]
    vals = [t[1] for t in top]
    bar_colors = [TOKEN_COLORS[token_type(tid)] for tid in ids]
    print(f"  Using FLUX key importance from {gf}")

    fig, ax = plt.subplots(figsize=(FIG_HALF, 2.4), dpi=300)

    x_pos = np.arange(len(ids))
    ax.bar(x_pos, vals, color=bar_colors, alpha=0.9, width=0.72,
           edgecolor="none")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(t) for t in ids], rotation=45, ha="right",
                       fontsize=FS_SMALL)
    ax.set_ylabel("Key Importance")
    ax.set_xlabel("Token Index (ranked by importance)", fontsize=FS_LABEL)
    _clean_ax(ax)
    ax.tick_params(axis="y", labelsize=FS_SMALL)
    ax.tick_params(axis="x", labelsize=FS_SMALL, pad=1)

    # EOS annotation
    if FLUX_EOS_IDX in ids:
        eos_pos = ids.index(FLUX_EOS_IDX)
        ax.annotate(
            "<EOS>", xy=(eos_pos, vals[eos_pos]),
            xytext=(eos_pos + 5, vals[eos_pos] * 0.72),
            fontsize=FS_ANNOT, color=C_RED, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=0.8),
        )

    # Legend
    patches = [
        mpatches.Patch(color=C_RED, label="<EOS>"),
        mpatches.Patch(color=C_ORANGE, label="<PAD>"),
        mpatches.Patch(color=C_BLUE, label="Semantic Text"),
        mpatches.Patch(color=C_GRAY, label="Image"),
    ]
    ax.legend(handles=patches, fontsize=FS_SMALL, loc="upper right",
              framealpha=0.85, handlelength=1.0, handletextpad=0.3,
              borderpad=0.3, labelspacing=0.2)

    _save_fig(fig, out_dir, "fig_flux_key_importance.pdf")


def plot_fig_norm_paradox(data, out_dir):
    """Fig 6: The Norm Paradox — comparable norms, divergent attention."""
    if not _check_keys(data, ["layer_list", "K_sink_curve", "K_rand_curve",
                               "Attention_sink_curve", "Attention_rand_curve"],
                       "fig_norm_paradox"):
        return

    layer_list = data["layer_list"]
    k_sink_l2 = np.array(data["K_sink_curve"])
    k_rand_l2 = np.array(data["K_rand_curve"])
    attn_sink = np.array(data["Attention_sink_curve"])
    attn_rand = np.array(data["Attention_rand_curve"])
    n = len(layer_list)
    x = range(n)

    k_ratio = k_sink_l2 / np.clip(k_rand_l2, EPS, None)
    attn_ratio = attn_sink / np.clip(attn_rand, EPS, None)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(FIG_FULL, 2.0), dpi=300)

    # --- Left: K L2-norm ---
    _plot_sink_vs_rand(ax_left, x, k_sink_l2, k_rand_l2,
                       "Key L2-norm", "Key Norms (comparable)", "lower right")
    tick_step = max(1, n // 8)
    ax_left.set_xticks(range(0, n, tick_step))
    ax_left.set_xticklabels([str(layer_list[i]) for i in range(0, n, tick_step)], fontsize=FS_SMALL)

    ax_left_twin = ax_left.twinx()
    ax_left_twin.bar(x, k_ratio, alpha=0.12, color=C_GRAY, width=0.6, zorder=0)
    ax_left_twin.axhline(1.0, color=C_GRAY, ls="--", lw=0.5, alpha=0.5)
    ax_left_twin.set_ylabel("Sink / Random", fontsize=FS_TICK, color=C_GRAY)
    ax_left_twin.set_ylim(0.5, 2.0)
    ax_left_twin.tick_params(labelsize=FS_SMALL, colors=C_GRAY)
    ax_left_twin.spines["top"].set_visible(False)

    # --- Right: Attention Score ---
    _plot_sink_vs_rand(ax_right, x, attn_sink, attn_rand,
                       "Key Importance", "Attention Score (divergent)")
    ax_right.set_xticks(range(0, n, tick_step))
    ax_right.set_xticklabels([str(layer_list[i]) for i in range(0, n, tick_step)], fontsize=FS_SMALL)

    ax_right_twin = ax_right.twinx()
    ax_right_twin.bar(x, attn_ratio, alpha=0.12, color=C_RED, width=0.6, zorder=0)
    ax_right_twin.set_ylabel("Sink / Random", fontsize=FS_TICK, color=C_RED)
    ax_right_twin.tick_params(labelsize=FS_SMALL, colors=C_RED)
    ax_right_twin.spines["top"].set_visible(False)

    peak_idx = int(np.argmax(attn_ratio))
    ax_right_twin.annotate(
        f"{attn_ratio[peak_idx]:.0f}\u00d7",
        xy=(peak_idx, attn_ratio[peak_idx]),
        fontsize=FS_LABEL, fontweight="bold", color=C_RED, ha="center",
        xytext=(0, 6), textcoords="offset points",
    )

    _save_fig(fig, out_dir, "fig_norm_paradox.pdf")


def plot_fig_same_norm_diff_dist(data, out_dir):
    """Fig: Same Norm, Different Distribution.

    3-column layout: K Sink heatmap | K Random heatmap | dual-axis line chart
    Heatmaps annotated with High/Low freq direction.
    Line chart shows K L2-norm (nearly equal) and Attention Score (divergent).
    """
    if not _check_keys(data, ["k_sink_heatmap", "k_rand_heatmap", "layer_list",
                               "K_sink_curve", "K_rand_curve",
                               "Attention_sink_curve", "Attention_rand_curve"],
                       "fig_same_norm_diff_dist"):
        return

    k_sink_hm = np.array(data["k_sink_heatmap"])   # (num_layers, 128)
    k_rand_hm = np.array(data["k_rand_heatmap"])
    layer_list = data["layer_list"]
    k_sink_l2 = np.array(data["K_sink_curve"])
    k_rand_l2 = np.array(data["K_rand_curve"])
    attn_sink = np.array(data["Attention_sink_curve"])
    attn_rand = np.array(data["Attention_rand_curve"])

    n_layers = len(layer_list)
    n_channels = k_sink_hm.shape[1]

    vmin = min(k_sink_hm.min(), k_rand_hm.min())
    vmax = max(k_sink_hm.max(), k_rand_hm.max())
    tick_pos, tick_lbl = _layer_ticks(layer_list)

    fig = plt.figure(figsize=(FIG_FULL, 2.4), dpi=300)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.0, 1.0, 0.03, 1.1],
        wspace=0.06,
    )

    # --- (a) K Sink heatmap ---
    ax_ks = fig.add_subplot(gs[0, 0])
    im = ax_ks.imshow(
        k_sink_hm.T, aspect="auto", cmap="magma",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax_ks.set_title("(a) Sink Key", fontweight="bold")
    ax_ks.set_ylabel("Channel index")
    ax_ks.set_xlabel("Layer")
    ax_ks.set_xticks(tick_pos)
    ax_ks.set_xticklabels(tick_lbl)
    ax_ks.text(-0.02, 1.0, "High freq", transform=ax_ks.transAxes,
               fontsize=FS_SMALL, color=C_GRAY, ha="right", va="bottom", style="italic")
    ax_ks.text(-0.02, 0.0, "Low freq", transform=ax_ks.transAxes,
               fontsize=FS_SMALL, color=C_GRAY, ha="right", va="top", style="italic")
    safe_start = int(n_channels * 0.7)
    ax_ks.axhline(y=safe_start - 0.5, color="white", linewidth=0.6,
                  linestyle="--", alpha=0.5)

    # --- (b) K Random heatmap ---
    ax_kr = fig.add_subplot(gs[0, 1])
    ax_kr.imshow(
        k_rand_hm.T, aspect="auto", cmap="magma",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax_kr.set_title("(b) Random Image Key", fontweight="bold")
    ax_kr.set_xlabel("Layer")
    ax_kr.set_xticks(tick_pos)
    ax_kr.set_xticklabels(tick_lbl)
    ax_kr.set_yticks([])
    ax_kr.axhline(y=safe_start - 0.5, color="white", linewidth=0.6,
                  linestyle="--", alpha=0.5)

    # Shared colorbar (thin, between heatmaps and line chart)
    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=FS_SMALL)
    cbar.set_label("$\\|\\mathbf{k}\\|$", fontsize=FS_TICK, rotation=0, labelpad=8, y=0.5)

    # --- (c) Dual-axis line chart ---
    ax_norm = fig.add_subplot(gs[0, 3])
    x = np.arange(n_layers)

    # Left Y: K L2-norm (nearly overlapping → same norm)
    ln1 = ax_norm.plot(x, k_sink_l2, "o-", color=C_RED, lw=0.9, ms=2,
                       label="Sink $\\|K\\|_2$", alpha=0.85)
    ln2 = ax_norm.plot(x, k_rand_l2, "o-", color=C_BLUE, lw=0.9, ms=2,
                       label="Random $\\|K\\|_2$", alpha=0.85)
    ax_norm.set_ylabel("Key $\\ell_2$-norm")
    ax_norm.set_xlabel("Layer")
    ax_norm.set_xticks(tick_pos)
    ax_norm.set_xticklabels(tick_lbl)
    ax_norm.tick_params(axis="y", labelsize=FS_SMALL)

    # Right Y: Attention Score (divergent → different attention)
    ax_attn = ax_norm.twinx()
    ln3 = ax_attn.plot(x, attn_sink, "s--", color=C_RED, lw=1.2, ms=2.5,
                       label="Sink Attention", zorder=5)
    ln4 = ax_attn.plot(x, attn_rand, "s--", color=C_BLUE, lw=1.2, ms=2.5,
                       label="Random Attention", zorder=5)
    ax_attn.set_ylabel("Key Importance")
    ax_attn.tick_params(axis="y", labelsize=FS_SMALL)

    # Combined legend
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax_norm.legend(lns, labs, fontsize=FS_SMALL, loc="upper left", framealpha=0.85,
                   ncol=1, handlelength=1.2)

    ax_norm.set_title("(c) Norm vs Attention", fontweight="bold")
    _clean_ax(ax_norm)

    # Annotation: highlight the paradox
    if len(attn_sink) > 0:
        peak_idx = int(np.argmax(attn_sink))
        peak_val = attn_sink[peak_idx]
        rand_at_peak = attn_rand[peak_idx] if peak_idx < len(attn_rand) else 0
        if rand_at_peak > 0:
            ratio = peak_val / rand_at_peak
            ax_attn.annotate(
                f"{ratio:.0f}$\\times$",
                xy=(peak_idx, peak_val), xytext=(peak_idx - 2, peak_val * 0.75),
                fontsize=FS_LABEL, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
            )

    _save_fig(fig, out_dir, "fig_same_norm_diff_dist.pdf")


def plot_fig_mechanism_composite(data, out_dir, head_dim=128, base=10000.0, max_distance=50):
    """FAC mechanism figure: single-row 1×3 panel.

        (a) RoPE cos-similarity heatmap with safe harbor annotation
        (b) Normal Key per-channel bars (uniform)
        (c) Sink Key per-channel bars (low-freq concentrated)
    """
    required = ["k_sink_heatmap", "k_rand_heatmap", "layer_list"]
    if not _check_keys(data, required, "fig_mechanism_composite"):
        return

    k_sink_hm = np.array(data["k_sink_heatmap"])
    k_rand_hm = np.array(data["k_rand_heatmap"])
    layer_list = data["layer_list"]

    n_raw_ch = k_sink_hm.shape[1]
    n_display = head_dim // 2  # 64 channels for bar charts

    # ── Compute analytical RoPE similarity ──
    freqs = base ** (-2.0 * np.arange(n_display) / head_dim)
    distances = np.arange(max_distance + 1)
    similarity = np.cos(freqs[:, None] * distances[None, :])

    # ── Per-channel key norms at a representative deep layer ──
    target_layer = 44
    if target_layer in layer_list:
        layer_idx = layer_list.index(target_layer)
    else:
        layer_idx = int(np.argmin(np.abs(np.array(layer_list) - target_layer)))
    layer_id = layer_list[layer_idx]

    sink_raw = k_sink_hm[layer_idx]
    rand_raw = k_rand_hm[layer_idx]
    if n_raw_ch > n_display:
        ratio_ch = n_raw_ch // n_display
        sink_norms = np.mean(sink_raw[:n_display * ratio_ch].reshape(n_display, ratio_ch), axis=1)
        rand_norms = np.mean(rand_raw[:n_display * ratio_ch].reshape(n_display, ratio_ch), axis=1)
    else:
        sink_norms = sink_raw[:n_display]
        rand_norms = rand_raw[:n_display]
    bar_xmax = max(sink_norms.max(), rand_norms.max()) * 1.1
    low_freq_y = int(n_display * 0.55)

    # ── Layout: single row, 4 columns (RoPE + bars + bars + colorbar) ──
    fig = plt.figure(figsize=(FIG_FULL, 2.4), dpi=300)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[2.0, 0.7, 0.7, 0.05],
        wspace=0.12,
    )

    y = np.arange(n_display)

    # (a) RoPE cos-similarity heatmap
    ax_rope = fig.add_subplot(gs[0, 0])
    im_rope = ax_rope.imshow(
        similarity, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        interpolation="nearest",
        extent=[0, max_distance, n_display - 0.5, -0.5],
    )
    from matplotlib.patches import FancyBboxPatch
    harbor_box = FancyBboxPatch(
        (1.0, low_freq_y + 0.5), max_distance - 2, n_display - low_freq_y - 1.5,
        boxstyle="round,pad=0,rounding_size=2.5",
        linewidth=1.5, edgecolor="#F0C75E", facecolor="#F0C75E",
        linestyle="-", zorder=3, alpha=0.08,
    )
    ax_rope.add_patch(harbor_box)
    harbor_border = FancyBboxPatch(
        (1.0, low_freq_y + 0.5), max_distance - 2, n_display - low_freq_y - 1.5,
        boxstyle="round,pad=0,rounding_size=2.5",
        linewidth=1.5, edgecolor="#F0C75E", facecolor="none",
        linestyle="-", zorder=4, alpha=0.9,
    )
    ax_rope.add_patch(harbor_border)
    anchor_y = (low_freq_y + n_display) / 2
    ax_rope.text(max_distance / 2, anchor_y, "SAFE  HARBOR",
                 fontsize=FS_LABEL, color="white", ha="center", va="center",
                 fontweight="bold", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#D4A843", ec="none", alpha=0.75))
    ax_rope.text(max_distance / 2, anchor_y + 5,
                 r"$\cos(\theta_c \cdot \Delta) \approx 1$",
                 fontsize=FS_TICK, color="#F0C75E", ha="center", va="center", zorder=5)
    ax_rope.set_xlabel("Token Distance")
    ax_rope.set_ylabel("Channel  (high freq → low freq)")
    ax_rope.set_xticks([0, 10, 20, 30, 40, 50])
    ax_rope.set_yticks([0, 16, 32, 48, 63])
    _clean_ax(ax_rope)
    ax_rope.set_title("(a) RoPE Similarity", fontweight="bold")

    # (b) Normal Key bars
    ax_nk = fig.add_subplot(gs[0, 1], sharey=ax_rope)
    ax_nk.barh(y, rand_norms, height=0.8, color=C_BLUE, alpha=0.85)
    ax_nk.set_xlim(0, bar_xmax)
    ax_nk.set_title(f"(b) Normal Key\n(L{layer_id})", fontweight="bold",
                     color=C_BLUE)
    ax_nk.set_xlabel("|Key|", fontsize=FS_TICK)
    ax_nk.tick_params(axis="y", labelleft=False)
    ax_nk.tick_params(axis="x", labelsize=FS_SMALL)
    _clean_ax(ax_nk)
    ax_nk.axhspan(low_freq_y - 0.5, n_display - 0.5, color=C_BLUE, alpha=0.05, zorder=0)

    # (c) Sink Key bars
    ax_sk = fig.add_subplot(gs[0, 2], sharey=ax_rope)
    ax_sk.barh(y, sink_norms, height=0.8, color=C_RED, alpha=0.85)
    ax_sk.set_xlim(0, bar_xmax)
    ax_sk.set_title(f"(c) Sink Key\n(L{layer_id})", fontweight="bold",
                     color=C_RED)
    ax_sk.set_xlabel("|Key|", fontsize=FS_TICK)
    ax_sk.tick_params(axis="y", labelleft=False)
    ax_sk.tick_params(axis="x", labelsize=FS_SMALL)
    _clean_ax(ax_sk)
    ax_sk.axhspan(low_freq_y - 0.5, n_display - 0.5, color=C_RED, alpha=0.08, zorder=0)
    ax_sk.text(bar_xmax * 0.5, (low_freq_y + n_display) / 2,
               "safe\nharbor", fontsize=FS_TICK, color=C_RED,
               ha="center", va="center", fontstyle="italic", alpha=0.7)

    # Colorbar
    cax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im_rope, cax=cax, orientation="vertical")
    cbar.set_label("cos sim", fontsize=FS_SMALL)
    cbar.ax.tick_params(labelsize=FS_SMALL)

    _save_fig(fig, out_dir, "fig_mechanism_composite.pdf")


def _make_paper_cmap():
    """Build a custom sequential colormap matching the paper red theme.

    Transition: black → deep blue → C_RED → warm orange → bright yellow.
    """
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = [
        (0.00, "#0A0A14"),   # near-black
        (0.15, "#1A2456"),   # C_DEEP
        (0.40, "#A72B4A"),   # C_RED
        (0.70, "#D4734E"),   # C_ORANGE
        (1.00, "#F5E663"),   # warm yellow highlight
    ]
    positions = [c[0] for c in colors_list]
    hex_cols = [c[1] for c in colors_list]
    rgb = [mcolors.to_rgb(h) for h in hex_cols]
    return LinearSegmentedColormap.from_list("paper_heat", list(zip(positions, rgb)))


def _build_sink_spatial_panels(ax_img, ax_heat, ax_cbar,
                               spatial_json_path, flux_image_path,
                               num_context=512, grid_h=64, grid_w=64):
    """Render original image + attention-overlay heatmap in two axes.

    Parameters
    ----------
    ax_img   : Axes for the generated image (top)
    ax_heat  : Axes for the overlaid heatmap (bottom)
    ax_cbar  : Axes for the horizontal colorbar
    spatial_json_path : str – flux_keyimportance_stats.json
    flux_image_path   : str – FLUX-generated image
    """
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    with open(spatial_json_path, encoding="utf-8") as f:
        stats = json.load(f)
    importance = np.array(stats["importance"])

    # Extract image-token importance → spatial grid
    image_imp = importance[num_context : num_context + grid_h * grid_w]
    spatial = image_imp.reshape(grid_h, grid_w)

    # Paper-consistent custom colormap
    cmap = _make_paper_cmap()

    # ── Top: original generated image ──
    img = Image.open(flux_image_path).convert("RGB")
    ax_img.imshow(img, aspect="equal")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.5)
    ax_img.set_title("Generated Image", fontweight="bold", pad=4)

    # ── Bottom: pure spatial heatmap with log normalization ──
    # Light smooth + log scale to reveal spatial structure in sparse data
    spatial_smooth = gaussian_filter(spatial, sigma=1.0)
    log_norm = mcolors.LogNorm(vmin=max(spatial_smooth[spatial_smooth > 0].min(), 1e-5),
                                vmax=spatial_smooth.max())

    im_heat = ax_heat.imshow(spatial_smooth, cmap=cmap, aspect="equal",
                             interpolation="bilinear", norm=log_norm)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for spine in ax_heat.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.5)
    ax_heat.set_title("Sink Key Attention Map", fontweight="bold", pad=4)

    # ── Annotate top hotspot ──
    peak_r, peak_c = np.unravel_index(spatial_smooth.argmax(),
                                       spatial_smooth.shape)
    peak_val = spatial[peak_r, peak_c]
    # Place label near hotspot, offset towards center
    off_x = grid_w * 0.25 if peak_c < grid_w * 0.5 else -grid_w * 0.25
    off_y = -grid_h * 0.18 if peak_r > grid_h * 0.5 else grid_h * 0.18
    ax_heat.annotate(
        f"peak = {peak_val:.3f}",
        xy=(peak_c, peak_r), xytext=(peak_c + off_x, peak_r + off_y),
        fontsize=FS_LABEL, color="white", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.0,
                        shrinkA=0, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.25", fc=C_RED, ec="none", alpha=0.85),
    )

    # ── Colorbar (log-scaled, paper cmap) ──
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=log_norm)
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.ax.tick_params(labelsize=FS_LABEL, length=2, pad=1)
    cbar.set_label("Key Importance (avg, 100 images, log scale)",
                   fontsize=FS_LABEL, labelpad=2)


def plot_fig_sink_value_info(data, out_dir, spatial_json=None, flux_image=None):
    """Fig 10: Sinks Carry Minimal Information.

    Layout (with spatial data): left = spatial attention overlay on image,
        right = 2 heatmaps (V Sink, V Random) + 2 line charts.
    Layout (without spatial data): 2 heatmaps + 2 line charts (legacy).
    Data from step_*_sink_curves.json saved by task_attn_sink_tokens_analysis.
    """

    if not _check_keys(data, ["v_sink_heatmap", "v_rand_heatmap", "layer_list",
                               "Attention_sink_curve", "Attention_rand_curve"],
                       "fig_sink_value_info"):
        return

    v_sink_hm = np.array(data["v_sink_heatmap"])   # (num_layers, channels)
    v_rand_hm = np.array(data["v_rand_heatmap"])    # (num_layers, channels)
    layer_list = data["layer_list"]
    attn_sink = data["Attention_sink_curve"]
    attn_rand = data["Attention_rand_curve"]

    # Shared colorbar range
    vmin = min(v_sink_hm.min(), v_rand_hm.min())
    vmax = max(v_sink_hm.max(), v_rand_hm.max())

    # Layer tick positions — show every ~5th layer to avoid crowding
    n_layers = len(layer_list)
    tick_pos, tick_lbl = _layer_ticks(layer_list)

    has_spatial = (spatial_json and os.path.exists(spatial_json)
                   and flux_image and os.path.exists(flux_image))

    if has_spatial:
        # ── Wide layout: left (image + heatmap) + right (V-heatmaps + charts) ──
        fig = plt.figure(figsize=(FIG_FULL, 4.5), dpi=300)
        gs_outer = fig.add_gridspec(1, 2, width_ratios=[0.36, 0.64], wspace=0.06)

        # Left panel: image (top) + heatmap (bottom) + colorbar
        gs_left = gs_outer[0].subgridspec(
            3, 1, height_ratios=[1.0, 1.0, 0.06], hspace=0.15)
        ax_img = fig.add_subplot(gs_left[0])
        ax_heat = fig.add_subplot(gs_left[1])
        ax_cbar = fig.add_subplot(gs_left[2])
        _build_sink_spatial_panels(ax_img, ax_heat, ax_cbar,
                                   spatial_json, flux_image)

        # Right panel: original 3×3 grid
        gs_right = gs_outer[1].subgridspec(
            3, 3, width_ratios=[1.0, 1.0, 0.04],
            height_ratios=[1.0, 1.0, 1.0], wspace=0.25, hspace=0.4,
        )
    else:
        # ── Legacy layout (no spatial data) ──
        fig = plt.figure(figsize=(FIG_FULL, 4.5), dpi=300)
        gs_right = fig.add_gridspec(
            3, 3, width_ratios=[1.0, 1.0, 0.04],
            height_ratios=[1.0, 1.0, 1.0], wspace=0.25, hspace=0.4,
        )

    # --- (a) V Sink heatmap ---
    ax_vs = fig.add_subplot(gs_right[0, :2])
    im = ax_vs.imshow(
        v_sink_hm.T, aspect="auto", cmap="magma",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax_vs.set_title("V Sink", fontweight="bold")
    ax_vs.set_ylabel("Channel")
    ax_vs.set_xticks(tick_pos)
    ax_vs.set_xticklabels([], fontsize=FS_TICK)
    ax_vs.tick_params(labelsize=FS_TICK)

    # --- (b) V Random heatmap ---
    ax_vr = fig.add_subplot(gs_right[1, :2])
    ax_vr.imshow(
        v_rand_hm.T, aspect="auto", cmap="magma",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax_vr.set_title("V Random", fontweight="bold")
    ax_vr.set_ylabel("Channel")
    ax_vr.set_xlabel("Layer")
    ax_vr.set_xticks(tick_pos)
    ax_vr.set_xticklabels(tick_lbl, fontsize=FS_TICK)
    ax_vr.tick_params(labelsize=FS_TICK)

    # --- Shared colorbar ---
    cax = fig.add_subplot(gs_right[:2, 2])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=FS_LABEL)
    cbar.set_label("Mean |value|", fontsize=FS_LABEL)

    # --- (c) Attention Score line chart ---
    ax_attn = fig.add_subplot(gs_right[2, 0])
    _plot_sink_vs_rand(ax_attn, range(n_layers), attn_sink, attn_rand,
                       "Key Importance", "Attention Score")
    ax_attn.set_xticks(tick_pos)
    ax_attn.set_xticklabels(tick_lbl, fontsize=FS_TICK)

    # --- (d) Value L2-norm line chart ---
    v_sink_l2 = data.get("V_sink_curve", [])
    v_rand_l2 = data.get("V_rand_curve", [])
    ax_vnorm = fig.add_subplot(gs_right[2, 1])
    if v_sink_l2 and len(v_sink_l2) == n_layers:
        _plot_sink_vs_rand(ax_vnorm, range(n_layers), v_sink_l2, v_rand_l2,
                           "L2-norm", "Value L2-norm")
    ax_vnorm.set_xticks(tick_pos)
    ax_vnorm.set_xticklabels(tick_lbl, fontsize=FS_TICK)

    _save_fig(fig, out_dir, "fig_sink_value_info.pdf")


def plot_fig_cheating_strategy(data, out_dir):
    """Fig: Cheating Strategy — per-channel Key norm comparison (Sink vs Random).

    Two side-by-side horizontal bar charts at a single representative layer.
    Sink Keys concentrate energy in low-frequency channels; Random Keys spread
    energy uniformly.  This is THE core mechanism figure.

    Data keys used:
        k_sink_heatmap  (num_layers, 128)
        k_rand_heatmap  (num_layers, 128)
        layer_list
    """
    if not _check_keys(data, ["k_sink_heatmap", "k_rand_heatmap", "layer_list"],
                       "fig_cheating_strategy"):
        return

    k_sink_hm = np.array(data["k_sink_heatmap"])   # (num_layers, 128)
    k_rand_hm = np.array(data["k_rand_heatmap"])
    layer_list = data["layer_list"]

    # Representative deep layer: layer 44 (or nearest available)
    target_layer = 44
    if target_layer in layer_list:
        layer_idx = layer_list.index(target_layer)
    else:
        layer_idx = min(range(len(layer_list)),
                        key=lambda i: abs(layer_list[i] - target_layer))
    layer_id = layer_list[layer_idx]
    n_channels = k_sink_hm.shape[1]

    sink_norms = k_sink_hm[layer_idx]   # (128,)
    rand_norms = k_rand_hm[layer_idx]

    # Channel indices: 0 = highest RoPE frequency, 127 = lowest.
    # Display: high-freq at top (y=127), low-freq at bottom (y=0).
    # So we flip: display_y[i] = (n_channels - 1 - i).
    channels = np.arange(n_channels)
    display_y = n_channels - 1 - channels  # flip for display

    # Shared x-axis limit for visual comparison
    x_max = max(sink_norms.max(), rand_norms.max()) * 1.08

    # Low-frequency region: bottom ~30 channels (indices 98-127)
    low_freq_threshold = 30  # number of low-freq channels to highlight
    shade_y_min = -0.5
    shade_y_max = low_freq_threshold - 0.5  # display_y for index 98..127 → 0..29

    fig, (ax_sink, ax_rand) = plt.subplots(
        1, 2, figsize=(FIG_FULL, 3.2), dpi=300, sharey=True,
    )

    # --- Left: Sink Key ---
    ax_sink.barh(display_y, sink_norms, height=0.8, color=C_RED, alpha=0.85)
    ax_sink.set_xlim(0, x_max)
    ax_sink.set_ylim(-1, n_channels)
    ax_sink.set_xlabel("Mean |Key| norm")
    ax_sink.set_ylabel("Channel index  (high freq \u2192 low freq \u2193)")
    ax_sink.set_title(f"Sink Key  (Layer {layer_id})", fontweight="bold")
    _clean_ax(ax_sink)

    # Shade low-frequency region
    ax_sink.axhspan(shade_y_min, shade_y_max, color=C_RED, alpha=0.08, zorder=0)
    ax_sink.text(
        x_max * 0.75, shade_y_max * 0.45,
        "Low freq\n(safe harbor)", fontsize=FS_ANNOT, color=C_RED,
        ha="center", va="center", fontstyle="italic", alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
    )

    # Y-axis labels
    ytick_step = 16
    ytick_positions = display_y[::ytick_step]
    ytick_labels = channels[::ytick_step]
    ax_sink.set_yticks(ytick_positions)
    ax_sink.set_yticklabels(ytick_labels)

    # --- Right: Random Key ---
    ax_rand.barh(display_y, rand_norms, height=0.8, color=C_BLUE, alpha=0.85)
    ax_rand.set_xlim(0, x_max)
    ax_rand.set_xlabel("Mean |Key| norm")
    ax_rand.set_title(f"Random Key  (Layer {layer_id})", fontweight="bold")
    _clean_ax(ax_rand)

    # Shade low-frequency region
    ax_rand.axhspan(shade_y_min, shade_y_max, color=C_BLUE, alpha=0.06, zorder=0)

    _save_fig(fig, out_dir, "fig_cheating_strategy.pdf")


def plot_fig_sink_overview(data, out_dir, spatial_json=None, flux_image=None):
    """Main paper: Sink QKV analysis — heatmaps + key curves.

    Top row: 4 channel-wise heatmaps (K Sink, K Random, V Sink, V Random)
             + thin shared colorbar
    Bottom row: 2 curves (Attention Score, Value L2-norm)
    V Sink panel: "NEAR-ZERO" bounding box highlighting minimal information.

    When spatial_json + flux_image are provided, adds a left column with
    generated image + sink key attention map (merged from fig_sink_value_info).
    """
    required = ["k_sink_heatmap", "k_rand_heatmap", "v_sink_heatmap", "v_rand_heatmap",
                "K_sink_curve", "K_rand_curve", "V_sink_curve", "V_rand_curve",
                "Attention_sink_curve", "Attention_rand_curve", "layer_list"]
    if not _check_keys(data, required, "fig_sink_overview"):
        return

    layer_list = data["layer_list"]
    n_layers = len(layer_list)
    tick_pos, tick_lbl = _layer_ticks(layer_list)

    has_spatial = (spatial_json and os.path.exists(spatial_json)
                   and flux_image and os.path.exists(flux_image))

    if has_spatial:
        # ── Wide layout: left (image + heatmap) | right (4 heatmaps + 2 curves) ──
        fig = plt.figure(figsize=(FIG_FULL, 3.5), dpi=300)
        gs_outer = fig.add_gridspec(1, 2, width_ratios=[0.28, 0.72], wspace=0.06)

        # Left panel: image (top) + attention heatmap (bottom) + colorbar
        gs_left = gs_outer[0].subgridspec(
            3, 1, height_ratios=[1.0, 1.0, 0.06], hspace=0.15)
        ax_img = fig.add_subplot(gs_left[0])
        ax_heat = fig.add_subplot(gs_left[1])
        ax_cbar_left = fig.add_subplot(gs_left[2])
        _build_sink_spatial_panels(ax_img, ax_heat, ax_cbar_left,
                                   spatial_json, flux_image)

        # Right panel: 4 heatmaps + cbar (top) + 2 curves (bottom)
        gs_right = gs_outer[1].subgridspec(
            2, 5,
            width_ratios=[1, 1, 1, 1, 0.04],
            height_ratios=[1.2, 1.0],
            hspace=0.30, wspace=0.20,
        )
        gs = gs_right
    else:
        # ── Compact layout (no spatial data) ──
        fig = plt.figure(figsize=(FIG_FULL, 3.5), dpi=300)
        gs = fig.add_gridspec(
            2, 5,
            width_ratios=[1, 1, 1, 1, 0.04],
            height_ratios=[1.2, 1.0],
            hspace=0.30, wspace=0.20,
        )

    # ── Top row: 4 heatmaps + shared colorbar ──
    heatmap_specs = [
        ("K Sink", "k_sink_heatmap"),
        ("K Random", "k_rand_heatmap"),
        ("V Sink", "v_sink_heatmap"),
        ("V Random", "v_rand_heatmap"),
    ]

    # Percentile-based color range across all 4 heatmaps
    all_hm = np.concatenate([np.array(data[k]).ravel() for _, k in heatmap_specs])
    vmin, vmax = np.percentile(all_hm, 2), np.percentile(all_hm, 98)

    im = None
    for col, (title, key) in enumerate(heatmap_specs):
        ax = fig.add_subplot(gs[0, col])
        hm = np.array(data[key])  # (n_layers, n_channels)
        n_ch = hm.shape[1]
        im = ax.imshow(hm.T, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Channel")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl if col >= 2 else [])
        ax.set_yticks(np.linspace(0, n_ch - 1, 4, dtype=int))

        # ── "NEAR-ZERO" bounding box on V Sink ──
        if key == "v_sink_heatmap":
            # Find where V Sink values drop to near-zero (relative to own max)
            row_means = hm.mean(axis=1)
            own_max = row_means.max()
            if own_max > 0:
                near_zero_mask = row_means < own_max * 0.05
                # Find the first sustained near-zero region (≥3 consecutive layers)
                nz_start = None
                for idx in range(len(near_zero_mask)):
                    if near_zero_mask[idx:idx + 3].all():
                        nz_start = idx
                        break
                if nz_start is not None:
                    nz_end = n_layers - 1
                    rect = mpatches.FancyBboxPatch(
                        (nz_start - 0.5, -0.5), nz_end - nz_start + 1, n_ch,
                        boxstyle="round,pad=0.2",
                        linewidth=1.2, edgecolor=C_ORANGE, facecolor="none",
                        zorder=5,
                    )
                    ax.add_patch(rect)
                    mid_x = (nz_start + nz_end) / 2
                    ax.text(
                        mid_x, n_ch * 0.50,
                        "NEAR-ZERO",
                        fontsize=FS_ANNOT, fontweight="bold", color=C_ORANGE,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1A2456",
                                  ec="none", alpha=0.85),
                        zorder=6,
                    )
                    ax.text(
                        mid_x, n_ch * 0.65,
                        r"$\approx$ drop attention",
                        fontsize=FS_SMALL, color="white", ha="center", va="center",
                        fontstyle="italic", zorder=6,
                    )

    # Thin shared colorbar
    cax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=FS_SMALL, length=2)
    cbar.set_label("Channel norm", fontsize=FS_TICK, labelpad=2)

    # ── Bottom row: 2 key curves (evenly split across 4 heatmap cols) ──
    curve_specs = [
        ("Attention Score", "Attention_sink_curve", "Attention_rand_curve",
         "Key Importance"),
        ("Value L2-norm", "V_sink_curve", "V_rand_curve", "L2-norm"),
    ]

    for i, (title, sink_key, rand_key, ylabel) in enumerate(curve_specs):
        ax = fig.add_subplot(gs[1, i * 2:(i + 1) * 2])

        sink_vals = np.array(data[sink_key])
        rand_vals = np.array(data[rand_key])
        x = np.arange(n_layers)
        ax.plot(x, sink_vals, "o-", color=C_RED, lw=0.9, ms=1.8, label="Sink")
        ax.plot(x, rand_vals, "s-", color=C_BLUE, lw=0.9, ms=1.8, label="Random")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl)
        _clean_ax(ax)
        if i == 0:
            ax.legend(framealpha=0.8, edgecolor="none", loc="upper left")

    _save_fig(fig, out_dir, "fig_sink_overview.pdf")


def plot_fig_rope_phase_shift(out_dir, head_dim=128, base=10000.0,
                              max_distance=50, n_display_channels=64):
    """Fig: RoPE Phase Shift — channel-wise QK dot-product similarity vs distance.

    Analytical computation (no model data needed):
      cos(theta_i * distance), where theta_i = base^(-2i/head_dim).
    High-freq channels oscillate; low-freq channels stay ~1 ("Phase Shift ≈ 0").
    """
    n_pairs = head_dim // 2  # each pair shares one frequency
    # RoPE frequencies: theta_i = base^(-2i / head_dim)
    freqs = base ** (-2.0 * np.arange(n_pairs) / head_dim)  # (n_pairs,)

    distances = np.arange(max_distance + 1)  # 0..50
    # cos(theta_i * d) for each channel pair i and distance d
    # Shape: (n_pairs, n_distances)
    similarity = np.cos(freqs[:, None] * distances[None, :])

    # Display only first n_display_channels pairs (0=highest freq, 63=lowest)
    similarity = similarity[:n_display_channels]

    fig, ax = plt.subplots(figsize=(FIG_HALF, 2.5), dpi=300)
    im = ax.imshow(
        similarity, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        interpolation="nearest", extent=[0, max_distance, n_display_channels, 0],
    )

    # Low-frequency "safe harbor" box (bottom ~40% of channels)
    low_freq_start = int(n_display_channels * 0.35)
    rect = plt.Rectangle(
        (0, low_freq_start), max_distance, n_display_channels - low_freq_start,
        linewidth=1.2, edgecolor=C_ORANGE, facecolor="none",
        linestyle="--", zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        max_distance / 2, (low_freq_start + n_display_channels) / 2,
        "Phase Shift $\\approx$ 0",
        fontsize=FS_LABEL, color="white", ha="center", va="center",
        fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.2", fc=C_RED, ec="none", alpha=0.6),
    )

    ax.set_xlabel("Distance  (/ Token)")
    ax.set_ylabel("Channel")

    ax.text(-3.5, n_display_channels * 0.12, "High Freq.",
            fontsize=FS_TICK, color=C_RED, fontweight="bold", ha="right", va="center")
    ax.text(-3.5, n_display_channels * 0.82, "Low Freq.",
            fontsize=FS_TICK, color=C_ORANGE, fontweight="bold", ha="right", va="center")

    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 16, 32, 48, 64])
    _clean_ax(ax)

    legend_elements = [
        mpatches.Patch(facecolor="#8B1A1A", label="$q \\cdot k^T = 1$"),
        mpatches.Patch(facecolor="#1A3A8B", label="$q \\cdot k^T = -1$"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              framealpha=0.8, edgecolor="none")

    fig.tight_layout()
    _save_fig(fig, out_dir, "fig_rope_phase_shift.pdf")


def plot_fig_safe_harbor_composite(out_dir, data=None, head_dim=128,
                                   base=10000.0, max_distance=50):
    """Composite: (a) RoPE heatmap | (b) Normal Key bars | (c) Sink Key bars.

    All three panels share the Y-axis (channel index, high freq at top).
    Visual logic: low-freq channels have phase shift ~ 0 in (a),
    normal keys spread energy uniformly in (b), sink keys concentrate
    in low-freq in (c) — exploiting the safe harbor.
    """
    n_channels = head_dim // 2  # 64 display channels

    # ── (a) analytical heatmap ──
    freqs = base ** (-2.0 * np.arange(n_channels) / head_dim)
    distances = np.arange(max_distance + 1)
    similarity = np.cos(freqs[:, None] * distances[None, :])

    # ── (b)(c) per-channel Key norms ──
    has_bars = (data is not None and
                _check_keys(data, ["k_sink_heatmap", "k_rand_heatmap",
                                   "layer_list"], "safe_harbor_bars"))

    if has_bars:
        k_sink_hm = np.array(data["k_sink_heatmap"])
        k_rand_hm = np.array(data["k_rand_heatmap"])
        layer_list = data["layer_list"]
        # Pick a deep layer (~44) where sink behavior is most pronounced
        target_layer = 44
        if target_layer in layer_list:
            layer_idx = layer_list.index(target_layer)
        else:
            # Fallback: pick the layer closest to target
            layer_idx = int(np.argmin(np.abs(np.array(layer_list) - target_layer)))

        # Map 128 channels → 64 display channels (pair average)
        sink_raw = k_sink_hm[layer_idx]
        rand_raw = k_rand_hm[layer_idx]
        n_raw = len(sink_raw)
        if n_raw > n_channels:
            ratio = n_raw // n_channels
            sink_norms = np.mean(sink_raw[:n_channels * ratio].reshape(n_channels, ratio), axis=1)
            rand_norms = np.mean(rand_raw[:n_channels * ratio].reshape(n_channels, ratio), axis=1)
        else:
            sink_norms = sink_raw[:n_channels]
            rand_norms = rand_raw[:n_channels]

        x_max = max(sink_norms.max(), rand_norms.max()) * 1.1

    # ── layout: 3 columns, shared Y ──
    fig = plt.figure(figsize=(FIG_FULL, 2.6), dpi=300)

    if has_bars:
        gs = fig.add_gridspec(1, 4, width_ratios=[2.5, 1, 1, 0.04],
                              wspace=0.06)
        ax_hm = fig.add_subplot(gs[0, 0])
        ax_rand = fig.add_subplot(gs[0, 1], sharey=ax_hm)
        ax_sink = fig.add_subplot(gs[0, 2], sharey=ax_hm)
        cax = fig.add_subplot(gs[0, 3])
    else:
        fig, ax_hm = plt.subplots(figsize=(FIG_HALF, 2.6), dpi=300)

    y = np.arange(n_channels)

    # ── (a) RoPE phase-shift heatmap ──
    im = ax_hm.imshow(
        similarity, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        interpolation="nearest",
        extent=[0, max_distance, n_channels - 0.5, -0.5],
    )

    # Safe harbor zone — rounded gold rectangle + text label
    low_freq_y = int(n_channels * 0.55)
    from matplotlib.patches import FancyBboxPatch
    harbor_box = FancyBboxPatch(
        (1.0, low_freq_y + 0.5), max_distance - 2, n_channels - low_freq_y - 1.5,
        boxstyle="round,pad=0,rounding_size=2.5",
        linewidth=1.5, edgecolor="#F0C75E", facecolor="#F0C75E",
        linestyle="-", zorder=3, alpha=0.08,
    )
    ax_hm.add_patch(harbor_box)
    # Gold border (separate for crisp edge)
    harbor_border = FancyBboxPatch(
        (1.0, low_freq_y + 0.5), max_distance - 2, n_channels - low_freq_y - 1.5,
        boxstyle="round,pad=0,rounding_size=2.5",
        linewidth=1.5, edgecolor="#F0C75E", facecolor="none",
        linestyle="-", zorder=4, alpha=0.9,
    )
    ax_hm.add_patch(harbor_border)

    anchor_y = (low_freq_y + n_channels) / 2
    # Main label with pill-shaped background
    ax_hm.text(
        max_distance / 2, anchor_y,
        "SAFE  HARBOR",
        fontsize=FS_LABEL, color="white", ha="center", va="center",
        fontweight="bold", zorder=5, fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.3", fc="#D4A843", ec="none", alpha=0.75),
    )
    ax_hm.text(
        max_distance / 2, anchor_y + 5.5,
        "cos$(\u03B8_c \\cdot \\Delta) \\approx 1$",
        fontsize=FS_TICK, color="#F0C75E", ha="center", va="center",
        zorder=5,
    )

    ax_hm.set_xlabel("Token Distance")
    ax_hm.set_ylabel("Channel  (high freq $\\rightarrow$ low freq)")
    ax_hm.set_xticks([0, 10, 20, 30, 40, 50])
    ax_hm.set_yticks([0, 16, 32, 48, 63])
    _clean_ax(ax_hm)

    ax_hm.text(0.02, 0.97, "(a) RoPE Similarity",
               transform=ax_hm.transAxes, fontsize=FS_TITLE, fontweight="bold",
               va="top", color="white",
               bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4))

    # ── (b) Normal Key per-channel bars ──
    if has_bars:
        ax_rand.barh(y, rand_norms, height=0.8, color=C_BLUE, alpha=0.85)
        ax_rand.set_xlim(0, x_max)
        ax_rand.set_title("(b) Normal Key", fontweight="bold", color=C_BLUE)
        ax_rand.set_xlabel("Mean |Key|")
        ax_rand.tick_params(axis="y", labelleft=False)
        ax_rand.tick_params(axis="x", labelsize=FS_SMALL)
        _clean_ax(ax_rand)

        ax_sink.barh(y, sink_norms, height=0.8, color=C_RED, alpha=0.85)
        ax_sink.set_xlim(0, x_max)
        ax_sink.set_title("(c) Sink Key", fontweight="bold", color=C_RED)
        ax_sink.set_xlabel("Mean |Key|")
        ax_sink.tick_params(axis="y", labelleft=False)
        ax_sink.tick_params(axis="x", labelsize=FS_SMALL)
        _clean_ax(ax_sink)

        ax_rand.axhspan(low_freq_y - 0.5, n_channels - 0.5,
                        color=C_BLUE, alpha=0.05, zorder=0)
        ax_sink.axhspan(low_freq_y - 0.5, n_channels - 0.5,
                        color=C_RED, alpha=0.08, zorder=0)
        ax_sink.text(x_max * 0.5, (low_freq_y + n_channels) / 2,
                     "safe\nharbor", fontsize=FS_TICK, color=C_RED,
                     ha="center", va="center", fontstyle="italic", alpha=0.7)

        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_label("cos similarity", fontsize=FS_TICK)
        cbar.ax.tick_params(labelsize=FS_SMALL)

    _save_fig(fig, out_dir, "fig_safe_harbor_composite.pdf")


# ── Sink temporal robustness figures ──

def _generate_demo_sink_time():
    """Generate realistic FLUX layers x steps sink heatmap demo data."""
    rng = np.random.default_rng(42)
    n_layers = 19
    layers = list(range(0, 57, 3))[:n_layers]
    steps = [0, 5, 10, 15, 20, 30, 40, 50]
    n_steps = len(steps)

    arr = np.zeros((n_layers, n_steps))
    for i in range(n_layers):
        base = 0.002 + 0.02 * np.exp(-((i - 10) ** 2) / 30)
        for j in range(n_steps):
            arr[i, j] = base * (0.8 + 0.4 * rng.random())
    arr[:4, :] *= 0.3
    arr[7:15, :] *= 2.5
    arr[15:, :] *= 0.6
    return arr, layers, steps


def _load_sink_time_data(exp_root):
    """Load real sink_time data from exp00d stats directory."""
    if not exp_root:
        return None, None, None
    stats_dir = os.path.join(exp_root, "attn_sink_task/stats/per_step")
    if not os.path.exists(stats_dir):
        return None, None, None

    layer_dirs = sorted(
        [d for d in os.listdir(stats_dir) if d.startswith("layer_")],
        key=lambda x: int(x.split("_")[1]))
    layers, steps, data = [], None, []
    for ld in layer_dirs:
        layers.append(int(ld.split("_")[1]))
        lpath = os.path.join(stats_dir, ld)
        vals = {}
        for sf in os.listdir(lpath):
            if "keyimportance_softmax_stats" not in sf:
                continue
            step = int(sf.split("_")[3])
            with open(os.path.join(lpath, sf)) as f:
                for line in f:
                    if line.strip().startswith("max:"):
                        vals[step] = float(line.strip().split("max:")[1])
                        break
        if steps is None:
            steps = sorted(vals.keys())
        data.append([vals.get(s, 1e-6) for s in steps])
    return np.clip(np.array(data), 1e-4, None), layers, steps


def plot_fig_sink_time_v3(out_dir, exp_root=None):
    """V3: Overlaid curves + temporal flatness — proves robustness visually.

    Left: Max Key Importance vs Layer, one curve per step (color gradient).
          Curves overlap → sink pattern is step-invariant.
    Right: Key Importance vs Step for 3-4 selected layers.
           Flat lines → temporally robust.
    """
    arr, layers, steps = _load_sink_time_data(exp_root)
    if arr is None:
        print("  Using demo data for sink_time_v3")
        arr, layers, steps = _generate_demo_sink_time()

    n_layers, n_steps = arr.shape

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(FIG_FULL, 2.0), dpi=300,
        gridspec_kw={"wspace": 0.22, "width_ratios": [1.2, 1]},
    )

    # ── Left: overlaid layer-wise curves, one per step ──
    cmap = plt.cm.magma_r
    for j in range(n_steps):
        color = cmap(0.2 + 0.6 * j / max(n_steps - 1, 1))
        alpha = 0.4 + 0.5 * j / max(n_steps - 1, 1)
        ax_left.plot(range(n_layers), arr[:, j], "-", color=color,
                     lw=0.8, alpha=alpha)

    # Highlight first (pure noise) and last step
    ax_left.plot(range(n_layers), arr[:, 0], "o-", color=C_ORANGE,
                 lw=1.0, ms=2, label=f"Step {steps[0]} (noise)", zorder=5)
    ax_left.plot(range(n_layers), arr[:, -1], "s-", color=C_RED,
                 lw=1.0, ms=2, label=f"Step {steps[-1]} (final)", zorder=5)

    tick_pos, tick_lbl = _layer_ticks(layers, every=6)
    ax_left.set_xticks(tick_pos)
    ax_left.set_xticklabels(tick_lbl)
    ax_left.set_xlabel("Layer")
    ax_left.set_ylabel("Max Key Importance")
    ax_left.set_title("(a) Layer-wise profile at each step", fontweight="bold")
    ax_left.legend(frameon=True, framealpha=0.9, loc="upper left")
    _clean_ax(ax_left)

    # ── Right: temporal traces for selected layers ──
    # Pick layers: early (low sink), transition, peak sink, deep
    layer_peaks = np.mean(arr, axis=1)
    peak_layer_idx = int(np.argmax(layer_peaks))
    selected = sorted(set([
        1,                          # early layer
        max(1, peak_layer_idx - 3), # pre-transition
        peak_layer_idx,             # peak sink
        min(n_layers - 1, peak_layer_idx + 4),  # post-peak
    ]))

    markers = ["o", "D", "s", "^"]
    colors_sel = [C_GRAY, C_BLUE, C_RED, C_ORANGE]
    for k, li in enumerate(selected):
        ax_right.plot(range(n_steps), arr[li, :],
                      f"{markers[k % 4]}-", color=colors_sel[k % 4],
                      lw=0.9, ms=2, label=f"Layer {layers[li]}")

    ax_right.set_xticks(range(n_steps))
    ax_right.set_xticklabels([str(s) for s in steps], fontsize=FS_SMALL)
    ax_right.set_xlabel("Denoising Step")
    ax_right.set_ylabel("Max Key Importance")
    ax_right.set_title("(b) Temporal stability per layer", fontweight="bold")
    ax_right.legend(frameon=True, framealpha=0.9, loc="best")
    _clean_ax(ax_right)

    _save_fig(fig, out_dir, "fig_sink_time_v3.pdf")


# ── FLUX token layout constants ──
# Token 17 = EOS (T5 end-of-sentence)
# 49-73 = semantic text tokens (user prompt)
# 0-16, 18-48, 74-511 = PAD tokens
# 512+ = image latent tokens
FLUX_CONTEXT_LEN = 512
FLUX_EOS_IDX = 17
FLUX_TEXT_RANGE = (49, 73)  # inclusive

# PixArt token layout
# Token 0 = BOS, Token 31 = EOS (sink), rest = text/pad, then image
PIXART_CONTEXT_LEN = 120
PIXART_EOS_IDX = 31


def _flux_token_color(tid):
    """Return bar color for a FLUX context token."""
    if tid == FLUX_EOS_IDX:
        return C_RED
    if FLUX_TEXT_RANGE[0] <= tid <= FLUX_TEXT_RANGE[1]:
        return C_BLUE
    return C_ORANGE  # PAD


def _pixart_token_color(tid):
    """Return bar color for a PixArt context token."""
    if tid == PIXART_EOS_IDX:
        return C_RED
    if tid == 0:
        return C_ORANGE
    return C_BLUE


def _generate_demo_flux(rng):
    """Generate realistic FLUX per-token attention (context-only, 512 tokens).

    Pattern from Padding Tone (NAACL 2025) + our exp00d observations:
    - EOS (token 17): highest spike (~0.22)
    - PAD tokens: moderate-high (~0.04-0.08), FLUX *uses* them
    - Text tokens (49-73): moderate (~0.01-0.03)
    """
    n = FLUX_CONTEXT_LEN
    attn = np.zeros(n)

    # PAD tokens: base level with noise
    pad_base = 0.05
    attn[:] = pad_base + rng.uniform(-0.015, 0.015, n)

    # Text tokens: lower than PAD
    for i in range(FLUX_TEXT_RANGE[0], FLUX_TEXT_RANGE[1] + 1):
        attn[i] = 0.015 + rng.uniform(0, 0.012)

    # EOS spike
    attn[FLUX_EOS_IDX] = 0.22 + rng.uniform(-0.01, 0.01)

    # BOS (token 0) also gets some attention
    attn[0] = 0.08 + rng.uniform(-0.01, 0.01)

    # A few other special tokens get small bumps
    for idx in [1, 2, 48]:
        attn[idx] = 0.06 + rng.uniform(-0.01, 0.01)

    return np.clip(attn, 0, None)


def _generate_demo_pixart(rng):
    """Generate realistic PixArt per-token attention (context-only, 120 tokens).

    Pattern: token 31 (EOS) is the dominant sink, token 0 (BOS) moderate,
    rest of text tokens low and uniform.
    """
    n = PIXART_CONTEXT_LEN
    attn = np.zeros(n)

    # Text tokens: low uniform
    attn[:] = 0.004 + rng.uniform(-0.001, 0.002, n)

    # BOS
    attn[0] = 0.025 + rng.uniform(-0.003, 0.003)

    # EOS sink at token 31
    attn[PIXART_EOS_IDX] = 0.14 + rng.uniform(-0.01, 0.01)

    return np.clip(attn, 0, None)


def _draw_token_bar(ax, attn, color_fn, model_label,
                    skip_range=None, annotate_eos=None):
    """Draw a single padding-tone-style bar chart on the given axes.

    Args:
        ax: matplotlib Axes
        attn: 1-D attention array (context tokens only)
        color_fn: callable(token_idx) -> color string
        model_label: e.g. "FLUX" shown as bold text inside the panel
        skip_range: optional (start, end) of token indices to collapse
        annotate_eos: optional token index to annotate with an arrow
    """
    n = len(attn)

    if skip_range is not None:
        s_start, s_end = skip_range
        keep_left = list(range(0, s_start))
        keep_right = list(range(s_end + 1, n))
        kept_indices = keep_left + keep_right
        display_attn = attn[kept_indices]
        display_colors = [color_fn(i) for i in kept_indices]
        x = np.arange(len(kept_indices))
        break_x = len(keep_left)  # where to draw the break
    else:
        kept_indices = list(range(n))
        display_attn = attn
        display_colors = [color_fn(i) for i in range(n)]
        x = np.arange(n)
        break_x = None

    ax.bar(x, display_attn, width=1.0, color=display_colors,
           edgecolor="none", alpha=0.9)

    ax.set_xlim(-0.5, len(x) - 0.5)
    _clean_ax(ax)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=FS_LABEL, length=3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Model label inside panel
    ax.text(0.02, 0.92, model_label, transform=ax.transAxes,
            fontweight="bold", va="top",
            color="#333333")

    # Break indicator
    if break_x is not None:
        ax.axvline(break_x - 0.5, color=C_GRAY, ls="--", lw=0.8, alpha=0.5)
        n_omitted = skip_range[1] - skip_range[0] + 1
        ax.text(break_x - 1.5, ax.get_ylim()[1] * 0.92,
                f"{n_omitted} PAD\ntokens\nomitted",
                fontsize=FS_TICK, color=C_GRAY, va="top", ha="right",
                fontstyle="italic")

    # EOS annotation
    if annotate_eos is not None:
        if annotate_eos in kept_indices:
            disp_pos = kept_indices.index(annotate_eos)
            ax.annotate(
                "<EOS>",
                xy=(disp_pos, display_attn[disp_pos]),
                xytext=(disp_pos + len(x) * 0.08,
                        display_attn[disp_pos] * 0.70),
                fontsize=FS_LABEL, color=C_RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=0.8),
            )


def _draw_placeholder(ax, label="Image\nPlaceholder"):
    """Draw a gray placeholder rectangle when no image is available."""
    ax.set_facecolor("#F0F0F0")
    ax.text(0.5, 0.5, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=FS_LABEL, color="#999999",
            fontstyle="italic")
    ax.set_xticks([])
    ax.set_yticks([])




def _build_flux_real_importance(global_txt_path):
    """Build a full 512-token importance array from GLOBAL top100 softmax file.

    Returns np.array of shape (512,) with real per-token key importance values.
    Tokens not in the top-100 list get a small baseline value.
    """
    tokens = parse_top100(global_txt_path)
    n_ctx = FLUX_CONTEXT_LEN
    attn = np.zeros(n_ctx)
    # Fill in context-range tokens from the parsed data
    for tid, val in tokens.items():
        if 0 <= tid < n_ctx:
            attn[tid] = val
    # Tokens not in top-100 get a small baseline (below the visible floor)
    if np.count_nonzero(attn) == 0:
        print(f"  Warning: no context-range tokens found in {global_txt_path}")
        return attn
    baseline = min(v for tid, v in tokens.items() if 0 <= tid < n_ctx) * 0.3
    attn[attn == 0] = baseline
    return attn


def plot_fig_sink_identity(out_dir, exp_root="./outputs"):
    """FLUX per-token Key Importance bar — standalone panel for subfigure layout.

    Outputs fig_sink_identity_flux.pdf. The PixArt 3D distribution is a
    separate PDF (fig_pixart_sink_3d.pdf); they are combined via LaTeX subfigure.
    """
    flux_global_txt = None
    for candidate in [
        os.path.join(exp_root, "flux_GLOBAL_top100_softmax.txt"),
        "outputs/flux_GLOBAL_top100_softmax.txt",
    ]:
        if os.path.exists(candidate):
            flux_global_txt = candidate
            break

    skip_range = (80, 490)
    n_ctx = FLUX_CONTEXT_LEN
    keep_left = list(range(0, skip_range[0]))
    keep_right = list(range(skip_range[1] + 1, n_ctx))
    kept_indices = keep_left + keep_right
    n_display = len(kept_indices)

    fig = plt.figure(figsize=(FIG_HALF + 0.8, 2.4), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.07, 1], hspace=0.08)

    # Token-type strip
    ax_strip = fig.add_subplot(gs[0])
    bar_colors = [_flux_token_color(idx) for idx in kept_indices]
    for i, color in enumerate(bar_colors):
        rect = plt.Rectangle((i, 0), 1, 1,
                              facecolor=color, edgecolor="none", alpha=0.85)
        ax_strip.add_patch(rect)
    ax_strip.set_xlim(-0.5, n_display - 0.5)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    eos_disp = kept_indices.index(FLUX_EOS_IDX)
    ax_strip.text(eos_disp, 0.5, "E", ha="center", va="center",
                  fontsize=FS_SMALL, color="white", fontweight="bold")
    text_start_disp = kept_indices.index(FLUX_TEXT_RANGE[0])
    text_end_disp = kept_indices.index(FLUX_TEXT_RANGE[1])
    ax_strip.text((text_start_disp + text_end_disp) / 2, 0.5, "Text",
                  ha="center", va="center", fontsize=FS_SMALL,
                  color="white", fontweight="bold")
    break_x = len(keep_left)
    ax_strip.axvline(break_x - 0.5, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)

    # Key importance bar
    ax_flux = fig.add_subplot(gs[1])
    if flux_global_txt and os.path.exists(flux_global_txt):
        flux_attn = _build_flux_real_importance(flux_global_txt)
    else:
        rng = np.random.default_rng(42)
        flux_attn = _generate_demo_flux(rng)

    _draw_token_bar(ax_flux, flux_attn, _flux_token_color, "",
                    skip_range=skip_range, annotate_eos=FLUX_EOS_IDX)
    ax_flux.set_ylabel("Key Importance", fontsize=FS_LABEL, labelpad=4)

    patches = [
        mpatches.Patch(color=C_RED, label="<EOS>"),
        mpatches.Patch(color=C_ORANGE, label="<PAD>"),
        mpatches.Patch(color=C_BLUE, label="Text"),
    ]
    ax_flux.legend(handles=patches, loc="upper right", ncol=3,
                   fontsize=FS_SMALL, frameon=True, framealpha=0.9)

    _save_fig(fig, out_dir, "fig_sink_identity_flux.pdf")


def plot_fig_sink_types(out_dir, exp_root="./outputs",
                        flux_image=None, pixart_image=None, wan_image=None,
                        flux_global_txt=None):
    """Fig 6: Sink Identity Across DiT Architectures — 3 separate panel PDFs.

    Panel (a) PixArt-alpha: image + 32x32 spatial grid + attention curve
    Panel (b) FLUX.1: image + prompt bar + real per-token key importance bar
    Panel (c) Wan2.1: image + layer-wise sink curve + top-30 token bar chart
    """
    pixart_json = os.path.join(exp_root, "exp00d_pixart_sink_analysis/step_19_sink_curves.json")
    flux_json = os.path.join(exp_root, "exp05a_flux_sink_value_info/step_27_sink_curves.json")

    pixart_data = None
    if os.path.exists(pixart_json):
        with open(pixart_json, encoding="utf-8") as f:
            pixart_data = json.load(f)

    flux_data = None
    if os.path.exists(flux_json):
        with open(flux_json, encoding="utf-8") as f:
            flux_data = json.load(f)

    # Try to find GLOBAL key importance file
    if flux_global_txt is None:
        for candidate in [
            os.path.join(exp_root, "flux_GLOBAL_top100_softmax.txt"),
            "outputs/flux_GLOBAL_top100_softmax.txt",
        ]:
            if os.path.exists(candidate):
                flux_global_txt = candidate
                break

    # Try to find Wan stats directory
    wan_stats_dir = None
    for candidate in [
        os.path.join(exp_root, "wan21_sink_stats"),
        "outputs/wan21_sink_stats",
    ]:
        if os.path.isdir(candidate):
            wan_stats_dir = candidate
            break

    _plot_panel_pixart(out_dir, pixart_data, pixart_image)
    _plot_panel_flux(out_dir, flux_data, flux_image, flux_global_txt)
    _plot_panel_wan(out_dir, wan_image, wan_stats_dir)


def _plot_panel_pixart(out_dir, data, image_path):
    """Panel (a): PixArt — image | spatial grid + attention curve."""
    fig = plt.figure(figsize=(FIG_HALF, 2.2), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1],
                          height_ratios=[1.2, 1], wspace=0.25, hspace=0.3)

    ax_img = fig.add_subplot(gs[:, 0])
    if image_path and os.path.exists(image_path):
        img = plt.imread(image_path)
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
    else:
        _draw_placeholder(ax_img, "PixArt-$\\alpha$\nGenerated Image")

    ax_grid = fig.add_subplot(gs[0, 1])
    grid = np.zeros((32, 32))
    sink_positions = []
    if data:
        for idx in data.get("sink_idx", [31, 0, 1023]):
            r, c = divmod(idx, 32)
            grid[r, c] = 1.0
            sink_positions.append((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 32 and 0 <= nc < 32:
                        grid[nr, nc] = max(grid[nr, nc], 0.4)

    grid[0, :] = np.maximum(grid[0, :], 0.15)
    grid[-1, :] = np.maximum(grid[-1, :], 0.15)
    grid[:, 0] = np.maximum(grid[:, 0], 0.15)
    grid[:, -1] = np.maximum(grid[:, -1], 0.15)

    ax_grid.imshow(grid, cmap="Reds", vmin=0, vmax=1.0,
                   interpolation="nearest", aspect="equal")
    for r, c in sink_positions:
        ax_grid.plot(c, r, "x", color=C_RED, markersize=6, markeredgewidth=2)
    ax_grid.set_title("Sink Locations (32$\\times$32 latent)", fontsize=FS_LABEL, fontweight="bold")
    ax_grid.set_xticks([0, 15, 31])
    ax_grid.set_yticks([0, 15, 31])
    ax_grid.tick_params(labelsize=6)

    ax_attn = fig.add_subplot(gs[1, 1])
    if data:
        layers = data["layer_list"]
        attn_s = np.array(data["Attention_sink_curve"])
        attn_r = np.array(data["Attention_rand_curve"])
        x = np.arange(len(layers))
        ax_attn.plot(x, attn_s, "o-", color=C_RED, lw=1.4, ms=2, label="Sink")
        ax_attn.plot(x, attn_r, "s-", color=C_BLUE, lw=1.4, ms=2, label="Random")
        tick_pos, tick_lbl = _layer_ticks(layers, every=5)
        ax_attn.set_xticks(tick_pos)
        ax_attn.set_xticklabels(tick_lbl, fontsize=FS_TICK)
        ax_attn.legend(fontsize=FS_TICK, frameon=True, framealpha=0.9)
    else:
        ax_attn.text(0.5, 0.5, "No data", transform=ax_attn.transAxes,
                     ha="center", fontsize=FS_LABEL, color=C_GRAY)
    ax_attn.set_xlabel("Layer")
    ax_attn.set_ylabel("Attention Score")
    ax_attn.tick_params(labelsize=6)
    _clean_ax(ax_attn)

    _save_fig(fig, out_dir, "fig_sink_types_pixart.pdf", tight=False)


def _plot_panel_flux(out_dir, data, image_path, global_txt=None):
    """Panel (b): FLUX — image | prompt bar + key importance bar.

    Prompt bar and bar chart share the same x-axis: 512 context tokens
    with skip_range (80, 490) applied to both, so they are pixel-aligned.
    """
    skip_range = (80, 490)
    n_ctx = FLUX_CONTEXT_LEN  # 512

    # Compute display indices (matching _draw_token_bar logic)
    keep_left = list(range(0, skip_range[0]))
    keep_right = list(range(skip_range[1] + 1, n_ctx))
    kept_indices = keep_left + keep_right
    n_display = len(kept_indices)

    # Auto-discover flux image from assets if not provided
    if not image_path or not os.path.exists(str(image_path or "")):
        for candidate in [
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "assets", "flux_astronaut.jpg"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "assets", "sd3_astronaut.jpg"),
        ]:
            if os.path.exists(candidate):
                image_path = candidate
                break

    # Manual axes layout: [left, bottom, width, height] in figure coords
    # Image on left, chart+bar+legend on right with clear separation
    img_l, img_b, img_w, img_h = 0.02, 0.12, 0.24, 0.85
    chart_l = 0.44  # leave gap for y-axis label
    chart_r_margin = 0.02
    chart_w = 1.0 - chart_l - chart_r_margin

    fig = plt.figure(figsize=(FIG_HALF, 2.6), dpi=300)

    # Left: generated image
    ax_img = fig.add_axes([img_l, img_b, img_w, img_h])
    if image_path and os.path.exists(image_path):
        img = plt.imread(image_path)
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
    else:
        _draw_placeholder(ax_img, "FLUX.1\nGenerated Image")

    # Right top: prompt token-type bar
    bar_h = 0.06
    bar_b = img_b + img_h - bar_h
    ax_bar = fig.add_axes([chart_l, bar_b, chart_w, bar_h])
    bar_colors = [_flux_token_color(idx) for idx in kept_indices]
    for i, color in enumerate(bar_colors):
        rect = plt.Rectangle((i, 0), 1, 1,
                              facecolor=color, edgecolor="none", alpha=0.85)
        ax_bar.add_patch(rect)
    ax_bar.set_xlim(-0.5, n_display - 0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # Region labels on bar
    eos_disp = kept_indices.index(FLUX_EOS_IDX)
    ax_bar.text(eos_disp, 0.5, "E", ha="center", va="center",
                fontsize=FS_SMALL, color="white", fontweight="bold")
    text_start_disp = kept_indices.index(FLUX_TEXT_RANGE[0])
    text_end_disp = kept_indices.index(FLUX_TEXT_RANGE[1])
    text_mid = (text_start_disp + text_end_disp) / 2
    ax_bar.text(text_mid, 0.5, "Text", ha="center", va="center",
                fontsize=FS_SMALL, color="white", fontweight="bold")
    break_x = len(keep_left)
    ax_bar.axvline(break_x - 0.5, color=C_GRAY, ls="--", lw=0.6, alpha=0.5)

    # Right middle: key importance bar chart
    ki_gap = 0.02
    leg_h = 0.06
    ki_b = img_b + leg_h + 0.02
    ki_h = bar_b - ki_gap - ki_b
    ax_ki = fig.add_axes([chart_l, ki_b, chart_w, ki_h])

    if global_txt and os.path.exists(global_txt):
        flux_attn = _build_flux_real_importance(global_txt)
        print(f"  Using real FLUX key importance from {global_txt}")
    else:
        rng = np.random.default_rng(42)
        flux_attn = _generate_demo_flux(rng)
        if data:
            attn_s = np.array(data["Attention_sink_curve"])
            peak_ratio = max(attn_s) / (np.mean(attn_s) + 1e-8)
            flux_attn[FLUX_EOS_IDX] = np.mean(flux_attn) * min(peak_ratio, 6.0)
        print("  Using demo FLUX key importance (no GLOBAL txt found)")

    _draw_token_bar(
        ax_ki, flux_attn, _flux_token_color, "",
        skip_range=skip_range,
        annotate_eos=FLUX_EOS_IDX,
    )
    ax_ki.set_ylabel("Key Importance")
    ax_ki.tick_params(labelsize=5)

    # Right bottom: legend
    ax_leg = fig.add_axes([chart_l, img_b, chart_w, leg_h])
    ax_leg.axis("off")
    patches = [
        mpatches.Patch(color=C_RED, label="<EOS>"),
        mpatches.Patch(color=C_ORANGE, label="<PAD>"),
        mpatches.Patch(color=C_BLUE, label="Text"),
    ]
    ax_leg.legend(handles=patches, loc="center", ncol=3,
                  fontsize=FS_TICK, frameon=False)

    _save_fig(fig, out_dir, "fig_sink_types_flux.pdf", tight=False)


def _plot_panel_wan(out_dir, image_path, wan_stats_dir=None):
    """Panel (c): Wan2.1 — image | layer-wise sink curve + top-token bar."""
    # Auto-discover wan image from assets if not provided
    if not image_path or not os.path.exists(str(image_path or "")):
        candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "assets", "wan_horse_frame0.jpg")
        if os.path.exists(candidate):
            image_path = candidate

    fig = plt.figure(figsize=(FIG_HALF, 2.6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1],
                          height_ratios=[1, 1], wspace=0.25, hspace=0.40)

    # Left: video first frame
    ax_img = fig.add_subplot(gs[:, 0])
    if image_path and os.path.exists(image_path):
        img = plt.imread(image_path)
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
    else:
        _draw_placeholder(ax_img, "Wan2.1\nVideo First Frame")

    # Load per-layer data if available
    has_data = False
    layers, top1_vals = [], []
    global_tokens = {}

    if wan_stats_dir and os.path.isdir(wan_stats_dir):
        # Parse per-layer files for top-1 key importance
        for lf in sorted(Path(wan_stats_dir).glob("layer_*_softmax.txt"),
                         key=lambda p: int(p.name.split("_")[1])):
            layer_id = int(lf.name.split("_")[1])
            tokens = parse_top100(str(lf))
            if tokens:
                layers.append(layer_id)
                top1_vals.append(max(tokens.values()))
        # Parse GLOBAL file
        gf = os.path.join(wan_stats_dir, "GLOBAL_over_steps_avg_top100_softmax.txt")
        if os.path.exists(gf):
            global_tokens = parse_top100(gf)
        has_data = bool(layers)
        if has_data:
            print(f"  Using real Wan data ({len(layers)} layers)")

    # Right top: layer-wise top-1 key importance
    ax_layer = fig.add_subplot(gs[0, 1])
    if has_data:
        x = np.arange(len(layers))
        ax_layer.plot(x, top1_vals, "o-", color=C_RED, lw=1.4, ms=3)
        ax_layer.fill_between(x, top1_vals, alpha=0.1, color=C_RED)
        tick_pos, tick_lbl = _layer_ticks(layers, every=5)
        ax_layer.set_xticks(tick_pos)
        ax_layer.set_xticklabels(tick_lbl, fontsize=FS_TICK)
    else:
        ax_layer.text(0.5, 0.5, "No data", transform=ax_layer.transAxes,
                      ha="center", fontsize=FS_LABEL, color=C_GRAY)
    ax_layer.set_xlabel("Layer")
    ax_layer.set_ylabel("Top-1 Key Importance")
    ax_layer.set_title("Layer-wise Sink Concentration", fontsize=FS_LABEL, fontweight="bold")
    ax_layer.tick_params(labelsize=6)
    _clean_ax(ax_layer)

    # Right bottom: top-30 token bar chart from GLOBAL
    ax_bar = fig.add_subplot(gs[1, 1])
    if global_tokens:
        top = sorted(global_tokens.items(), key=lambda t: -t[1])[:30]
        ids = [t[0] for t in top]
        vals = [t[1] for t in top]
        # All tokens are video latent positions (no text/pad distinction like FLUX)
        ax_bar.bar(range(len(ids)), vals, color=C_RED, alpha=0.85, width=0.7)
        ax_bar.set_xticks(range(0, len(ids), 5))
        ax_bar.set_xticklabels([str(ids[i]) for i in range(0, len(ids), 5)],
                               rotation=45, ha="right", fontsize=FS_SMALL)
        ax_bar.set_xlabel("Token Index")
    else:
        ax_bar.text(0.5, 0.5, "No data", transform=ax_bar.transAxes,
                    ha="center", fontsize=FS_LABEL, color=C_GRAY)
    ax_bar.set_ylabel("Key Importance")
    ax_bar.set_title("Top-30 Sink Tokens (Stochastic)", fontsize=FS_LABEL, fontweight="bold")
    ax_bar.tick_params(labelsize=6)
    _clean_ax(ax_bar)

    _save_fig(fig, out_dir, "fig_sink_types_wan.pdf", tight=False)


def plot_fig_layerwise_multimodel(out_dir, model_jsons=None):
    """Layer-wise Key Importance: 2×2 grid, one panel per model.

    Each panel: Sink (red) vs Random (blue) attention score across layers.
    Clean, publication-quality styling with peak ratio annotation.
    """
    if not model_jsons:
        def _last_step(pattern):
            """Pick the highest-step JSON matching a glob pattern."""
            hits = sorted(Path(".").glob(pattern))
            return str(hits[-1]) if hits else None

        def _first_found(*paths):
            """Return the first existing path, or None."""
            for p in paths:
                if p and os.path.exists(p):
                    return p
            return None

        candidates = [
            _first_found(
                "outputs/exp05a_flux_sink_value_info/step_27_sink_curves.json",
                "outputs/sink_curves_full/flux.json"),
            _first_found(
                "outputs/exp00d_pixart_sink_analysis/step_19_sink_curves.json",
                "outputs/sink_curves_full/pixart.json"),
            _first_found(
                _last_step("outputs/Attn_Project_Exp_by_id/exp00d_zimage_sink_analysis/*/*/"
                           "sink_tokens_analysis/stats/step_*_sink_curves.json"),
                "outputs/sink_curves_full/zimage.json"),
            _first_found(
                "outputs/exp05a_wan_sink_value_info/step_49_sink_curves.json",
                "outputs/sink_curves_full/wan.json"),
            _first_found(
                _last_step("outputs/Attn_Project_Exp_by_id/exp00d_qwenimage_sink_analysis/stats/"
                           "step_*_sink_curves.json"),
                "outputs/sink_curves_full/qwenimage.json"),
            _first_found(
                _last_step("outputs/Attn_Project_Exp_by_id/exp00d_ltx_sink_analysis/*/*/"
                           "sink_tokens_analysis/stats/step_*_sink_curves.json"),
                "outputs/sink_curves_full/ltx.json"),
        ]
        model_jsons = [p for p in candidates if p]

    if not model_jsons:
        print("  No model data found for layerwise comparison")
        return

    models = []
    for p in model_jsons:
        with open(p, encoding="utf-8") as f:
            models.append(json.load(f))

    display_names = {
        "FLUX.1-dev": "FLUX.1 (2D RoPE)",
        "PixArt-XL-2-512x512": r"PixArt-$\alpha$ (AbsPE)",
        "Z-Image-Turbo": "Z-Image (3-axis RoPE)",
        "Wan2.1-T2V-1.3B-Diffusers": "Wan2.1 (3D RoPE)",
        "Qwen-Image": "Qwen-Image (MM-DiT + RoPE)",
        "LTX-Video": "LTX Video (3D RoPE)",
    }
    n_models = len(models)
    panel_labels = [f"({chr(ord('a') + i)})" for i in range(n_models)]
    ncols = min(n_models, 3)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(FIG_FULL, 1.8 * nrows), dpi=300,
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
        squeeze=False,
    )

    for idx, mdata in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        layer_list = mdata["layer_list"]
        n = len(layer_list)
        x = np.arange(n)

        sink_key = "Attention_sink_curve"
        rand_key = "Attention_rand_curve"
        if sink_key not in mdata:
            ax.text(0.5, 0.5, "No attention data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS_LABEL, color=C_GRAY)
            continue

        sink_vals = np.array(mdata[sink_key])
        rand_vals = np.array(mdata[rand_key])

        # Plot with filled area under sink curve
        ax.fill_between(x, sink_vals, alpha=0.06, color=C_RED)
        ax.plot(x, sink_vals, "o-", color=C_RED, lw=0.9, ms=1.8,
                label="Sink token", zorder=3)
        ax.plot(x, rand_vals, "s-", color=C_BLUE, lw=0.9, ms=1.8,
                label="Random token", zorder=3)

        # Tick labels — use every=5 to keep ≤6 ticks per panel in 3-col layout
        tick_pos, tick_lbl = _layer_ticks(layer_list, every=5)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl)
        if row == nrows - 1:
            ax.set_xlabel("Layer")
        if col == 0:
            ax.set_ylabel("Key Importance")
        _clean_ax(ax)

        model_name = mdata.get("model_name", f"Model {idx}")
        base = model_name.split("/")[-1]
        short = display_names.get(base, base)
        ax.set_title(f"{panel_labels[idx]} {short}", fontweight="bold")

        # Legend only on first panel
        if idx == 0:
            ax.legend(frameon=True, framealpha=0.9, loc="best")

        # Peak ratio annotation
        ratio = sink_vals / np.clip(rand_vals, EPS, None)
        peak = int(np.argmax(ratio))
        if ratio[peak] > 2:
            x_off = min(2, n * 0.08)
            y_off = sink_vals[peak] * 0.75
            ax.annotate(
                f"{ratio[peak]:.0f}\u00d7",
                xy=(peak, sink_vals[peak]),
                xytext=(peak + x_off, y_off),
                fontsize=FS_ANNOT, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=0.6),
            )

    # Hide unused axes
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    _save_fig(fig, out_dir, "fig_layerwise_multimodel.pdf")


def _annotate_hot_zones(ax, hot_mask, n_layers, color, min_span=3):
    """Draw translucent highlight rectangles on contiguous hot-channel regions."""
    # Find contiguous groups of True in hot_mask
    groups = []
    in_group = False
    for i, val in enumerate(hot_mask):
        if val and not in_group:
            start = i
            in_group = True
        elif not val and in_group:
            groups.append((start, i - 1))
            in_group = False
    if in_group:
        groups.append((start, len(hot_mask) - 1))

    for y0, y1 in groups:
        span = y1 - y0 + 1
        if span < min_span:
            continue
        rect = mpatches.Rectangle(
            (-0.5, y0 - 0.5), n_layers, span,
            linewidth=1.2, edgecolor=color, facecolor=color,
            alpha=0.12, linestyle="--", zorder=4,
        )
        ax.add_patch(rect)


def plot_fig_band_norm(out_dir, sink_curves_data=None):
    """Frequency-Aware Concentration: Sink vs Normal Key heatmaps + norm curves.

    Layout: (a) Sink Key channel-norm heatmap  (b) Normal Key  (c) Key norm curves
    Y-axis shows RoPE 2D-axis structure (h-axis / w-axis split at ch 64 for FLUX).
    Panel (c) plots K_sink_curve vs K_rand_curve across layers.

    Data source: sink_curves JSON from exp05a (k_sink_heatmap, k_rand_heatmap,
    K_sink_curve, K_rand_curve, Attention_sink_curve).
    """
    if sink_curves_data is None:
        print("  Skipping band_norm: no sink_curves data")
        return

    required = ["k_sink_heatmap", "k_rand_heatmap", "K_sink_curve",
                "K_rand_curve", "layer_list"]
    if not _check_keys(sink_curves_data, required, "band_norm"):
        return

    d = sink_curves_data
    k_sink = np.array(d["k_sink_heatmap"])    # (n_layers, n_ch)
    k_rand = np.array(d["k_rand_heatmap"])    # (n_layers, n_ch)
    layers = d["layer_list"]
    K_sink_curve = np.array(d["K_sink_curve"])
    K_rand_curve = np.array(d["K_rand_curve"])
    Attn_sink = np.array(d.get("Attention_sink_curve", []))

    n_layers, n_ch = k_sink.shape
    axis_boundary = n_ch // 2   # h|w split at ch 64 for FLUX 2D RoPE

    tick_pos, tick_lbl = _layer_ticks(layers, every=6)

    # Shared color range (percentile-clipped for robustness)
    all_vals = np.concatenate([k_sink.ravel(), k_rand.ravel()])
    vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)

    fig = plt.figure(figsize=(FIG_FULL, 2.4), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.22)
    ax_sink = fig.add_subplot(gs[0])
    ax_rand = fig.add_subplot(gs[1])
    ax_curve = fig.add_subplot(gs[2])

    # ── helper: style a heatmap panel ────────────────────────────────────
    def _style_hm(ax, hm, title_label, show_ylabel):
        im = ax.imshow(hm.T, aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title_label, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl)
        # h/w axis boundary
        ax.axhline(axis_boundary - 0.5, color="white", ls="--", lw=0.8, alpha=0.85)
        if show_ylabel:
            ax.set_ylabel("RoPE Channel Index")
            seg_mids = [axis_boundary // 2, axis_boundary + axis_boundary // 2]
            ax.set_yticks(seg_mids)
            ax.set_yticklabels(["$D_h$ (height)", "$D_w$ (width)"],
                               fontsize=FS_TICK, fontstyle="italic")
            ax.tick_params(axis="y", length=0, pad=3)
            for y_frac, label in [(0.99, r"High $\theta$"), (0.52, r"Low $\theta$"),
                                  (0.48, r"High $\theta$"), (0.01, r"Low $\theta$")]:
                ax.annotate(label, xy=(-0.15, y_frac), xycoords="axes fraction",
                            fontsize=FS_SMALL, fontstyle="italic", color="#777",
                            ha="center", va="center")
        else:
            ax.set_yticks([])
        return im

    # (a) Sink Key — with low-frequency activation zone highlight
    _style_hm(ax_sink, k_sink, "(a) Sink Key", show_ylabel=True)

    # Data-driven: find high-energy zones via per-channel mean energy ratio
    sink_profile = k_sink.mean(axis=0)
    rand_profile = k_rand.mean(axis=0)
    ratio = sink_profile / (rand_profile + 1e-8)

    # Highlight contiguous sink-specific hot zones (ratio > 80th percentile)
    hot_mask = ratio > np.percentile(ratio, 80)
    _annotate_hot_zones(ax_sink, hot_mask, n_layers, C_ORANGE)

    # Summary label at bottom-right
    ax_sink.text(
        0.97, 0.05, "Low-freq\nconcentrated",
        transform=ax_sink.transAxes, fontsize=FS_SMALL, fontweight="bold",
        color="white", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc=C_ORANGE, ec="none", alpha=0.8),
        zorder=6,
    )

    # (b) Normal Key — with high-frequency activation zone highlight
    im = _style_hm(ax_rand, k_rand, "(b) Normal Key", show_ylabel=False)

    # Highlight normal-key specific hot zones (inverse ratio)
    inv_ratio = rand_profile / (sink_profile + 1e-8)
    hot_mask_r = inv_ratio > np.percentile(inv_ratio, 80)
    _annotate_hot_zones(ax_rand, hot_mask_r, n_layers, C_BLUE)

    ax_rand.text(
        0.97, 0.05, "Broadly\ndistributed",
        transform=ax_rand.transAxes, fontsize=FS_SMALL, fontweight="bold",
        color="white", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc=C_BLUE, ec="none", alpha=0.8),
        zorder=6,
    )

    # Colorbar between heatmaps and curve panel
    cbar = fig.colorbar(im, ax=ax_rand, shrink=0.85, pad=0.03)
    cbar.set_label("Channel $\\ell_2$-norm", fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_SMALL)

    # ── (c) Key norm curves ──────────────────────────────────────────────
    x = np.arange(n_layers)
    ax_curve.plot(x, K_sink_curve, color=C_RED, lw=1.0, label="Sink $\\|\\mathbf{k}\\|$")
    ax_curve.plot(x, K_rand_curve, color=C_BLUE, lw=1.0, label="Normal $\\|\\mathbf{k}\\|$")
    ax_curve.set_xlabel("Layer")
    ax_curve.set_ylabel("Key $\\ell_2$-norm")
    ax_curve.set_title("(c) Key Norm", fontweight="bold")
    ax_curve.set_xticks(tick_pos)
    ax_curve.set_xticklabels(tick_lbl)
    _clean_ax(ax_curve)

    # Overlay attention sink ratio on twin axis
    if len(Attn_sink) > 0:
        ax2 = ax_curve.twinx()
        ax2.fill_between(x, 0, Attn_sink, color=C_ORANGE, alpha=0.12)
        ax2.plot(x, Attn_sink, color=C_ORANGE, lw=0.8, ls="--", alpha=0.7,
                 label="Attn sink ratio")
        ax2.set_ylabel("Attention weight", color=C_ORANGE)
        ax2.tick_params(axis="y", colors=C_ORANGE, labelsize=FS_SMALL)
        ax2.spines["top"].set_visible(False)
        h1, l1 = ax_curve.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax_curve.legend(h1 + h2, l1 + l2, loc="upper left",
                        frameon=True, framealpha=0.9)
    else:
        ax_curve.legend(loc="upper left", frameon=True, framealpha=0.9)

    _save_fig(fig, out_dir, "fig_band_norm.pdf")


def plot_fig_band_comparison(out_dir, model_jsons=None):
    """2×N heatmap: Q (top) and K (bottom) frequency usage across all layers.

    Each column is a model. Reads q_rand_heatmap / k_rand_heatmap from
    sink_curves JSON (shape: n_layers × 128 channels).  Falls back to
    k_rand_heatmap for Q if q_rand_heatmap is absent.

    White dashed lines mark RoPE axis boundaries if segment info is available.
    """
    if model_jsons is None:
        # Auto-discover
        candidates = [
            ("outputs/exp05a_flux_sink_value_info/step_27_sink_curves.json", "FLUX.1 (2D RoPE)"),
            ("outputs/exp05a_wan_sink_value_info/step_49_sink_curves.json", "Wan2.1 (3D RoPE)"),
        ]
        # Also try nested paths
        wan_cands = list(Path("outputs").glob(
            "Attn_Project_Exp_by_id/exp05a_wan*/Wan*/*/sink_tokens_analysis/stats/step_*_sink_curves.json"
        ))
        if wan_cands and not os.path.exists(candidates[1][0]):
            candidates[1] = (str(wan_cands[0]), "Wan2.1 (3D RoPE)")
        model_jsons = [(p, lbl) for p, lbl in candidates if os.path.exists(p)]

    if not model_jsons:
        print("  Skipping band_comparison: no data found")
        return

    # Load all model data
    models = []
    for jpath, label in model_jsons:
        with open(jpath, encoding="utf-8") as f:
            d = json.load(f)
        q_hm = np.array(d.get("q_rand_heatmap") or d["k_rand_heatmap"])
        k_hm = np.array(d["k_rand_heatmap"])
        layers = d["layer_list"]
        models.append({"label": label, "q": q_hm, "k": k_hm, "layers": layers,
                        "n_ch": q_hm.shape[1]})

    n_models = len(models)

    fig, axes = plt.subplots(
        2, n_models, figsize=(FIG_HALF * n_models, 3.5), dpi=300,
        gridspec_kw={"hspace": 0.20, "wspace": 0.10},
        squeeze=False,
    )

    # Global color range
    all_vals = np.concatenate([np.concatenate([m["q"].ravel(), m["k"].ravel()]) for m in models])
    vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)

    row_labels = ["Query", "Key"]
    for col, m in enumerate(models):
        for row, (comp, lbl) in enumerate([(m["q"], "Query"), (m["k"], "Key")]):
            ax = axes[row, col]
            im = ax.imshow(comp.T, aspect="auto", cmap="magma",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            if row == 0:
                ax.set_title(m["label"], fontweight="bold")
            ax.set_xlabel("Layer")
            if col == 0:
                ax.set_ylabel(lbl, fontweight="bold")
            tp, tl = _layer_ticks(m["layers"], every=6)
            ax.set_xticks(tp)
            ax.set_xticklabels(tl, fontsize=FS_TICK)

            n_ch = m["n_ch"]
            ax.set_yticks([0, n_ch // 4, n_ch // 2, 3 * n_ch // 4, n_ch - 1])
            ax.set_yticklabels(["0", str(n_ch // 4), str(n_ch // 2),
                                str(3 * n_ch // 4), str(n_ch - 1)],
                               fontsize=FS_TICK)

            # White dashed lines for axis boundaries
            for key_prefix, segs in ROPE_AXIS_SEGS.items():
                if key_prefix.lower() in m["label"].lower():
                    for seg_ch in segs:
                        if seg_ch < n_ch:
                            ax.axhline(seg_ch - 0.5, color="white", ls="--",
                                       lw=1.2, alpha=0.8)
                    break

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Mean norm", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_LABEL)

    fig.suptitle("Frequency Usage Across Layers", fontsize=FS_TITLE, fontweight="bold", y=0.98)
    _save_fig(fig, out_dir, "fig_flux_wan_band_comparison.pdf")


def plot_fig_ape_vs_rope_band(out_dir):
    """Compact 1×N Key heatmap: RoPE models show band structure, APE models don't.

    Single row of Key channel-norm heatmaps across all available models.
    RoPE models are placed first, then APE models, separated visually.
    """
    # Auto-discover all models
    candidates = [
        # RoPE models first
        ("outputs/exp05a_flux_sink_value_info/step_27_sink_curves.json",
         "FLUX.1\n(2D RoPE)"),
        ("outputs/exp05a_wan_sink_value_info/step_49_sink_curves.json",
         "Wan2.1\n(3D RoPE)"),
    ]
    # Z-Image
    zi_cands = list(Path("outputs").glob(
        "Attn_Project_Exp_by_id/exp00d_zimage_sink_analysis/Z-Image*/**/step_*_sink_curves.json"
    ))
    if zi_cands:
        zi_sorted = sorted(zi_cands, key=lambda p: p.name)
        candidates.append((str(zi_sorted[-1]), "Z-Image\n(3-axis RoPE)"))
    # LTX Video
    ltx_cands = list(Path("outputs").glob(
        "Attn_Project_Exp_by_id/exp00d_ltx_sink_analysis/LTX*/**/step_*_sink_curves.json"
    ))
    if ltx_cands:
        # Pick final step
        ltx_sorted = sorted(ltx_cands, key=lambda p: p.name)
        candidates.append((str(ltx_sorted[-1]), "LTX Video\n(3D RoPE)"))
    # APE models
    candidates += [
        ("outputs/exp05a_sd3_sink_value_info/step_27_sink_curves.json",
         "SD3\n(Sinusoidal)"),
        ("outputs/exp00d_pixart_sink_analysis/step_19_sink_curves.json",
         "PixArt-α\n(AbsPE)"),
    ]

    model_jsons = [(p, lbl) for p, lbl in candidates if os.path.exists(p)]
    if len(model_jsons) < 3:
        print("  Skipping ape_vs_rope_band: need ≥3 models, found", len(model_jsons))
        return

    # Load data
    models = []
    for jpath, label in model_jsons:
        with open(jpath, encoding="utf-8") as f:
            d = json.load(f)
        k_hm = np.array(d["k_rand_heatmap"])
        layers = d["layer_list"]
        models.append({"label": label, "k": k_hm, "layers": layers,
                        "n_ch": k_hm.shape[1]})

    n = len(models)

    # Figure: 2×3 grid for better aspect ratio
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(FIG_FULL, 2.0 * nrows), dpi=300,
        gridspec_kw={"wspace": 0.15, "hspace": 0.45},
    )
    axes_flat = axes.ravel() if n > 1 else [axes]

    # Global color range from Key only
    all_k = np.concatenate([m["k"].ravel() for m in models])
    vmin, vmax = np.percentile(all_k, 1), np.percentile(all_k, 99)

    for i, m in enumerate(models):
        ax = axes_flat[i]
        im = ax.imshow(m["k"].T, aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(m["label"], fontsize=FS_LABEL, fontweight="bold", linespacing=1.1)
        ax.set_xlabel("Layer", fontsize=FS_TICK)
        if i % ncols == 0:
            ax.set_ylabel("Channel", fontsize=FS_TICK)

        tp, tl = _layer_ticks(m["layers"], every=8)
        ax.set_xticks(tp)
        ax.set_xticklabels(tl)

        n_ch = m["n_ch"]
        ax.set_yticks([0, n_ch - 1])
        ax.set_yticklabels(["0", str(n_ch - 1)], fontsize=FS_SMALL)

        # RoPE axis boundary lines
        for key_prefix, segs in ROPE_AXIS_SEGS.items():
            if key_prefix.lower() in m["label"].lower():
                for seg_ch in segs:
                    if seg_ch < n_ch:
                        ax.axhline(seg_ch - 0.5, color="white", ls="--",
                                   lw=0.8, alpha=0.7)
                break

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes_flat[:n], shrink=0.8, pad=0.02, aspect=20)
    cbar.set_label("Mean |Key|", fontsize=FS_TICK)
    cbar.ax.tick_params(labelsize=FS_SMALL)

    _save_fig(fig, out_dir, "fig_ape_vs_rope_band.pdf")


def plot_fig_wan_3d_rope_bottleneck(out_dir, wan_json=None):
    """Wan2.1 3D-RoPE band-norm heatmap with D_t / D_h / D_w axis annotations.

    Layout: (a) Query frequency usage  (b) Key frequency usage
    White dashed lines separate the three RoPE axis chunks.
    Bracket-style labels on the left mark each segment.
    """
    # ── locate data ──────────────────────────────────────────────────────
    if wan_json is None:
        candidates = [
            Path("./outputs/exp05a_wan_sink_value_info/step_49_sink_curves.json"),
        ] + list(Path("./outputs").glob(
            "Attn_Project_Exp_by_id/exp05a_wan*/Wan*/*/sink_tokens_analysis/stats/step_*_sink_curves.json"
        ))
        for c in candidates:
            if c.exists():
                wan_json = str(c)
                break
    if wan_json is None or not os.path.exists(wan_json):
        print("  Skipping wan_3d_rope_bottleneck: no Wan data found")
        return

    with open(wan_json, encoding="utf-8") as f:
        d = json.load(f)

    q_hm = np.array(d["q_rand_heatmap"])   # (30, 128)
    k_hm = np.array(d["k_rand_heatmap"])   # (30, 128)
    layers = d["layer_list"]               # [0..29]
    n_layers, n_ch = q_hm.shape

    # 3D RoPE axis boundaries for Wan2.1 (128-dim head split into t/h/w)
    # Temporal: ch 0-43, Height: ch 44-85, Width: ch 86-127
    seg_bounds = ROPE_AXIS_SEGS["Wan"]
    seg_labels = [
        (0, 43, "$D_t$\n(Temporal)"),
        (44, 85, "$D_h$\n(Height)"),
        (86, 127, "$D_w$\n(Width)"),
    ]

    tick_pos, tick_lbl = _layer_ticks(layers, every=6)

    # ── shared color range (percentile clip for robustness) ──────────────
    all_vals = np.concatenate([q_hm.ravel(), k_hm.ravel()])
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 97)

    fig, (ax_q, ax_k) = plt.subplots(
        1, 2, figsize=(FIG_FULL, 2.6), dpi=300,
        gridspec_kw={"wspace": 0.06, "width_ratios": [1, 1.08]},
    )

    for ax, hm, label in [(ax_q, q_hm, "(a) Query frequency usage — Wan2.1 (3D RoPE)"),
                           (ax_k, k_hm, "(b) Key frequency usage — Wan2.1 (3D RoPE)")]:
        im = ax.imshow(
            hm.T, aspect="auto", cmap="magma",
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(label, fontweight="bold", pad=4)
        ax.set_xlabel("Layer")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=FS_TICK)

        # White dashed lines at axis boundaries
        for b in seg_bounds:
            ax.axhline(b - 0.5, color="white", ls="--", lw=1.3, alpha=0.85)

    # ── Y-axis: segment labels on left of ax_q ──────────────────────────
    # Use small ticks at segment midpoints with axis-chunk labels
    seg_mids = [(lo + hi) / 2 for lo, hi, _ in seg_labels]
    seg_short = ["$D_t$ (Temporal)", "$D_h$ (Height)", "$D_w$ (Width)"]
    ax_q.set_yticks(seg_mids)
    ax_q.set_yticklabels(seg_short, fontsize=FS_TICK, fontstyle="italic")
    ax_q.tick_params(axis="y", length=0, pad=4)

    # "High freq" / "Low freq" arrows at top-left and bottom-left of heatmap
    ax_q.annotate("High freq", xy=(-0.14, 0.98), xycoords="axes fraction",
                  fontsize=FS_TICK, fontstyle="italic", color="#666",
                  ha="center", va="top")
    ax_q.annotate("Low freq", xy=(-0.14, 0.02), xycoords="axes fraction",
                  fontsize=FS_TICK, fontstyle="italic", color="#666",
                  ha="center", va="bottom")

    ax_k.set_yticks([])

    # ── Colorbar ─────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax_k, shrink=0.85, pad=0.02)
    cbar.set_label("Mean norm", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_LABEL)

    _save_fig(fig, out_dir, "fig_wan_3d_rope_bottleneck.pdf")


# ── Model display config for multi-model figures ────────────────────────
_MODEL_DISPLAY = {
    "flux":   {"label": "FLUX.1",       "color": C_RED,    "pe": "2D RoPE",  "marker": "o"},
    "pixart": {"label": "PixArt-α",     "color": C_ORANGE, "pe": "AbsPE",    "marker": "s"},
    "sd3":    {"label": "SD3",          "color": C_GRAY,   "pe": "Sinusoidal","marker": "^"},
    "wan":    {"label": "Wan2.1",       "color": C_BLUE,   "pe": "3D RoPE",  "marker": "D"},
    "ltx":    {"label": "LTX Video",    "color": C_DEEP,   "pe": "3D RoPE",  "marker": "v"},
}

_MULTISTEP_DIRS = {
    "flux":   "exp09j_flux_multistep_100",
    "sd3":    "exp09k_sd3_multistep_100",
    "pixart": "exp09i_pixart_multistep_100",
    "wan":    "exp09l_wan_multistep_100",
    "ltx":    "exp09m_ltx_multistep_100",
}


def _load_multistep_stats(base_dir):
    """Load all 100-sample JSONs and aggregate intensity + position stats."""
    jsons = sorted(glob.glob(
        os.path.join(base_dir, "**", "sink_over_time_data.json"), recursive=True
    ))
    if not jsons:
        return None

    # Collect per-sample: intensity and top-1 position at each step
    all_top1_scores = []  # (n_samples, n_steps)
    all_top1_idx = []     # (n_samples, n_steps)
    meta = None

    for jpath in jsons:
        with open(jpath, encoding="utf-8") as f:
            d = json.load(f)
        if meta is None:
            meta = {
                "store_time_ids": d["store_time_ids"],
                "num_inference_steps": d["num_inference_steps"],
                "seq_len": d["seq_len"],
                "num_image_tokens": d["num_image_tokens"],
                "sink_layers": d["sink_layers"],
                "model": d["model"],
            }
        scores_row = []
        idx_row = []
        for sid in d["store_time_ids"]:
            sinks = d["per_step_sinks"].get(str(sid), [])
            if sinks:
                scores_row.append(sinks[0]["score"])
                idx_row.append(sinks[0].get("token_idx"))
            else:
                scores_row.append(0.0)
                idx_row.append(None)
        all_top1_scores.append(scores_row)
        all_top1_idx.append(idx_row)

    scores_arr = np.array(all_top1_scores)  # (n_samples, n_steps)
    meta["intensity_mean"] = scores_arr.mean(axis=0)
    meta["intensity_std"] = scores_arr.std(axis=0)
    meta["intensity_max"] = scores_arr.max(axis=0)
    meta["n_samples"] = len(jsons)

    # Normalized denoising progress
    n_steps = meta["num_inference_steps"]
    meta["progress"] = [s / n_steps for s in meta["store_time_ids"]]

    # Top-1 position analysis: unique token count per step
    n_time = len(meta["store_time_ids"])
    position_diversity = []
    for t in range(n_time):
        idxs = [row[t] for row in all_top1_idx if row[t] is not None]
        position_diversity.append(len(set(idxs)) if idxs else 0)
    meta["position_diversity"] = position_diversity

    return meta


def plot_fig_sink_dynamics_multimodel(out_dir, exp_root="./outputs/Attn_Project_Exp_by_id"):
    """Multi-model sink dynamics: intensity curves + token position stability.

    Layout: (a) Overlaid intensity curves (mean ± std from 100 samples)
            (b) Token position consistency (unique top-1 count per step)
    """
    # ── Load all models ──────────────────────────────────────────────────
    model_order = ["flux", "pixart", "wan"]
    model_data = {}
    for key in model_order:
        dirname = _MULTISTEP_DIRS[key]
        base = os.path.join(exp_root, dirname)
        if not os.path.isdir(base):
            base = os.path.join("./outputs", dirname)
        if not os.path.isdir(base):
            print(f"  Skipping {key}: directory not found")
            continue
        stats = _load_multistep_stats(base)
        if stats is not None:
            model_data[key] = stats
            print(f"  Loaded {key}: {stats['n_samples']} samples, "
                  f"{len(stats['store_time_ids'])} steps")

    if len(model_data) < 2:
        print("  Skipping sink_dynamics_multimodel: need ≥2 models")
        return

    # ── Figure: 2 rows ───────────────────────────────────────────────────
    fig, (ax_int, ax_pos) = plt.subplots(
        1, 2, figsize=(FIG_FULL, 2.4), dpi=300,
        gridspec_kw={"wspace": 0.25},
    )

    # (a) Sink intensity curves ───────────────────────────────────────────
    for key in model_order:
        if key not in model_data:
            continue
        md = model_data[key]
        disp = _MODEL_DISPLAY[key]
        prog = md["progress"]
        mean = md["intensity_mean"]
        std = md["intensity_std"]

        ax_int.plot(prog, mean, disp["marker"] + "-", color=disp["color"],
                    lw=2, ms=5, label=f'{disp["label"]} ({disp["pe"]})', zorder=3)
        ax_int.fill_between(prog, mean - std, mean + std,
                            color=disp["color"], alpha=0.12, zorder=1)

    ax_int.set_xlabel("Denoising Progress")
    ax_int.set_ylabel("Top-1 Key Importance")
    ax_int.set_title("(a) Sink Intensity Across Models", fontweight="bold")
    ax_int.legend(fontsize=FS_LABEL, frameon=True, framealpha=0.9, loc="upper left")
    _clean_ax(ax_int)

    # (b) Token position diversity ────────────────────────────────────────
    bar_width = 0.15
    n_models = len(model_data)
    active_keys = [k for k in model_order if k in model_data]

    # Use the max number of steps across models for x-axis grouping
    # Normalize step labels to ["Early", "Mid-Early", "Mid", "Mid-Late", "Late"]
    step_labels = ["Early", "Mid-\nEarly", "Mid", "Mid-\nLate", "Late"]

    for mi, key in enumerate(active_keys):
        md = model_data[key]
        disp = _MODEL_DISPLAY[key]
        n_steps = len(md["position_diversity"])
        # Pad or trim to 5 steps
        div = md["position_diversity"][:5]
        while len(div) < 5:
            div.append(0)
        x = np.arange(5) + mi * bar_width
        ax_pos.bar(x, div, bar_width, color=disp["color"], alpha=0.8,
                   label=disp["label"], edgecolor="white", linewidth=0.5)

    ax_pos.set_xticks(np.arange(5) + bar_width * (n_models - 1) / 2)
    ax_pos.set_xticklabels(step_labels, fontsize=FS_TICK)
    ax_pos.set_xlabel("Denoising Stage")
    ax_pos.set_ylabel("Unique Top-1 Positions\n(across 100 images)", fontsize=FS_LABEL)
    ax_pos.set_title("(b) Sink Position Stability", fontweight="bold")
    ax_pos.legend(fontsize=FS_LABEL, frameon=True, framealpha=0.9, ncol=2, loc="upper left")
    _clean_ax(ax_pos)

    _save_fig(fig, out_dir, "fig_sink_dynamics_multimodel.pdf")


def plot_fig_norm_paradox_panel(data, out_dir):
    """Whensink-style mixed panel: top row line charts + bottom row heatmaps.

    Replaces the simpler fig_norm_paradox and fig_same_norm_diff_dist with
    a single information-dense figure.

    Layout (3 cols × 2 rows):
        Top:    (a) Hidden States L2-norm  (b) Key L2-norm  (c) Key Importance
        Bottom: (d) K Sink (heatmap)       (e) K Random (heatmap)  + colorbar
    """
    required = ["layer_list", "Hidden_States_sink_curve", "Hidden_States_rand_curve",
                "K_sink_curve", "K_rand_curve",
                "Attention_sink_curve", "Attention_rand_curve",
                "k_sink_heatmap", "k_rand_heatmap"]
    if not _check_keys(data, required, "fig_norm_paradox_panel"):
        return

    layer_list = data["layer_list"]
    n = len(layer_list)
    x = np.arange(n)

    hs_sink = np.array(data["Hidden_States_sink_curve"])
    hs_rand = np.array(data["Hidden_States_rand_curve"])
    k_sink = np.array(data["K_sink_curve"])
    k_rand = np.array(data["K_rand_curve"])
    attn_sink = np.array(data["Attention_sink_curve"])
    attn_rand = np.array(data["Attention_rand_curve"])
    k_sink_hm = np.array(data["k_sink_heatmap"])
    k_rand_hm = np.array(data["k_rand_heatmap"])

    tick_pos, tick_lbl = _layer_ticks(layer_list, every=6)

    fig = plt.figure(figsize=(FIG_FULL, 3.0), dpi=300)
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.03],
        height_ratios=[1.0, 1.2],
        wspace=0.25, hspace=0.45,
    )

    # ── Top row: line charts ──

    line_kw_sink = dict(color=C_RED, lw=0.9, ms=2, marker="o", label="Sink token")
    line_kw_rand = dict(color=C_BLUE, lw=0.9, ms=2, marker="s", label="Random token")

    # (a) Hidden States L2-norm
    ax_hs = fig.add_subplot(gs[0, 0])
    ax_hs.plot(x, hs_sink, **line_kw_sink)
    ax_hs.plot(x, hs_rand, **line_kw_rand)
    ax_hs.set_ylabel("$\\ell_2$-norm")
    ax_hs.set_title("(a) Hidden States", fontweight="bold")
    ax_hs.set_xticks(tick_pos)
    ax_hs.set_xticklabels(tick_lbl, fontsize=FS_TICK)
    ax_hs.legend(fontsize=FS_LABEL, frameon=True, framealpha=0.9, loc="best")
    _clean_ax(ax_hs)

    # (b) Key L2-norm
    ax_k = fig.add_subplot(gs[0, 1])
    ax_k.plot(x, k_sink, **line_kw_sink)
    ax_k.plot(x, k_rand, **line_kw_rand)
    ax_k.set_title("(b) Key $\\ell_2$-norm", fontweight="bold")
    ax_k.set_xticks(tick_pos)
    ax_k.set_xticklabels(tick_lbl, fontsize=FS_TICK)
    _clean_ax(ax_k)

    # Highlight the paradox: norms are comparable
    k_ratio = k_sink / np.clip(k_rand, EPS, None)
    mean_ratio = np.mean(k_ratio)
    ax_k.text(0.95, 0.08, f"ratio \u2248 {mean_ratio:.2f}\u00d7",
              transform=ax_k.transAxes, fontsize=FS_LABEL, ha="right", va="bottom",
              color=C_GRAY, fontstyle="italic",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_GRAY, alpha=0.6, lw=0.5))

    # (c) Key Importance / Attention Score
    ax_attn = fig.add_subplot(gs[0, 2])
    ax_attn.plot(x, attn_sink, **line_kw_sink)
    ax_attn.plot(x, attn_rand, **line_kw_rand)
    ax_attn.set_title("(c) Key Importance", fontweight="bold")
    ax_attn.set_xticks(tick_pos)
    ax_attn.set_xticklabels(tick_lbl, fontsize=FS_TICK)
    _clean_ax(ax_attn)

    # Annotate peak ratio
    attn_ratio = attn_sink / np.clip(attn_rand, EPS, None)
    peak_idx = int(np.argmax(attn_ratio))
    ax_attn.annotate(
        f"{attn_ratio[peak_idx]:.0f}\u00d7",
        xy=(peak_idx, attn_sink[peak_idx]),
        xytext=(peak_idx + 1, attn_sink[peak_idx] * 0.85),
        fontsize=FS_LABEL, fontweight="bold", color=C_RED,
        arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=0.8),
    )

    # ── Bottom row: heatmaps ──

    vmin = min(k_sink_hm.min(), k_rand_hm.min())
    vmax = max(k_sink_hm.max(), k_rand_hm.max())

    # (d) K Sink heatmap
    ax_ks = fig.add_subplot(gs[1, 0:2])
    im = ax_ks.imshow(
        k_sink_hm.T, aspect="auto", cmap="magma",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax_ks.set_title("(d) Key channel norms — Sink token", fontweight="bold")
    ax_ks.set_ylabel("Channel (high freq \u2192 low freq \u2193)", fontsize=FS_LABEL)
    ax_ks.set_xlabel("Block")
    ax_ks.set_xticks(tick_pos)
    ax_ks.set_xticklabels(tick_lbl, fontsize=FS_TICK)

    # (e) K Random heatmap
    ax_kr = fig.add_subplot(gs[1, 2])
    ax_kr.imshow(
        k_rand_hm.T, aspect="auto", cmap="magma",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax_kr.set_title("(e) Random token", fontweight="bold")
    ax_kr.set_xlabel("Block")
    ax_kr.set_xticks(tick_pos)
    ax_kr.set_xticklabels(tick_lbl, fontsize=FS_TICK)
    ax_kr.set_yticks([])

    # Shared colorbar
    cax = fig.add_subplot(gs[1, 3])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=FS_SMALL)
    cbar.set_label("Mean |Key|", fontsize=FS_LABEL)

    _save_fig(fig, out_dir, "fig_norm_paradox_panel.pdf")


def plot_fig_token_attention_bar(out_dir, per_token_json=None):
    """Padding-tone style: per-token attention strength bar chart.

    Two vertically stacked panels — FLUX (top) and PixArt (bottom).
    Context tokens only, color-coded by type.

    Data source: per_token_json or demo data.
    JSON format: {"flux": {"attn": [float...]}, "pixart": {"attn": [float...]}}
    """
    is_demo = False
    if per_token_json and os.path.exists(per_token_json):
        with open(per_token_json, encoding="utf-8") as f:
            data = json.load(f)
        flux_attn = np.array(data["flux"]["attn"][:FLUX_CONTEXT_LEN])
        pixart_attn = np.array(data["pixart"]["attn"][:PIXART_CONTEXT_LEN])
    else:
        print("  Using demo data for token attention bar")
        is_demo = True
        rng = np.random.default_rng(42)
        flux_attn = _generate_demo_flux(rng)
        pixart_attn = _generate_demo_pixart(rng)

    fig, (ax_flux, ax_pixart) = plt.subplots(
        2, 1, figsize=(FIG_FULL, 1.8), dpi=300,
        gridspec_kw={"hspace": 0.3},
    )

    fig.text(0.5, 1.0, "Diffusion attention strength per token",
             ha="center", va="top", fontweight="bold")

    # FLUX: skip middle PAD range (80-490) for readability
    _draw_token_bar(
        ax_flux, flux_attn, _flux_token_color, "FLUX",
        skip_range=(80, 490),
        annotate_eos=FLUX_EOS_IDX,
    )
    ax_flux.set_ylabel("Attn", fontsize=FS_LABEL, labelpad=2)

    # PixArt: show all context tokens (only 120)
    _draw_token_bar(
        ax_pixart, pixart_attn, _pixart_token_color, "PixArt",
        annotate_eos=PIXART_EOS_IDX,
    )
    ax_pixart.set_ylabel("Attn", fontsize=FS_LABEL, labelpad=2)
    ax_pixart.set_xlabel("Token index")
    ax_pixart.tick_params(axis="x", bottom=True, labelbottom=True)

    # X-axis ticks for PixArt
    tick_step = 20
    ax_pixart.set_xticks(range(0, len(pixart_attn), tick_step))
    ax_pixart.set_xticklabels(
        [str(i) for i in range(0, len(pixart_attn), tick_step)],
        fontsize=FS_LABEL)

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_RED, label="<EOS> (sink)"),
        mpatches.Patch(color=C_ORANGE, label="<PAD> / <BOS>"),
        mpatches.Patch(color=C_BLUE, label="Semantic text"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=3, fontsize=FS_LABEL, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    suffix = "_demo" if is_demo else ""
    _save_fig(fig, out_dir, f"fig_token_attention_bar{suffix}.pdf")


# ── Sink-over-time figures (exp09c/d) ──

def plot_fig_sink_over_time_heatmap(out_dir, json_path, model_label=None):
    """2D heatmap: token position x denoising step from sink_over_time_data.json."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    ki_matrix = np.array(data["ki_matrix_aggregated"])  # (seq_len, num_steps)
    store_time_ids = data["store_time_ids"]
    seq_len = data["seq_len"]
    num_image_tokens = data["num_image_tokens"]
    model_name = model_label or data.get("model", "")

    fig, ax = plt.subplots(figsize=(FIG_HALF, 2.2), dpi=300)
    im = ax.imshow(
        ki_matrix, aspect="auto", cmap="magma", interpolation="nearest",
        origin="lower",
    )
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Token position")

    num_steps = len(store_time_ids)
    tick_step = max(1, num_steps // 8)
    ax.set_xticks(list(range(0, num_steps, tick_step)))
    ax.set_xticklabels([str(store_time_ids[i]) for i in range(0, num_steps, tick_step)], fontsize=FS_LABEL)

    num_text = seq_len - num_image_tokens
    if num_text > 0:
        ax.axhline(y=num_text - 0.5, color="white", linewidth=0.8, linestyle="--", alpha=0.6)

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    _clean_ax(ax)
    fig.tight_layout()

    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    fig.savefig(os.path.join(out_dir, f"fig_sink_over_time_heatmap_{safe_name}.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig_sink_over_time_heatmap_{safe_name}.pdf")


def plot_fig_sink_intensity_curve(out_dir, json_path, model_label=None):
    """Sink intensity (max & mean of top-k) over denoising steps."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    store_time_ids = data["store_time_ids"]
    per_step = data["per_step_sinks"]
    model_name = model_label or data.get("model", "")
    top_k = data.get("top_k", 10)

    steps = []
    max_scores = []
    mean_scores = []
    for sid in store_time_ids:
        sinks = per_step.get(str(sid), [])
        if not sinks:
            continue
        scores = [s["score"] for s in sinks[:top_k]]
        steps.append(sid)
        max_scores.append(max(scores))
        mean_scores.append(float(np.mean(scores)))

    if not steps:
        print(f"  [SKIP] No sink data for intensity curve")
        return

    fig, ax = plt.subplots(figsize=(FIG_HALF, 2.2), dpi=300)
    ax.plot(range(len(steps)), max_scores, "s-", color=C_RED, lw=0.9, ms=2, label="Top-1")
    ax.plot(range(len(steps)), mean_scores, "o-", color=C_BLUE, lw=0.9, ms=2, label=f"Mean top-{top_k}")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], fontsize=FS_LABEL)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Key importance")
    ax.legend(fontsize=FS_LABEL, framealpha=0.8)
    _clean_ax(ax)
    fig.tight_layout()

    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    fig.savefig(os.path.join(out_dir, f"fig_sink_intensity_{safe_name}.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig_sink_intensity_{safe_name}.pdf")


def plot_fig_sink_stability(out_dir, json_path, model_label=None):
    """IoU of top-k sinks between consecutive denoising steps."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    stability = data.get("stability", [])
    model_name = model_label or data.get("model", "")
    if not stability:
        print(f"  [SKIP] No stability data")
        return

    ious = [s["iou"] for s in stability]
    labels = [f"{s['from_step']}" for s in stability]

    fig, ax = plt.subplots(figsize=(FIG_HALF, 2.0), dpi=300)
    bars = ax.bar(range(len(ious)), ious, color=C_BLUE, alpha=0.85)
    ax.set_xticks(range(len(ious)))
    ax.set_xticklabels(labels, fontsize=FS_LABEL, rotation=45, ha="right")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Sink IoU")
    ax.set_ylim(0, 1.05)
    _clean_ax(ax)
    fig.tight_layout()

    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    fig.savefig(os.path.join(out_dir, f"fig_sink_stability_{safe_name}.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig_sink_stability_{safe_name}.pdf")


def plot_fig_sink_layer_x_step(out_dir, json_path, model_label=None):
    """Heatmap: layer x step with max key importance."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    store_time_ids = data["store_time_ids"]
    sink_layers = data["sink_layers"]
    per_layer_ki = data["per_layer_ki"]
    model_name = model_label or data.get("model", "")

    num_steps = len(store_time_ids)
    num_layers = len(sink_layers)
    matrix = np.zeros((num_layers, num_steps))

    for i, layer_idx in enumerate(sink_layers):
        layer_data = per_layer_ki.get(str(layer_idx), {})
        for j, sid in enumerate(store_time_ids):
            ki_vec = layer_data.get(str(sid), None)
            if ki_vec is not None:
                matrix[i, j] = max(ki_vec)

    fig, ax = plt.subplots(figsize=(FIG_HALF, max(2.0, num_layers * 0.18)), dpi=300)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Layer")
    tick_step = max(1, num_steps // 8)
    ax.set_xticks(list(range(0, num_steps, tick_step)))
    ax.set_xticklabels([str(store_time_ids[i]) for i in range(0, num_steps, tick_step)], fontsize=FS_LABEL)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(l) for l in sink_layers], fontsize=FS_LABEL)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()

    safe_name = model_name.replace("/", "_").replace(" ", "_").lower()
    fig.savefig(os.path.join(out_dir, f"fig_sink_layer_x_step_{safe_name}.pdf"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> fig_sink_layer_x_step_{safe_name}.pdf")


def plot_fig_prepost_rope(out_dir, json_path=None, exp_root="./outputs/Attn_Project_Exp_by_id"):
    """Fig 7: Pre-RoPE vs Post-RoPE Q·K dot product comparison.

    Reads sink_rope_analysis_results_t{step}.json produced by
    task_sink_rope_prepost_qk_analysis (exp05c), averages across heads,
    and draws a 1×2 panel: left = Pre-RoPE, right = Post-RoPE.
    """
    # Discover JSON
    if json_path and os.path.exists(json_path):
        jpath = json_path
    else:
        candidates = list(Path(exp_root).glob(
            "exp05c_flux*/FLUX*/*/sink_rope_prepost_qk/sink_rope_analysis_results_t*.json"
        ))
        if not candidates:
            candidates = list(Path(exp_root).glob(
                "exp05c_flux*/sink_rope_prepost_qk/sink_rope_analysis_results_t*.json"
            ))
        if not candidates:
            # Also check flat outputs dir
            candidates = list(Path("./outputs").glob(
                "exp05c_*/sink_rope_analysis_results_t*.json"
            ))
        if not candidates:
            print("  Skipping prepost_rope: no JSON found")
            return
        jpath = str(sorted(candidates)[-1])  # Take the last step

    print(f"  prepost_rope JSON: {jpath}")
    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)

    # Extract step number from data or filename
    task_name = data.get("task_name", "")
    step_match = None
    for part in Path(jpath).stem.split("_"):
        if part.startswith("t") and part[1:].isdigit():
            step_match = int(part[1:])
    if step_match is None:
        step_match = 27  # default

    layer_analysis = data.get("layer_analysis", {})
    if not layer_analysis:
        print("  Skipping prepost_rope: empty layer_analysis")
        return

    layer_indices = []
    sink_prerope, sink_postrope = [], []
    rand_prerope, rand_postrope = [], []

    for layer_key in sorted(layer_analysis.keys(), key=lambda k: int(k.split("_")[1])):
        layer_data = layer_analysis[layer_key]
        layer_idx = int(layer_key.split("_")[1])
        layer_indices.append(layer_idx)

        head_sink_pre, head_sink_post = [], []
        head_rand_pre, head_rand_post = [], []
        for head_key, head_data in layer_data["heads_analysis"].items():
            head_sink_pre.append(
                head_data["prerope_q@k_analysis"]["outlier_as_key"]["value"])
            head_sink_post.append(
                head_data["postrope_q@k_analysis"]["outlier_as_key_dot_mean"]["value"])
            head_rand_pre.append(
                head_data["prerope_q@k_analysis"]["random_100_as_key"]["value"])
            head_rand_post.append(
                head_data["postrope_q@k_analysis"]["random_100_as_key_dot_mean"]["value"])

        sink_prerope.append(sum(head_sink_pre) / len(head_sink_pre))
        sink_postrope.append(sum(head_sink_post) / len(head_sink_post))
        rand_prerope.append(sum(head_rand_pre) / len(head_rand_pre))
        rand_postrope.append(sum(head_rand_post) / len(head_rand_post))

    x = np.arange(len(layer_indices))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_FULL, 2.4), sharey=True)

    # Left: Pre-RoPE
    ax1.plot(x, sink_prerope, "o-", color=C_RED, lw=0.9, ms=2, label="Sink tokens")
    ax1.plot(x, rand_prerope, "s-", color=C_BLUE, lw=0.9, ms=2, label="Random tokens")
    ax1.set_title("Pre-RoPE  Q\u00b7K", fontweight="bold")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean dot product")
    ax1.legend(fontsize=FS_LABEL, frameon=True, framealpha=0.9)
    tick_pos, tick_lbl = _layer_ticks(layer_indices)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_lbl)
    _clean_ax(ax1)

    # Right: Post-RoPE
    ax2.plot(x, sink_postrope, "o-", color=C_RED, lw=0.9, ms=2, label="Sink tokens")
    ax2.plot(x, rand_postrope, "s-", color=C_BLUE, lw=0.9, ms=2, label="Random tokens")
    ax2.set_title("Post-RoPE  Q\u00b7K", fontweight="bold")
    ax2.set_xlabel("Layer")
    ax2.legend(fontsize=FS_LABEL, frameon=True, framealpha=0.9)
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lbl)
    _clean_ax(ax2)

    _save_fig(fig, out_dir, f"fig_prepost_rope_step{step_match:02d}.pdf")


def plot_fig_cross_attn_comparison(out_dir, exp_root="./outputs/Attn_Project_Exp_by_id"):
    """Supp figure: Cross-attn sink comparison for separated DiTs.

    Reads cross_attn_key_importance.json from exp00f (Wan) and exp00g (PixArt).
    Produces a composite figure with:
      - Row 1 (PixArt-α): generated image + heatmap (layers × tokens) + annotated bar
      - Row 2 (Wan2.1): heatmap (layers × tokens, truncated at semantic boundary) + annotated bar
    """
    from matplotlib.gridspec import GridSpec
    import matplotlib.image as mpimg

    # ── Discover JSON files ──
    pattern = os.path.join(exp_root, "exp00[fg]_*", "*", "*",
                           "cross_attn_sink_analysis", "stats",
                           "cross_attn_key_importance.json")
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        pattern2 = os.path.join(exp_root, "exp00[fg]_*",
                                "cross_attn_sink_analysis", "stats",
                                "cross_attn_key_importance.json")
        json_files = sorted(glob.glob(pattern2))
    if not json_files:
        print("  Skipping cross_attn_comparison: no JSON found")
        return

    # ── Load data for each model ──
    model_data = {}
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        parts = Path(jf).parts
        exp_part = [p for p in parts if p.startswith("exp00")]
        if exp_part and "wan" in exp_part[0].lower():
            key = "wan"
        else:
            key = "pixart"
        step_key = list(data.keys())[0]
        model_data[key] = {"data": data[step_key], "path": jf}

    if "pixart" not in model_data or "wan" not in model_data:
        print("  Skipping cross_attn_comparison: need both PixArt and Wan data")
        return

    # ── Helper: build heatmap matrix from layers data ──
    def _build_heatmap(layers_data):
        layer_ids = [l["layer"] for l in layers_data]
        ki_matrix = np.array([l["full_importance"] for l in layers_data])
        return layer_ids, ki_matrix

    # ── Helper: find generated image near JSON ──
    def _find_image(json_path):
        parent = Path(json_path).parent.parent.parent
        for ext in ("*.jpg", "*.png"):
            imgs = list(parent.glob(ext))
            if imgs:
                return str(imgs[0])
            for sub in parent.iterdir():
                if sub.is_dir():
                    imgs = list(sub.glob(ext))
                    if imgs:
                        return str(imgs[0])
        return None

    # ── Extract data ──
    px_layers = model_data["pixart"]["data"]["layers"]
    px_layer_ids, px_hm = _build_heatmap(px_layers)
    px_txt_len = px_hm.shape[1]
    px_avg = px_hm.mean(axis=0)
    px_spikiness = float(px_avg.max() / (px_avg.mean() + 1e-8))

    wan_layers = model_data["wan"]["data"]["layers"]
    wan_layer_ids, wan_hm = _build_heatmap(wan_layers)
    wan_txt_len = wan_hm.shape[1]
    wan_avg = wan_hm.mean(axis=0)
    wan_spikiness = float(wan_avg.max() / (wan_avg.mean() + 1e-8))

    # Truncate Wan display at PAD boundary.
    # PAD tokens have near-identical importance across all layers (constant ≈0.00199).
    # Scan from the end to find where the constant-value suffix begins.
    wan_pad_start = wan_txt_len
    pad_val = wan_hm[0, -1]  # reference PAD value from layer 0, last token
    for i in range(wan_txt_len - 1, 0, -1):
        # Check if this token is still PAD-like across all layers
        col = wan_hm[:, i]
        if np.std(col) > 0.001 or abs(col.mean() - pad_val) > 0.001:
            wan_pad_start = i + 1
            break
    wan_semantic_end = min(wan_pad_start + 3, wan_txt_len)  # small margin
    wan_hm_trunc = wan_hm[:, :wan_semantic_end]
    wan_avg_trunc = wan_avg[:wan_semantic_end]

    # ── Find PixArt generated image ──
    pixart_img_path = _find_image(model_data["pixart"]["path"])

    # ── Build figure ──
    # Layout: 2 rows.
    #   Row 0 (PixArt): [image | heatmap | bar]  widths ~1:3:2
    #   Row 1 (Wan):    [heatmap          | bar]  widths ~4:2
    fig = plt.figure(figsize=(FIG_FULL, 5.0))

    if pixart_img_path:
        gs = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.4,
                      height_ratios=[1, 1])
        ax_img = fig.add_subplot(gs[0, 0:1])
        ax_px_hm = fig.add_subplot(gs[0, 1:4])
        ax_px_bar = fig.add_subplot(gs[0, 4:6])
        ax_wan_hm = fig.add_subplot(gs[1, 0:4])
        ax_wan_bar = fig.add_subplot(gs[1, 4:6])
    else:
        gs = GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.4,
                      height_ratios=[1, 1])
        ax_img = None
        ax_px_hm = fig.add_subplot(gs[0, 0:3])
        ax_px_bar = fig.add_subplot(gs[0, 3:5])
        ax_wan_hm = fig.add_subplot(gs[1, 0:3])
        ax_wan_bar = fig.add_subplot(gs[1, 3:5])

    # ── (a) PixArt generated image ──
    if ax_img is not None and pixart_img_path:
        img = mpimg.imread(pixart_img_path)
        ax_img.imshow(img)
        ax_img.set_title("Generated", fontsize=FS_LABEL, fontweight="bold")
        ax_img.axis("off")

    # ── (b) PixArt heatmap ──
    im_px = ax_px_hm.imshow(px_hm, aspect="auto", cmap="magma",
                             interpolation="nearest")
    ax_px_hm.set_xlabel("Text Token Index")
    ax_px_hm.set_ylabel("Layer")
    ax_px_hm.set_yticks(range(len(px_layer_ids)))
    ax_px_hm.set_yticklabels([str(l) for l in px_layer_ids], fontsize=FS_LABEL)
    ax_px_hm.set_title(r"PixArt-$\alpha$ — Cross-Attn Key Importance (layers $\times$ tokens)",
                        fontweight="bold")
    # Annotate EOS column
    eos_idx = 18
    ax_px_hm.axvline(x=eos_idx, color="white", ls="--", lw=1.0, alpha=0.8)
    # Place EOS label below the heatmap (inside plot area, bottom-right of the column)
    ax_px_hm.annotate("EOS", xy=(eos_idx, len(px_layer_ids) - 0.5),
                       xytext=(eos_idx + 8, len(px_layer_ids) - 1),
                       fontsize=FS_LABEL, fontweight="bold", color="white",
                       arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
                       bbox=dict(boxstyle="round,pad=0.15", fc=C_RED, ec="none", alpha=0.9))
    cb_px = fig.colorbar(im_px, ax=ax_px_hm, shrink=0.75, pad=0.02)
    cb_px.ax.tick_params(labelsize=FS_LABEL)

    # ── (c) PixArt bar chart with annotation ──
    bar_colors_px = []
    for i in range(px_txt_len):
        if i == eos_idx:
            bar_colors_px.append(C_RED)
        elif i < eos_idx:
            bar_colors_px.append(C_BLUE)
        else:
            bar_colors_px.append(C_GRAY)
    ax_px_bar.bar(range(px_txt_len), px_avg, color=bar_colors_px, alpha=0.85, width=1.0)
    ax_px_bar.set_xlabel("Text Token Index")
    ax_px_bar.set_ylabel("Avg Key Importance")
    ax_px_bar.set_title(f"Spikiness = {px_spikiness:.1f}×", fontsize=FS_LABEL)
    # Annotate EOS peak
    ax_px_bar.annotate(f"<EOS> (idx {eos_idx})\n{px_avg[eos_idx]:.1%} attn",
                       xy=(eos_idx, px_avg[eos_idx]),
                       xytext=(eos_idx + 20, px_avg[eos_idx] * 0.85),
                       fontsize=FS_LABEL, color=C_RED, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
    _clean_ax(ax_px_bar)
    # Legend
    from matplotlib.patches import Patch
    px_legend = [Patch(fc=C_RED, label="<EOS>"),
                 Patch(fc=C_BLUE, label="Semantic"),
                 Patch(fc=C_GRAY, label="<PAD>")]
    ax_px_bar.legend(handles=px_legend, fontsize=FS_LABEL, loc="upper right",
                     framealpha=0.9)

    # ── (d) Wan heatmap (truncated at semantic boundary) ──
    # Use a capped vmax for better contrast on the distributed pattern
    wan_p95 = np.percentile(wan_hm_trunc, 99)
    im_wan = ax_wan_hm.imshow(wan_hm_trunc, aspect="auto", cmap="magma",
                               interpolation="nearest",
                               vmin=0, vmax=wan_p95 * 1.2)
    ax_wan_hm.set_xlabel("Text Token Index")
    ax_wan_hm.set_ylabel("Layer")
    ax_wan_hm.set_yticks(range(len(wan_layer_ids)))
    ax_wan_hm.set_yticklabels([str(l) for l in wan_layer_ids], fontsize=FS_LABEL)
    ax_wan_hm.set_title(f"Wan2.1 — Cross-Attn Key Importance "
                        f"(first {wan_semantic_end} tokens; PAD truncated)",
                        fontweight="bold")
    # Mark PAD boundary
    if wan_pad_start < wan_semantic_end:
        ax_wan_hm.axvline(x=wan_pad_start - 0.5, color=C_GRAY, ls="--",
                          lw=0.8, alpha=0.7)
        ax_wan_hm.text(wan_pad_start + 0.5, 0.3, "PAD→", fontsize=FS_TICK,
                       color=C_GRAY, va="center")
    cb_wan = fig.colorbar(im_wan, ax=ax_wan_hm, shrink=0.75, pad=0.02)
    cb_wan.ax.tick_params(labelsize=FS_LABEL)

    # ── (e) Wan bar chart with semantic token annotations ──
    # Color: highlight top semantic tokens
    wan_top10 = set(np.argsort(wan_avg_trunc)[::-1][:10].tolist())
    bar_colors_wan = []
    for i in range(len(wan_avg_trunc)):
        if i in wan_top10:
            bar_colors_wan.append(C_RED)
        else:
            bar_colors_wan.append(C_BLUE)
    ax_wan_bar.bar(range(len(wan_avg_trunc)), wan_avg_trunc,
                   color=bar_colors_wan, alpha=0.85, width=1.0)
    ax_wan_bar.set_xlabel("Text Token Index")
    ax_wan_bar.set_ylabel("Avg Key Importance")
    ax_wan_bar.set_title(f"Spikiness = {wan_spikiness:.1f}×", fontsize=FS_LABEL)
    # Annotate top-1 token
    wan_top1 = int(np.argmax(wan_avg_trunc))
    n_trunc = len(wan_avg_trunc)
    # Place annotation to the left if top token is near the right edge
    if wan_top1 > n_trunc * 0.6:
        xt = wan_top1 - n_trunc * 0.2
    else:
        xt = wan_top1 + n_trunc * 0.1
    ax_wan_bar.annotate(f"Token {wan_top1}\n{wan_avg_trunc[wan_top1]:.4f}",
                        xy=(wan_top1, wan_avg_trunc[wan_top1]),
                        xytext=(xt, wan_avg_trunc[wan_top1] * 0.75),
                        fontsize=FS_LABEL, color=C_RED, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
    _clean_ax(ax_wan_bar)
    wan_legend = [Patch(fc=C_RED, label="Top-10 semantic"),
                  Patch(fc=C_BLUE, label="Other tokens")]
    ax_wan_bar.legend(handles=wan_legend, fontsize=FS_LABEL, loc="upper right",
                      framealpha=0.9)

    # ── Panel labels ──
    label_kw = dict(fontsize=FS_TITLE, fontweight="bold", va="top", ha="left")
    if ax_img is not None:
        ax_img.text(-0.15, 1.08, "(a)", transform=ax_img.transAxes, **label_kw)
    ax_px_hm.text(-0.08, 1.08, "(b)" if ax_img else "(a)",
                  transform=ax_px_hm.transAxes, **label_kw)
    ax_px_bar.text(-0.12, 1.08, "(c)" if ax_img else "(b)",
                   transform=ax_px_bar.transAxes, **label_kw)
    ax_wan_hm.text(-0.08, 1.08, "(d)" if ax_img else "(c)",
                   transform=ax_wan_hm.transAxes, **label_kw)
    ax_wan_bar.text(-0.12, 1.08, "(e)" if ax_img else "(d)",
                    transform=ax_wan_bar.transAxes, **label_kw)

    _save_fig(fig, out_dir, "fig_cross_attn_comparison.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", default="./outputs/Attn_Project_Exp_by_id")
    parser.add_argument("--out_dir", default="./paper/figures/generated")
    parser.add_argument("--sink_value_json", default=None,
                        help="Path to step_*_sink_curves.json for fig:sink_value_info")
    parser.add_argument("--per_token_json", default=None,
                        help="Path to per-token attention JSON (or omit for demo data)")
    parser.add_argument("--flux_image", default=None, help="Path to FLUX generated image")
    parser.add_argument("--spatial_json", default=None,
                        help="Path to flux_keyimportance_stats.json for spatial overlay")
    parser.add_argument("--pixart_image", default=None, help="Path to PixArt generated image")
    parser.add_argument("--wan_image", default=None, help="Path to Wan video first frame")
    parser.add_argument("--flux_global_txt", default=None,
                        help="Path to FLUX GLOBAL_over_steps_avg_top100_softmax.txt")
    parser.add_argument("--prepost_rope_json", default=None,
                        help="Path to sink_rope_analysis_results_t*.json for fig:pre_post_rope")
    parser.add_argument("--only", default=None,
                        help="Only generate specific figure, e.g. token_attention_bar, sink_types")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run_all = args.only is None

    # Fig: Sink Identity (clean 3-panel composite)
    if run_all or args.only == "sink_identity":
        print("=== Sink Identity (3-panel composite) ===")
        plot_fig_sink_identity(args.out_dir, exp_root="./outputs")

    # Fig 6: Sink types across architectures (legacy 3-panel)
    if args.only == "sink_types":
        print("=== Sink types (legacy) ===")
        plot_fig_sink_types(args.out_dir, exp_root="./outputs",
                            flux_image=args.flux_image,
                            pixart_image=args.pixart_image,
                            wan_image=args.wan_image,
                            flux_global_txt=args.flux_global_txt)

    # Token attention bar (padding-tone style)
    if run_all or args.only == "token_attention_bar":
        print("=== Token attention bar ===")
        plot_fig_token_attention_bar(args.out_dir, args.per_token_json)

    # ── Load sink_curves JSON once (shared by band_norm, norm_paradox, etc.) ──
    sink_json = None
    sink_data = None
    if args.sink_value_json and os.path.exists(args.sink_value_json):
        sink_json = args.sink_value_json
    else:
        candidates_05 = list(Path(args.exp_root).glob(
            "exp05a_flux*/FLUX*/*/sink_tokens_analysis/stats/step_*_sink_curves.json"
        ))
        if not candidates_05:
            direct = Path("./outputs/exp05a_flux_sink_value_info/step_27_sink_curves.json")
            if direct.exists():
                candidates_05 = [direct]
        if candidates_05:
            sink_json = str(candidates_05[0])
    if sink_json:
        print(f"Found sink_curves: {sink_json}")
        with open(sink_json, encoding="utf-8") as f:
            sink_data = json.load(f)

    # Band norm heatmap (Round-and-Round style)
    if run_all or args.only == "band_norm":
        print("=== Band norm heatmap ===")
        plot_fig_band_norm(args.out_dir, sink_curves_data=sink_data)

    # Wan 3D RoPE bottleneck heatmap (Fig 18)
    if run_all or args.only == "wan_3d_rope":
        print("=== Wan 3D RoPE bottleneck ===")
        plot_fig_wan_3d_rope_bottleneck(args.out_dir)

    # Multi-model sink dynamics (intensity + position stability)
    if run_all or args.only == "sink_dynamics":
        print("=== Sink dynamics (multi-model) ===")
        plot_fig_sink_dynamics_multimodel(args.out_dir, args.exp_root)

    # Band comparison: FLUX vs Wan (all layers × all channels)
    if run_all or args.only == "band_comparison":
        print("=== Band comparison (FLUX vs Wan) ===")
        plot_fig_band_comparison(args.out_dir)

    # APE vs RoPE band comparison (compact 1×N Key-only)
    if run_all or args.only == "ape_vs_rope_band":
        print("=== APE vs RoPE band comparison ===")
        plot_fig_ape_vs_rope_band(args.out_dir)

    # Layerwise multi-model comparison (whensink style)
    if run_all or args.only == "layerwise_multimodel":
        print("=== Layerwise multi-model ===")
        plot_fig_layerwise_multimodel(args.out_dir)

    # Sink temporal robustness (V3: overlaid curves + temporal flatness)
    if run_all or args.only == "sink_time_v3":
        print("=== Sink temporal robustness ===")
        exp00d_root = None
        exp00d_candidates = list(Path(args.exp_root).glob("exp00d_flux*/FLUX*/*/"))
        if exp00d_candidates:
            exp00d_root = str(exp00d_candidates[0])
        plot_fig_sink_time_v3(args.out_dir, exp_root=exp00d_root)

    # exp00d figures (legacy heatmap + layer + bar)
    if run_all or args.only in ("sink_time", "sink_layer", "key_importance"):
        candidates = list(Path(args.exp_root).glob("exp00d_flux*/FLUX*/*/"))
        if candidates:
            root = str(candidates[0])
            print(f"exp00d root: {root}")
            if run_all or args.only == "sink_time":
                plot_fig_sink_time(root, args.out_dir)
            if run_all or args.only == "sink_layer":
                plot_fig_sink_layer(root, args.out_dir)
        else:
            print("exp00d not found")
        if run_all or args.only == "key_importance":
            plot_fig_key_importance_bar(args.exp_root, args.out_dir,
                                       flux_image=args.flux_image)

    # exp05a figures (sink_curves JSON)
    if run_all or args.only in ("norm_paradox", "norm_paradox_panel", "same_norm", "sink_value", "cheating", "sink_overview"):
        if sink_data:
            if run_all or args.only == "norm_paradox":
                plot_fig_norm_paradox(sink_data, args.out_dir)
            if run_all or args.only == "norm_paradox_panel":
                plot_fig_norm_paradox_panel(sink_data, args.out_dir)
            if run_all or args.only == "same_norm":
                plot_fig_same_norm_diff_dist(sink_data, args.out_dir)
            if run_all or args.only == "sink_value":
                # Spatial key importance overlay (from exp00d 100-image stats)
                spatial_json = args.spatial_json
                if not spatial_json:
                    candidate = Path("./outputs/flux_keyimportance_stats.json")
                    if candidate.exists():
                        spatial_json = str(candidate)
                # Flux image for overlay background
                flux_img = args.flux_image
                if not flux_img:
                    puppy_candidates = list(Path("./paper/figures/concept_mixing").glob(
                        "puppy_screen_baseline_s40.jpg"))
                    if puppy_candidates:
                        flux_img = str(puppy_candidates[0])
                plot_fig_sink_value_info(sink_data, args.out_dir,
                                        spatial_json=spatial_json,
                                        flux_image=flux_img)
            if run_all or args.only == "cheating":
                plot_fig_cheating_strategy(sink_data, args.out_dir)
            if run_all or args.only == "mechanism_composite":
                plot_fig_mechanism_composite(sink_data, args.out_dir)
            if run_all or args.only == "sink_overview":
                # Reuse spatial data from sink_value block
                ov_spatial = args.spatial_json
                if not ov_spatial:
                    candidate = Path("./outputs/flux_keyimportance_stats.json")
                    if candidate.exists():
                        ov_spatial = str(candidate)
                ov_img = args.flux_image
                if not ov_img:
                    puppy_candidates = list(Path("./paper/figures/concept_mixing").glob(
                        "puppy_screen_baseline_s40.jpg"))
                    if puppy_candidates:
                        ov_img = str(puppy_candidates[0])
                plot_fig_sink_overview(sink_data, args.out_dir,
                                      spatial_json=ov_spatial,
                                      flux_image=ov_img)
        else:
            print("sink_curves JSON not found (run exp05a first)")

    # ── RoPE Phase Shift (analytical, no data needed) ──
    if run_all or args.only == "rope_phase":
        plot_fig_rope_phase_shift(args.out_dir)

    # ── Safe Harbor Composite (heatmap + key norm bars) ──
    if run_all or args.only == "safe_harbor":
        plot_fig_safe_harbor_composite(args.out_dir, data=sink_data)

    # ── Sink-over-time figures (exp09c/d) ──
    if run_all or args.only in ("sink_over_time", "sink_intensity", "sink_stability", "sink_layer_step"):
        sot_jsons = list(Path(args.exp_root).glob(
            "exp09*_sink_over_time/*/sink_over_time/stats/sink_over_time_data.json"
        ))
        if sot_jsons:
            for sot_json in sot_jsons:
                label = sot_json.parts[-4] if len(sot_json.parts) >= 4 else ""
                print(f"=== Sink-over-time: {label} ===")
                sot_path = str(sot_json)
                if run_all or args.only == "sink_over_time":
                    plot_fig_sink_over_time_heatmap(args.out_dir, sot_path, label)
                if run_all or args.only == "sink_intensity":
                    plot_fig_sink_intensity_curve(args.out_dir, sot_path, label)
                if run_all or args.only == "sink_stability":
                    plot_fig_sink_stability(args.out_dir, sot_path, label)
                if run_all or args.only == "sink_layer_step":
                    plot_fig_sink_layer_x_step(args.out_dir, sot_path, label)
        else:
            print("No sink_over_time data found (run exp09c/d first)")

    # ── Pre/Post RoPE dot product (exp05c) ──
    if run_all or args.only == "prepost_rope":
        print("=== Pre/Post RoPE (Fig 7) ===")
        plot_fig_prepost_rope(args.out_dir, json_path=args.prepost_rope_json,
                              exp_root=args.exp_root)

    # ── Cross-attention comparison (exp00f/g) ──
    if run_all or args.only == "cross_attn":
        print("=== Cross-attn comparison (supp) ===")
        plot_fig_cross_attn_comparison(args.out_dir, exp_root=args.exp_root)

    print(f"\nDone! -> {args.out_dir}")


if __name__ == "__main__":
    main()
