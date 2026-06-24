"""
Paper-ready figure palette and helpers.

Single source of truth for the red-theme color constants and shared
plotting utilities used across tools/ and src/tasks/ when generating
publication figures.  Import from here instead of copy-pasting hex codes.

Usage (from project root):
    from src.utils.paper_palette import C_RED, C_BLUE, clean_ax, save_fig
"""
import os

import matplotlib.pyplot as plt

# ── Colour constants (red theme, matches paper/colors.tex) ────────────
C_RED = "#A72B4A"       # Primary accent — EOS / sink
C_ORANGE = "#D4734E"    # Secondary — PAD tokens
C_BLUE = "#3D4F5F"      # Tertiary — text / random tokens
C_GRAY = "#8A9099"      # Quaternary — image tokens
C_DEEP = "#1A2456"      # Dark accent
C_BG = "white"

TOKEN_COLORS = {"eos": C_RED, "pad": C_ORANGE, "text": C_BLUE, "image": C_GRAY}
TOKEN_LABELS = {"eos": "<EOS>", "pad": "<PAD>", "text": "Semantic Text", "image": "Image"}

# ── NeurIPS figure size standards (inches) ────────────────────────────
FIG_FULL = 6.75    # full-width (double-column)
FIG_HALF = 3.25    # single-column

# ── Font size scale (pt) — NeurIPS-compact ────────────────────────────
FS_TITLE = 8
FS_LABEL = 7
FS_TICK = 6
FS_LEGEND = 6
FS_ANNOT = 6.5
FS_SMALL = 5.5

__all__ = [
    "C_RED", "C_ORANGE", "C_BLUE", "C_GRAY", "C_DEEP", "C_BG",
    "TOKEN_COLORS", "TOKEN_LABELS", "PAPER_RCPARAMS",
    "FIG_FULL", "FIG_HALF",
    "FS_TITLE", "FS_LABEL", "FS_TICK", "FS_LEGEND", "FS_ANNOT", "FS_SMALL",
    "apply_paper_style", "clean_ax", "layer_ticks",
    "plot_sink_vs_rand", "save_fig", "check_keys",
]

# ── rcParams preset (NeurIPS-compact) ────────────────────────────────
PAPER_RCPARAMS: dict = {
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "axes.titlepad": 3,
    "axes.linewidth": 0.5,
    "figure.facecolor": C_BG,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.08,
    "grid.linewidth": 0.3,
    "grid.color": C_GRAY,
    "xtick.major.size": 2,
    "xtick.major.width": 0.4,
    "xtick.major.pad": 1.5,
    "xtick.labelsize": 5.5,
    "ytick.major.size": 2,
    "ytick.major.width": 0.4,
    "ytick.major.pad": 1.5,
    "ytick.labelsize": 5.5,
    "lines.linewidth": 0.9,
    "lines.markersize": 1.8,
    "legend.fontsize": 6,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "none",
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.3,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.15,
    "legend.columnspacing": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_paper_style() -> None:
    """Apply ``PAPER_RCPARAMS`` to the current matplotlib session."""
    plt.rcParams.update(PAPER_RCPARAMS)


# ── Axis / figure helpers ─────────────────────────────────────────────

def clean_ax(ax: plt.Axes) -> None:
    """Hide top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def layer_ticks(layer_list, every: int = 8):
    """Compute evenly-spaced tick positions and labels for layer axes."""
    n = len(layer_list)
    step = max(1, n // every)
    pos = list(range(0, n, step))
    lbl = [str(layer_list[i]) for i in pos]
    return pos, lbl


def plot_sink_vs_rand(ax, x, sink, rand, ylabel, title, legend_loc="upper left"):
    """Plot Sink vs Random comparison line chart with standard styling."""
    ax.plot(x, sink, "o-", color=C_RED, lw=0.9, ms=1.8, label="Sink")
    ax.plot(x, rand, "o-", color=C_BLUE, lw=0.9, ms=1.8, label="Random")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=True, loc=legend_loc)
    clean_ax(ax)


def save_fig(fig, out_dir: str, filename: str,
             tight: bool = True) -> None:
    """Save figure as PDF/JPG, close it, and print confirmation.

    Args:
        tight: If True (default), call tight_layout() before saving.
               Set to False for figures with manual gridspec layout.
    """
    out_path = os.path.join(out_dir, filename)
    if tight:
        try:
            fig.tight_layout()
        except (ValueError, RuntimeError):
            pass
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def check_keys(data: dict, required, func_name: str) -> bool:
    """Return True if all *required* keys present, else print skip message."""
    missing = [k for k in required if k not in data]
    if missing:
        print(f"  Skipping {func_name}: missing keys {missing}")
        return False
    return True
