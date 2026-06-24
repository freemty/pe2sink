#!/usr/bin/env python3
"""Reproduce public demo figures from a small checked-in stats manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.plot_paper_figures import (
    plot_fig_band_norm,
    plot_fig_norm_paradox_panel,
    plot_fig_same_norm_diff_dist,
)


FIGURE_ORDER = ["token_attention_bar", "same_norm", "norm_paradox_panel", "band_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="demo_assets/paper_stats/manifest.json")
    parser.add_argument("--out-dir", default="outputs/public_demos/paper_stats")
    parser.add_argument("--only", nargs="*", default=None, help=f"Subset of: {', '.join(FIGURE_ORDER)}")
    parser.add_argument("--list", action="store_true", help="List available figures and exit")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    base = path.parent
    for name, spec in manifest.get("figures", {}).items():
        for key in ("stats", "extra_stats"):
            value = spec.get(key)
            if isinstance(value, str) and not (base / value).exists():
                raise FileNotFoundError(f"{name}: missing {key}: {base / value}")
    return manifest


def read_json(path: Path) -> dict:
    """Read JSON or JSON.GZ stats files."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    base = manifest_path.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available = [name for name in FIGURE_ORDER if name in manifest["figures"]]
    if args.list:
        print("\n".join(available))
        return

    selected = args.only or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown figure(s): {', '.join(unknown)}")

    generated: list[str] = []
    for name in selected:
        spec = manifest["figures"][name]
        stats_path = base / spec["stats"]
        print(f"=== {name}: {stats_path} ===")

        if name == "token_attention_bar":
            extra_stats = spec.get("extra_stats")
            if not extra_stats:
                raise ValueError("token_attention_bar requires extra_stats=flux_keyimportance_stats.json")
            subprocess.run(
                [
                    sys.executable,
                    "tools/plot_token_selection_figure.py",
                    "--json",
                    str(base / extra_stats),
                    "--comma_json",
                    str(stats_path),
                    "--out_dir",
                    str(out_dir),
                ],
                check=True,
            )
            generated.append("fig_token_selection_bars.pdf")
        elif name == "same_norm":
            data = read_json(stats_path)
            plot_fig_same_norm_diff_dist(data, str(out_dir))
            generated.append("fig_same_norm_diff_dist.pdf")
        elif name == "norm_paradox_panel":
            data = read_json(stats_path)
            plot_fig_norm_paradox_panel(data, str(out_dir))
            generated.append("fig_norm_paradox_panel.pdf")
        elif name == "band_norm":
            data = read_json(stats_path)
            plot_fig_band_norm(str(out_dir), sink_curves_data=data)
            generated.append("fig_band_norm.pdf")

    missing = [filename for filename in generated if not (out_dir / filename).exists()]
    if missing:
        raise RuntimeError(f"Expected output(s) not found: {', '.join(missing)}")

    print("\nGenerated:")
    for filename in generated:
        print(f"  {out_dir / filename}")


if __name__ == "__main__":
    main()
