#!/usr/bin/env python3
"""Smoke-test the public demo entry points."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/public_demo_smoke")
    parser.add_argument("--skip-figures", action="store_true", help="Skip matplotlib figure smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_public_*.py"])

    prompt_out = out_dir / "prompt"
    run([
        sys.executable,
        "examples/prompt_attack_flux.py",
        "--dry-run",
        "--prompt",
        "a puppy looking at a glowing screen",
        "--repeat",
        "5",
        "--out-dir",
        str(prompt_out),
    ])

    metadata_files = list(prompt_out.glob("*/metadata.json"))
    if not metadata_files:
        raise RuntimeError(f"prompt dry-run did not write metadata under {prompt_out}")

    if not args.skip_figures:
        figure_out = out_dir / "figures"
        run([
            sys.executable,
            "tools/reproduce_public_stats.py",
            "--manifest",
            "demo_assets/paper_stats/manifest.json",
            "--out-dir",
            str(figure_out),
            "--only",
            "token_attention_bar",
            "same_norm",
        ])
        expected = [figure_out / "fig_token_selection_bars.pdf", figure_out / "fig_same_norm_diff_dist.pdf"]
        missing = [path for path in expected if not path.exists()]
        if missing:
            raise RuntimeError(f"missing expected figure output(s): {missing}")

    print("public demo smoke passed")


if __name__ == "__main__":
    main()
