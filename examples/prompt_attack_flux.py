#!/usr/bin/env python3
"""Generate a baseline vs comma-padding FLUX demo.

Dry-run mode has no GPU or diffusers dependency:

    python examples/prompt_attack_flux.py --dry-run --prompt "a puppy on a screen"

GPU mode loads FLUX.1-dev and writes baseline/comma images plus metadata:

    CUDA_VISIBLE_DEVICES=0 python examples/prompt_attack_flux.py \
      --prompt "a puppy on a screen" --repeat 200 --seed 42
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.demo.comma_padding import make_prompt_variant, slugify_prompt, write_prompt_metadata


FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Original text prompt")
    parser.add_argument("--repeat", type=int, default=200, help="Number of punctuation tokens to append")
    parser.add_argument("--padding-token", default=",", help="Repeated token used for comma padding")
    parser.add_argument("--no-separator", action="store_true", help="Append punctuation directly after the prompt")
    parser.add_argument("--mode", choices=["baseline", "comma", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true", help="Only print and save prompt metadata")
    parser.add_argument("--out-dir", default="outputs/public_demos/prompt_attack_flux")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local FLUX.1-dev path. If omitted, PE2SINK_HF_PATH or the HF model id is used.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-dit-embedding-length", type=int, default=512)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--enable-cudnn",
        action="store_true",
        help="Enable cuDNN. Disabled by default to avoid known cuDNN 9 runtime symbol mismatches.",
    )
    return parser.parse_args()


def resolve_flux_model_path(explicit: str | None = None) -> str:
    """Resolve a FLUX.1-dev checkpoint path or fall back to the HF model id."""
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.exists() else explicit

    candidates: list[Path] = []
    cache_root = os.environ.get("PE2SINK_HF_PATH")
    if cache_root:
        candidates.append(Path(cache_root) / FLUX_MODEL_ID)

    data_root = os.environ.get("PE2SINK_DATA_ROOT")
    if data_root:
        candidates.extend([
            Path(data_root) / "huggingface" / FLUX_MODEL_ID,
            Path(data_root) / "modelscope" / FLUX_MODEL_ID,
        ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return FLUX_MODEL_ID


def _torch_dtype(torch_module, dtype_name: str):
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[dtype_name]


def _call_flux(pipe, prompt: str, generator, args: argparse.Namespace):
    kwargs = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "max_sequence_length": args.max_sequence_length,
        "max_dit_embedding_length": args.max_dit_embedding_length,
    }
    call_params = inspect.signature(pipe.__call__).parameters
    filtered = {key: value for key, value in kwargs.items() if key in call_params}
    return pipe(**filtered).images[0]


def _save_comparison(left_path: Path, right_path: Path, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    height = max(left.height, right.height)
    width = left.width + right.width
    canvas = Image.new("RGB", (width, height + 32), "white")
    canvas.paste(left, (0, 32))
    canvas.paste(right, (left.width, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "baseline", fill=(0, 0, 0))
    draw.text((left.width + 12, 8), "comma padding", fill=(0, 0, 0))
    canvas.save(out_path)


def generate_images(args: argparse.Namespace, out_dir: Path, variant) -> dict[str, str]:
    import torch
    from diffusers.pipelines import FluxPipeline

    if str(args.device).startswith("cuda") and not args.enable_cudnn:
        torch.backends.cudnn.enabled = False

    model_path = resolve_flux_model_path(args.model_path)
    dtype = _torch_dtype(torch, args.dtype)
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype).to(args.device)

    prompts = []
    if args.mode in ("baseline", "both"):
        prompts.append(("baseline", variant.original_prompt))
    if args.mode in ("comma", "both"):
        prompts.append((f"comma{args.repeat}", variant.modified_prompt))

    image_paths: dict[str, str] = {}
    generator_device = args.device if str(args.device).startswith("cuda") else "cpu"
    for label, prompt in prompts:
        generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        image = _call_flux(pipe, prompt, generator, args)
        path = out_dir / f"{label}_seed{args.seed}.png"
        image.save(path)
        image_paths[label] = str(path)
        print(f"saved {label}: {path}")

    if "baseline" in image_paths and f"comma{args.repeat}" in image_paths:
        comparison = out_dir / f"comparison_seed{args.seed}.png"
        _save_comparison(Path(image_paths["baseline"]), Path(image_paths[f"comma{args.repeat}"]), comparison)
        image_paths["comparison"] = str(comparison)
        print(f"saved comparison: {comparison}")

    return image_paths


def main() -> None:
    args = parse_args()
    separator = "" if args.no_separator else " "
    variant = make_prompt_variant(args.prompt, args.repeat, args.padding_token, separator)

    out_dir = Path(args.out_dir) / f"{slugify_prompt(args.prompt)}_comma{args.repeat}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_extra = {
        "mode": args.mode,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "dry_run": args.dry_run,
    }

    image_paths: dict[str, str] = {}
    if args.dry_run:
        print(json.dumps({
            "original_prompt": variant.original_prompt,
            "modified_prompt": variant.modified_prompt,
            "repeat": variant.repeat,
            "output_dir": str(out_dir),
        }, indent=2, ensure_ascii=False))
    else:
        image_paths = generate_images(args, out_dir, variant)

    (out_dir / "original_prompt.txt").write_text(variant.original_prompt + "\n", encoding="utf-8")
    (out_dir / "modified_prompt.txt").write_text(variant.modified_prompt + "\n", encoding="utf-8")
    write_prompt_metadata(out_dir / "metadata.json", variant, image_paths=image_paths, **metadata_extra)


if __name__ == "__main__":
    main()
