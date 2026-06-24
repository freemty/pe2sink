"""Utilities for the public comma-padding prompt demo."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptVariant:
    """A baseline prompt and its comma-padding counterpart."""

    original_prompt: str
    modified_prompt: str
    repeat: int
    padding_token: str
    separator: str


def build_comma_prompt(
    prompt: str,
    repeat: int = 200,
    padding_token: str = ",",
    separator: str = " ",
) -> str:
    """Append repeated punctuation to a prompt.

    The default separator matches the benchmark scripts used for the paper
    application experiments.
    """
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    if not padding_token:
        raise ValueError("padding_token must be non-empty")
    if repeat == 0:
        return prompt

    padding = padding_token * repeat
    if not separator:
        return f"{prompt}{padding}"
    if prompt.endswith(separator):
        return f"{prompt}{padding}"
    return f"{prompt}{separator}{padding}"


def make_prompt_variant(
    prompt: str,
    repeat: int = 200,
    padding_token: str = ",",
    separator: str = " ",
) -> PromptVariant:
    """Create the baseline/modified prompt pair used by the demo."""
    return PromptVariant(
        original_prompt=prompt,
        modified_prompt=build_comma_prompt(prompt, repeat, padding_token, separator),
        repeat=repeat,
        padding_token=padding_token,
        separator=separator,
    )


def slugify_prompt(prompt: str, max_len: int = 80) -> str:
    """Create a stable, filesystem-safe prefix from a prompt."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", prompt.strip()).strip("_").lower()
    return (slug or "prompt")[:max_len]


def write_prompt_metadata(path: str | Path, variant: PromptVariant, **extra: object) -> None:
    """Write prompt metadata as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(variant)
    payload.update(extra)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
