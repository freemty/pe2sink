# Reproduction Notes

This public export supports two reproduction levels.

## No-GPU Figure Smoke

The compact stats pack under `demo_assets/paper_stats/` can regenerate selected
paper figures:

```bash
python tools/reproduce_public_stats.py \
  --manifest demo_assets/paper_stats/manifest.json \
  --out-dir outputs/public_demos/paper_stats
```

The pack is deliberately small. It is meant to validate figure code paths and
selected reported patterns, not to replace the full raw experiment archive.

## FLUX Prompt Demo

The prompt demo needs FLUX.1-dev weights:

```bash
python examples/prompt_attack_flux.py \
  --model-path /path/to/black-forest-labs/FLUX.1-dev \
  --prompt "a puppy looking at a glowing screen" \
  --repeat 200 \
  --seed 42
```

Use `--dry-run` to inspect the modified prompt without loading a model.

## Recommended Public Verification

```bash
python scripts/smoke_public.py
python -m py_compile \
  src/demo/comma_padding.py \
  examples/prompt_attack_flux.py \
  tools/reproduce_public_stats.py \
  scripts/smoke_public.py \
  tests/test_public_comma_padding.py
```
