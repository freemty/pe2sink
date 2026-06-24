# PE2Sink

Public demos and compact statistics for studying how positional embeddings
shape attention sinks in diffusion transformers.

This repository is a clean public export of the demo-facing part of the
research code. It is intentionally small: it does not include private
experiment history, model weights, paper source, or full raw outputs.

## What Is Included

- A prompt-level comma-padding demo for FLUX.1-dev.
- A compact paper-statistics pack for reproducing selected figures without GPU.
- Minimal plotting utilities used by the public stats reproduction path.
- Unit and smoke tests for the public artifact.

## Main Claims Demonstrated Here

1. Some diffusion transformers allocate unusually high key importance to
   low-information tokens such as EOS or padding tokens.
2. Replacing padding-like text with repeated punctuation can redistribute
   attention budget at inference time.
3. The included statistics are enough to reproduce selected figure-level
   evidence without shipping full experiment outputs.

## Install

Create a Python environment, then install the lightweight dependencies:

```bash
git clone https://github.com/freemty/pe2sink.git
cd pe2sink

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the optional FLUX generation demo, also install PyTorch and Diffusers for
your CUDA or CPU environment. Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install diffusers transformers accelerate sentencepiece protobuf
```

FLUX.1-dev is a gated model. You must obtain access and provide your own local
checkpoint path or Hugging Face credentials.

## Quick Smoke Test

Run the public smoke test:

```bash
python scripts/smoke_public.py
```

This runs:

- unit tests for comma-padding prompt construction;
- a no-GPU dry run of the prompt attack demo;
- reproduction of two selected PDF figures from the checked-in stats pack.

If you do not have `matplotlib` and `numpy` installed, run only the no-figure
part:

```bash
python scripts/smoke_public.py --skip-figures
```

## Demo 1: Prompt Attack / Comma Padding

Dry-run mode only writes prompts and metadata:

```bash
python examples/prompt_attack_flux.py \
  --dry-run \
  --prompt "a puppy looking at a glowing screen" \
  --repeat 200
```

GPU mode generates a baseline image and a comma-padding image with the same
seed:

```bash
export PE2SINK_HF_PATH=/path/to/huggingface/cache
CUDA_VISIBLE_DEVICES=0 python examples/prompt_attack_flux.py \
  --prompt "a puppy looking at a glowing screen" \
  --repeat 200 \
  --seed 42 \
  --height 1024 \
  --width 1024 \
  --steps 12
```

You can also pass an explicit checkpoint:

```bash
python examples/prompt_attack_flux.py \
  --model-path /path/to/black-forest-labs/FLUX.1-dev \
  --prompt "a puppy looking at a glowing screen"
```

Outputs are written under `outputs/public_demos/prompt_attack_flux/`:

- `baseline_seed*.png`
- `comma*_seed*.png`
- `comparison_seed*.png`
- `original_prompt.txt`
- `modified_prompt.txt`
- `metadata.json`

The script disables cuDNN by default on CUDA because some systems expose
incompatible cuDNN 9 runtime symbols. Pass `--enable-cudnn` if your local stack
is clean.

## Demo 2: Reproduce Public Stats Figures

The checked-in `demo_assets/paper_stats/` pack is a compact subset of paper
statistics. It is intended for public verification and figure smoke tests, not
as a full experiment dump.

List available figures:

```bash
python tools/reproduce_public_stats.py --list
```

Generate all public figures:

```bash
python tools/reproduce_public_stats.py \
  --manifest demo_assets/paper_stats/manifest.json \
  --out-dir outputs/public_demos/paper_stats
```

Generate a smaller subset:

```bash
python tools/reproduce_public_stats.py \
  --only token_attention_bar same_norm
```

## Repository Layout

```text
demo_assets/paper_stats/    Compact JSON/JSON.GZ stats for public figures
examples/                   Runnable prompt attack demo
src/demo/                   Prompt construction helpers
src/utils/                  Minimal plotting style helpers
tools/                      Figure reproduction entry points
scripts/                    Public smoke test
tests/                      Unit tests for public helpers
configs/                    Small public demo config notes
docs/                       Reproduction and export notes
```

## Scope And Limitations

This repository does not include model weights, full benchmark outputs, full
paper source, private experiment logs, or private agent configuration. Full
paper-scale artifacts should be distributed separately as release assets or
datasets.

The prompt demo is an inference-time intervention. It is provided for
reproducibility and inspection, not as a general recommendation for production
image generation systems.

## Citation

```bibtex
@misc{yang2026pe2sink,
  title = {From Positional Embeddings to Attention Sinks in Diffusion Transformers},
  author = {Yang, Yuanbo and Liao, Yiyi and Gao, Jun},
  year = {2026},
  note = {Public demo artifact}
}
```

## License

Code in this repository is released under the MIT License. Model weights,
datasets, and third-party libraries are governed by their own licenses.
