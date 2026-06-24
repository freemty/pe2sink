# PE2Sink: From Positional Embeddings to Attention Sinks

[[Code](https://github.com/freemty/pe2sink)] [[Reproduction](docs/REPRODUCTION.md)] [[Prompt Demo](examples/prompt_attack_flux.py)] [[Stats Pack](demo_assets/paper_stats/manifest.json)]

PE2Sink is a public demo artifact for studying how positional embeddings shape
attention sinks in diffusion transformers, with a compact statistics pack and a
prompt-level comma-padding demo for FLUX.1-dev.

This repository is intentionally small. It does not include private experiment
history, model weights, paper source, or full raw outputs.

## News

- **[2026-06]** Clean public repository released at [freemty/pe2sink](https://github.com/freemty/pe2sink).
- **[2026-06]** Public stats pack released for no-GPU figure reproduction.
- **[2026-06]** Prompt attack / comma-padding demo released for FLUX.1-dev.

## Installation

### For Public Figure Reproduction

```bash
# Clone the repository
git clone https://github.com/freemty/pe2sink.git
cd pe2sink

# Create an environment
python -m venv .venv
source .venv/bin/activate

# Install lightweight plotting dependencies
pip install -r requirements.txt
```

### For FLUX Prompt Demo

Install PyTorch and Diffusers for your CUDA or CPU environment:

```bash
# Example for CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Diffusers stack
pip install diffusers transformers accelerate sentencepiece protobuf
```

FLUX.1-dev is a gated model. You must obtain access and provide a local
checkpoint path or Hugging Face credentials.

```bash
# Optional local checkpoint root.
# The demo checks PE2SINK_HF_PATH/black-forest-labs/FLUX.1-dev first.
export PE2SINK_HF_PATH=/path/to/huggingface/cache
```

## Usage

### Quick Smoke Test

```bash
python scripts/smoke_public.py
```

This runs unit tests, a no-GPU prompt dry run, and a two-figure stats
reproduction smoke test.

If you only want the standard-library path:

```bash
python scripts/smoke_public.py --skip-figures
```

### Public Stats Reproduction

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

The checked-in `demo_assets/paper_stats/` pack is a compact subset of paper
statistics. It is intended for public verification and figure smoke tests, not
as a full experiment dump.

### Prompt Attack / Comma Padding

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

## Repository Structure

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

## TODO

- Release the paper link when it becomes publicly available.
- Add a compact visual teaser to the README.
- Publish larger paper-scale artifacts separately as release assets or datasets.

## Citation

```bibtex
@misc{yang2026pe2sink,
  title = {From Positional Embeddings to Attention Sinks in Diffusion Transformers},
  author = {Yang, Yuanbo and Liao, Yiyi and Gao, Jun},
  year = {2026},
  note = {Public demo artifact},
  url = {https://github.com/freemty/pe2sink}
}
```

## License

Code in this repository is released under the MIT License. Model weights,
datasets, and third-party libraries are governed by their own licenses.
