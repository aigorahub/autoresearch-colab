# autoresearch-colab

A Colab-ready fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous AI-driven LLM training experiments on a single GPU.

## What is this?

Give an AI agent a small but real LLM training setup, let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

This repo packages everything to run on **Google Colab** (or any single-GPU machine).

## Quick Start (Colab)

1. Open `autoresearch_colab.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → GPU)
3. Run the cells in order — the notebook handles environment setup, data download, and the experiment loop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aigorahub/autoresearch-colab/blob/main/autoresearch_colab.ipynb)

## Quick Start (Local)

```bash
# Install uv project manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (~2 min)
uv run prepare.py

# Run a single training experiment (~5 min)
uv run train.py
```

## How it works

Three files that matter:

| File | Role | Who edits it? |
|------|------|---------------|
| `prepare.py` | Data download, tokenizer training, evaluation metric | Nobody (fixed) |
| `train.py` | GPT model, optimizer, training loop (~630 lines) | The AI agent |
| `program.md` | Instructions for the agent | You (the human) |

Each experiment trains for a **fixed 5-minute time budget**. The metric is **val_bpb** (validation bits per byte) — lower is better.

## Two ways to run

### 1. Self-contained loop (in the notebook)
The Colab notebook includes a built-in hyperparameter search loop that randomly proposes changes and keeps improvements. No external tools needed.

### 2. LLM agent (recommended)
Point a coding agent at `program.md`:
- **Claude Code**: `claude` → "Hi have a look at program.md and let's kick off a new experiment!"
- **Cursor / Windsurf**: Open folder, reference program.md
- **OpenAI Codex**: Point at this repo

The agent will read the code, reason about what to change, and run experiments — much smarter than random search.

## GPU compatibility

Defaults are tuned for H100. For smaller GPUs, the notebook includes auto-patching. Key adjustments:

| GPU | VRAM | Recommended DEPTH | DEVICE_BATCH_SIZE |
|-----|------|--------------------|-------------------|
| T4 | 16GB | 4 | 32 |
| L4 | 24GB | 6 | 64 |
| A100 40GB | 40GB | 8 | 64 |
| A100 80GB / H100 | 80GB | 8 (default) | 128 (default) |

## Credits

All core code by [Andrej Karpathy](https://github.com/karpathy/autoresearch). This repo just adds the Colab wrapper and GPU auto-tuning.

## License

MIT (same as upstream)
