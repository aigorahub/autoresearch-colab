# autoresearch-colab

A Colab-ready fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous AI-driven LLM training experiments on a single GPU.

## What is this?

Give an AI agent a small but real LLM training setup, let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

This repo packages everything to run on **Google Colab** (or any single-GPU machine).

## Quick Start (Colab)

1. Open `autoresearch_colab.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → select your GPU)
3. Run the cells in order — the notebook handles environment setup, data download, and the experiment loop

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aigorahub/autoresearch-colab/blob/master/autoresearch_colab.ipynb)

## Choosing a Colab GPU

Karpathy's defaults are tuned for an H100 (80GB). Smaller GPUs need adjusted settings. Here's what to expect on each Colab option:

### Recommended: A100 (40GB) — best balance of cost and capability

- **VRAM:** 40GB — enough to run autoresearch with minor adjustments
- **Cost:** ~$1.18/hr (~10.6 compute units/hr)
- **What to change:** Reduce `DEVICE_BATCH_SIZE` from 128 → 64, and `TOTAL_BATCH_SIZE` from 2^19 → 2^18. Everything else can stay at defaults (DEPTH=8, WINDOW_PATTERN="SSSL").
- **Expected pace:** ~12 experiments/hour (each ~5 min), ~100 overnight
- **Flash Attention:** FA3 works on A100 via the `kernels-community/flash-attn3` fallback (already handled in `train.py`). FA3 on Ampere is comparable to FA2 performance — fine for autoresearch.
- **Availability:** Generally available on Colab Pro. Select "A100" in the runtime GPU dropdown.

### Premium: H100 (80GB) — full speed, Karpathy's defaults work as-is

- **VRAM:** 80GB — runs the original code with zero changes
- **Cost:** ~$1.86/hr (~18 compute units/hr)
- **What to change:** Nothing. Defaults work out of the box.
- **Flash Attention:** Full FA3 with Hopper-native optimizations (WGMMA, TMA). 1.5–2x faster than FA2.
- **Availability:** Limited on Colab Pro — often unavailable and falls back to A100. More reliably available on Pro+ ($49.99/mo), which gives priority access to premium GPUs.

### Budget: L4 (22.5GB)

- **VRAM:** 22.5GB — tight, needs aggressive tuning
- **Cost:** ~$0.48/hr (~3 compute units/hr)
- **What to change:** `DEPTH` 8→6, `DEVICE_BATCH_SIZE` 128→64, `TOTAL_BATCH_SIZE` 2^19→2^17, `WINDOW_PATTERN` "SSSL"→"SL". May still OOM on some configurations the agent tries.
- **Tradeoff:** Cheaper per hour, but each experiment trains a smaller model. Agent-proposed changes that increase model size will crash more often.

### Not recommended: T4 (15GB)

- **VRAM:** 15GB — too constrained for meaningful autoresearch
- **What to change:** `DEPTH` 8→4, `DEVICE_BATCH_SIZE` 128→32, `TOTAL_BATCH_SIZE` 2^19→2^16, `WINDOW_PATTERN` "SSSL"→"L". The model is so small that improvements are marginal and don't transfer to larger models.
- **Flash Attention:** The `kernels` package may not support Turing architecture (compute capability 7.5). You may need to swap FA3 for PyTorch's built-in `scaled_dot_product_attention`.

### New: RTX 6000 Pro / "G4" (48GB)

- **VRAM:** 48GB — more headroom than A100
- **Cost:** ~$0.87/hr (~8.x compute units/hr)
- **What to change:** Same adjustments as A100 (reduce batch sizes slightly), or potentially run closer to defaults given the extra 8GB of VRAM.
- **Note:** Recently added to Colab (Feb 2026). Availability may vary.

### Quick reference

| GPU | VRAM | Cost/hr | DEPTH | DEVICE_BATCH_SIZE | TOTAL_BATCH_SIZE | WINDOW_PATTERN | Changes needed? |
|-----|------|---------|-------|-------------------|------------------|----------------|-----------------|
| H100 | 80GB | ~$1.86 | 8 | 128 | 2^19 | "SSSL" | None |
| RTX 6000 | 48GB | ~$0.87 | 8 | 80 | 2^18 | "SSSL" | Minor |
| A100 | 40GB | ~$1.18 | 8 | 64 | 2^18 | "SSSL" | Minor |
| L4 | 22.5GB | ~$0.48 | 6 | 64 | 2^17 | "SL" | Moderate |
| T4 | 15GB | ~$0.18 | 4 | 32 | 2^16 | "L" | Major (not recommended) |

**Bottom line:** If you have Colab Pro, go with the **A100**. It's the sweet spot — enough VRAM to run real experiments at defaults close to Karpathy's, and widely available. If you can get an H100, even better — everything works out of the box.

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
The Colab notebook includes a built-in hyperparameter search loop that randomly proposes changes and keeps improvements. No external tools needed. This is simpler but only tweaks knobs — it won't make architectural changes or reason about the code.

### 2. LLM agent (recommended)
Point a coding agent at `program.md`:
- **Claude Code**: `claude` → "Hi have a look at program.md and let's kick off a new experiment!"
- **Cursor / Windsurf**: Open folder, reference program.md
- **OpenAI Codex**: Point at this repo

The agent will read the code, reason about what to change, and run experiments — much smarter than random search. This is how Karpathy got his 11% improvement.

## Sentiment Fine-Tuning (New)

The `sentiment/` directory contains a separate autoresearch application: fine-tuning **Qwen3.5** (0.6B or 1.7B) for **domain-specific sentiment classification** using Unsloth + LoRA.

This is useful for packaged goods, where sentiment is context-dependent ("bitter" = positive for coffee, negative for chocolate).

### Quick Start

Open `sentiment_autoresearch.ipynb` in Colab, upload your labeled dataset (CSV/JSON/JSONL with text + label columns), and run all cells.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aigorahub/autoresearch-colab/blob/master/sentiment_autoresearch.ipynb)

### How it works

1. Loads your labeled data, splits into train/val
2. Trains a LoRA adapter on Qwen3.5-0.6B-Instruct with default hyperparameters (baseline)
3. Runs an autoresearch loop: mutates LoRA rank, learning rate, epochs, etc., keeps improvements
4. Exports the best model to GGUF for CPU deployment via Ollama / llama.cpp

See [`sentiment/README.md`](sentiment/README.md) for full documentation.

| Model | GPU | VRAM | Time/Experiment |
|-------|-----|------|-----------------|
| Qwen3.5-0.6B | T4 | ~5 GB | ~1-2 min |
| Qwen3.5-0.6B | A100 | ~5 GB | ~30-60 sec |
| Qwen3.5-1.7B | A100 | ~8 GB | ~2-4 min |

## Credits

All core LLM training code by [Andrej Karpathy](https://github.com/karpathy/autoresearch). Sentiment fine-tuning uses [Unsloth](https://github.com/unslothai/unsloth) for fast LoRA training. This repo adds the Colab notebooks and GPU configuration guide.

## License

MIT (same as upstream)
