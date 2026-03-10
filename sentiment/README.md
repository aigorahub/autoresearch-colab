# Sentiment Autoresearch

Fine-tune a small language model (Qwen3.5-0.6B/1.7B) for domain-specific sentiment classification using the autoresearch pattern: automated hyperparameter search that keeps improvements and discards regressions.

## Quick Start

Open `sentiment_autoresearch.ipynb` in Google Colab, upload your data, and run all cells.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aigorahub/autoresearch-colab/blob/master/sentiment_autoresearch.ipynb)

## Data Format

Your dataset needs at minimum two columns:

| Column | Required | Description |
|--------|----------|-------------|
| `text` | Yes | The open-ended text response to classify |
| `label` | Yes | The sentiment label (e.g., "positive", "negative", "neutral") |
| `category` | No | Product category for context-aware classification (e.g., "coffee", "chocolate") |

Supported formats: **CSV**, **JSON**, **JSONL**, **TSV**, **XLSX**

See `sample_data.csv` for an example.

### Why category matters

Sentiment in packaged goods is context-dependent. "Bitter" is positive for coffee but negative for chocolate. When you include a `category` column, the model's system prompt adapts to mention the specific product, helping it learn these domain-specific patterns.

## How It Works

1. **Baseline**: Trains the default config and measures macro F1 on a held-out validation set
2. **Autoresearch loop**: Each iteration mutates 2 hyperparameters from the current best config, trains, evaluates, and keeps the result if F1 improved
3. **Final export**: Re-trains the best config on all data, exports to GGUF for CPU deployment

### What gets searched

- LoRA rank (r) and alpha
- Learning rate, scheduler, optimizer
- Batch size, gradient accumulation steps
- Number of epochs, warmup ratio, weight decay
- Max sequence length
- Base model (0.6B vs 1.7B)

## Files

| File | Description |
|------|-------------|
| `config.py` | Hyperparameter search space and experiment config dataclass |
| `data.py` | Data loading (CSV/JSON/JSONL/TSV/XLSX), train/val split, chat template formatting |
| `evaluate.py` | Model evaluation: accuracy, macro F1, weighted F1, per-class metrics |
| `sample_data.csv` | Example dataset showing the expected format |

## GPU Requirements

| Model | GPU | VRAM Needed | Time per Experiment |
|-------|-----|-------------|-------------------|
| Qwen3.5-0.6B | T4 (16GB) | ~5 GB | ~1-2 min |
| Qwen3.5-0.6B | A100 (40GB) | ~5 GB | ~30-60 sec |
| Qwen3.5-1.7B | A100 (40GB) | ~8 GB | ~2-4 min |

## CPU Deployment

After export, run the GGUF model locally:

```bash
# With Ollama
ollama create sentiment-model -f Modelfile
ollama run sentiment-model "This coffee has a wonderful bitter edge"

# With llama.cpp
./llama-cli -m sentiment_gguf/model-Q8_0.gguf -p "<your prompt>"
```
