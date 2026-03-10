# Engineering Guide: Sentiment Autoresearch

This document is a handoff for anyone picking up development on the sentiment fine-tuning side of this repo. It captures the design rationale, known gotchas, and ideas for improvement.

## Architecture Overview

```
autoresearch-colab/
├── sentiment_autoresearch.ipynb   # Main Colab notebook (run this)
├── sentiment/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameter search space + ExperimentConfig dataclass
│   ├── data.py                    # Data loading, train/val split, chat template formatting
│   ├── evaluate.py                # Inference + classification metrics (F1, accuracy)
│   ├── sample_data.csv            # Example dataset (20 rows, 3 categories)
│   └── README.md                  # User-facing docs
├── autoresearch_colab.ipynb       # Original Karpathy autoresearch notebook (LLM pretraining)
├── train.py                       # Original Karpathy training script
├── prepare.py                     # Original Karpathy data prep
└── program.md                     # Original Karpathy agent instructions
```

The `sentiment/` package is completely independent from the original Karpathy autoresearch files. They share the same repo for convenience but don't import from each other.

## Key Design Decisions

### Why Qwen3.5-0.6B-Instruct (not 0.8B)

The Qwen3.5 lineup changed naming between announcement and release. On Hugging Face/Unsloth, the actual model IDs are:
- `unsloth/Qwen3.5-0.6B-Instruct` (what was initially called 0.8B in some announcements)
- `unsloth/Qwen3.5-1.7B-Instruct` (what was initially called 2B)

The config defaults to the 0.6B model. The 1.7B is in the search space so the autoresearch loop can try it, but it's 3x larger and may not be worth it for simple classification tasks.

**Verify model IDs before running.** If Unsloth updates their model names, the notebook will fail at the `FastLanguageModel.from_pretrained()` call. Check https://huggingface.co/unsloth for current IDs.

### Why bf16 LoRA, not QLoRA

Unsloth's docs explicitly recommend bf16 LoRA over QLoRA for Qwen3.5 small models — the VRAM savings from 4-bit quantization are minimal (0.6B already fits in 3-5GB), and QLoRA adds training instability. The notebook sets `load_in_4bit=False`.

### Why chat template formatting (not raw text)

The model is an Instruct variant, so it expects `<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant` tokens. We format each sample as:

```
system: You are a sentiment classifier for consumer product reviews...
        Available labels: negative, neutral, positive
user:   <the text>
assistant: <the label>
```

This matters because at inference time, we need the model to respond with just the label. Using the chat template ensures the model's instruction-following capabilities are preserved, not destroyed.

### Why `packing=True` in SFTConfig

Sentiment samples are short (50-200 tokens typically). Without packing, most of the sequence length is wasted padding. Packing concatenates multiple samples into one sequence, dramatically improving training throughput. This is especially important for the autoresearch loop where we want each experiment to finish fast.

### Why the system prompt mentions specific product categories

The whole point of this for Aigora is domain-specific sentiment. "Bitter" is positive for coffee, negative for chocolate. The system prompt in `data.py` has two variants:

1. **Generic**: "You are a sentiment classifier for consumer product reviews."
2. **Category-aware**: "You are a sentiment classifier for consumer product reviews. Given a text response about {category}, classify the sentiment. Note that sentiment is context-dependent — for example, 'bitter' might be positive for coffee but negative for chocolate."

If the dataset has a `category` column, the category-aware prompt is used. This gives the model explicit permission to interpret the same word differently depending on context.

## Known Limitations & Things to Watch

### 1. Evaluation is sequential (slow for large val sets)

`evaluate.py` runs inference one sample at a time. For a val set of 500+ samples, this can take 5-10 minutes on an A100. Batched inference would be faster but is tricky with variable-length chat template outputs. This is the single biggest bottleneck in the autoresearch loop.

**Improvement idea:** Batch the tokenization, pad, run forward pass in one go, then decode. The tricky part is handling the variable-length system prompts (category-aware vs generic).

### 2. No early stopping in the autoresearch loop

Each experiment trains for the full `num_epochs`. If the model is already converged after 1 epoch, we waste time on the remaining epochs. The `SFTTrainer` supports early stopping via callbacks, but we don't use it because:
- It adds complexity to the config
- The overhead is small for tiny models + small datasets
- The autoresearch loop handles this implicitly — configs with too many epochs will underperform due to overfitting, and get discarded

If datasets get large (10K+ samples), early stopping becomes worth adding.

### 3. Label parsing at inference time is fragile

The model is supposed to output just the label (e.g., "positive"). But sometimes it outputs "Positive" or "positive." or "The sentiment is positive." The `evaluate.py` tries:
1. Exact match (case-insensitive)
2. Partial match (label substring in output)
3. Falls back to marking as invalid

This works ~95% of the time for well-trained models. For poorly-trained early experiments, the invalid rate can be high. The `n_invalid` count in results tracks this.

**Improvement idea:** Add a `constrained_generation` mode that forces the model to output one of the valid label tokens. This is possible with Hugging Face's `LogitsProcessor` — create a mask that only allows tokens that are prefixes of valid labels.

### 4. The search space may need tuning for your data

The defaults in `config.py` are reasonable starting points, but:
- If your dataset is very small (<200 samples): reduce `num_epochs` candidates, increase `lora_dropout` options
- If your dataset is large (>5K samples): add `num_epochs=[1]` and consider adding gradient accumulation steps > 8
- If you have many labels (>5 classes): the model may need higher `lora_r` and more epochs
- If texts are long (>256 tokens average): increase `max_seq_length` candidates

### 5. DeltaNet dependencies can be finicky

Qwen3.5 uses a hybrid DeltaNet + Attention architecture, which requires `flash-linear-attention` and `causal_conv1d`. These have specific version requirements:
- `causal_conv1d==1.6.0` (must match torch 2.8.0)
- `flash-linear-attention` (needs `--no-build-isolation` flag)

If the install cell fails, check that the torch version matches. Unsloth's install block handles this, but it's the most likely point of failure.

### 6. GGUF export requires llama.cpp compilation

The `save_pretrained_gguf()` call triggers an automatic download and compilation of llama.cpp inside the Colab instance. This:
- Takes 2-5 minutes the first time
- Requires ~2GB of disk space
- Can fail if the Colab instance is low on disk (happens after many experiments write checkpoints)

**Mitigation:** The notebook exports GGUF at the end (after the autoresearch loop), not during the loop. If disk is tight, delete the `experiments/` directory before exporting.

## Ideas for Future Development

### Smart search (replace random with Bayesian)

The current loop randomly mutates 2 hyperparams per iteration. This is the Karpathy approach (simple, robust, no dependencies). But for LoRA fine-tuning where experiments are fast, you could use:
- **Optuna** for Bayesian optimization (Tree-Parzen Estimator)
- **Population-based training** (multiple configs evolving in parallel)

The tradeoff is complexity vs. sample efficiency. With 30+ experiments, random search with local mutation actually works surprisingly well because the search space is small (13 dimensions) and many configs are "good enough."

### Multi-task fine-tuning

Instead of training separate models per product category, train one model on all categories simultaneously. The category-aware system prompt already handles this — the model learns "when the system prompt says coffee, bitter is good." This is actually the default behavior if you provide a dataset with multiple categories.

### Curriculum learning

Start training on easy/clear examples (strongly positive/negative), then gradually introduce ambiguous ones. This could improve F1 on the "neutral" class, which is typically the hardest to classify.

### Distillation from a larger model

Before fine-tuning, use GPT-4 or Claude to label a large unlabeled dataset, then use those labels as training data. This is especially useful if the human-labeled dataset is small (<500 samples). The autoresearch loop could then optimize the fine-tuned small model against the larger model's predictions.

### Deployment pipeline

The GGUF export is the first step. A full deployment pipeline would be:
1. Export best model to GGUF (done)
2. Push to Hugging Face Hub (one line: `model.push_to_hub_gguf(...)`)
3. Create an Ollama Modelfile with the system prompt baked in
4. Wrap in a simple FastAPI server for batch classification
5. Integrate with THEUS for real-time sentiment scoring

### Confidence calibration

The current evaluation only looks at the argmax prediction. For production use, you'd want calibrated confidence scores. One approach: run the model N times with `temperature > 0` and compute agreement as a confidence proxy.

## Running Locally (without Colab)

The notebook is designed for Colab but the `sentiment/` package works anywhere:

```bash
# Install Unsloth locally (see https://unsloth.ai/docs/get-started/install)
pip install unsloth

# Clone repo
git clone https://github.com/aigorahub/autoresearch-colab.git
cd autoresearch-colab

# Run in Python
from sentiment.config import DEFAULT_CONFIG, sample_config
from sentiment.data import load_dataset, train_val_split, format_for_training
# ... same code as the notebook cells
```

You need a CUDA GPU with at least 6GB VRAM for the 0.6B model.

## Contact

This repo was created for Aigora by Dr. John Ennis. The sentiment fine-tuning is designed for packaged goods / sensory science applications where domain-specific sentiment understanding matters.
