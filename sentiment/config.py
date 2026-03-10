"""
Autoresearch configuration for sentiment fine-tuning.

This defines the hyperparameter search space that the autoresearch loop
explores. Each experiment samples a config from this space, trains a
LoRA adapter, evaluates on the validation set, and keeps/discards.
"""

import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ── Search space boundaries ────────────────────────────────────────────

SEARCH_SPACE = {
    # LoRA architecture
    "lora_r":            [4, 8, 16, 32, 64],
    "lora_alpha":        [8, 16, 32, 64, 128],
    "lora_dropout":      [0.0, 0.0, 0.05, 0.1],  # weighted toward 0

    # Training
    "learning_rate":     [5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3],
    "num_epochs":        [1, 2, 3, 4, 5],
    "per_device_batch":  [2, 4, 8, 16],
    "grad_accum_steps":  [1, 2, 4, 8],
    "warmup_ratio":      [0.0, 0.03, 0.05, 0.1],
    "weight_decay":      [0.0, 0.001, 0.01, 0.05, 0.1],
    "lr_scheduler":      ["linear", "cosine", "constant_with_warmup"],
    "optim":             ["adamw_8bit", "adamw_torch", "paged_adamw_8bit"],
    "max_seq_length":    [256, 512, 768, 1024],

    # Model choice
    "base_model": [
        "unsloth/Qwen3.5-0.6B-Instruct",
        "unsloth/Qwen3.5-1.7B-Instruct",
    ],
}


@dataclass
class ExperimentConfig:
    """A single experiment's hyperparameters."""
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch: int = 4
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    optim: str = "adamw_8bit"
    max_seq_length: int = 512

    # Model
    base_model: str = "unsloth/Qwen3.5-0.6B-Instruct"

    # Fixed (not searched)
    seed: int = 42
    logging_steps: int = 5

    def to_dict(self):
        return asdict(self)

    def describe(self) -> str:
        """Short human-readable description for logging."""
        parts = []
        for k, v in self.to_dict().items():
            if k in ("seed", "logging_steps"):
                continue
            parts.append(f"{k}={v}")
        return ", ".join(parts)


def sample_config(
    baseline: Optional[ExperimentConfig] = None,
    n_changes: int = 2,
) -> ExperimentConfig:
    """
    Sample a new experiment config.

    If a baseline is provided, mutate n_changes random hyperparameters
    from it (local search). Otherwise, sample everything randomly
    (global search).
    """
    if baseline is None:
        # Global random sample
        return ExperimentConfig(
            lora_r=random.choice(SEARCH_SPACE["lora_r"]),
            lora_alpha=random.choice(SEARCH_SPACE["lora_alpha"]),
            lora_dropout=random.choice(SEARCH_SPACE["lora_dropout"]),
            learning_rate=random.choice(SEARCH_SPACE["learning_rate"]),
            num_epochs=random.choice(SEARCH_SPACE["num_epochs"]),
            per_device_batch=random.choice(SEARCH_SPACE["per_device_batch"]),
            grad_accum_steps=random.choice(SEARCH_SPACE["grad_accum_steps"]),
            warmup_ratio=random.choice(SEARCH_SPACE["warmup_ratio"]),
            weight_decay=random.choice(SEARCH_SPACE["weight_decay"]),
            lr_scheduler=random.choice(SEARCH_SPACE["lr_scheduler"]),
            optim=random.choice(SEARCH_SPACE["optim"]),
            max_seq_length=random.choice(SEARCH_SPACE["max_seq_length"]),
            base_model=random.choice(SEARCH_SPACE["base_model"]),
        )

    # Local mutation: start from baseline, change n_changes params
    cfg = ExperimentConfig(**baseline.to_dict())
    mutable_keys = [k for k in SEARCH_SPACE.keys()]
    keys_to_change = random.sample(mutable_keys, min(n_changes, len(mutable_keys)))

    for key in keys_to_change:
        current = getattr(cfg, key)
        candidates = [v for v in SEARCH_SPACE[key] if v != current]
        if candidates:
            setattr(cfg, key, random.choice(candidates))

    return cfg


# ── Default baseline config ────────────────────────────────────────────

DEFAULT_CONFIG = ExperimentConfig()
