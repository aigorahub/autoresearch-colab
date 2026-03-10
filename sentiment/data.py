"""
Data loading and formatting for sentiment fine-tuning.

Supports CSV, JSON, and JSONL files with configurable column names.
Converts raw text+label data into the chat template format expected
by Qwen3.5-Instruct for classification fine-tuning.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ── Data Loading ───────────────────────────────────────────────────────

def load_dataset(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
    category_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a labeled dataset from CSV, JSON, or JSONL.

    Args:
        path: Path to the data file.
        text_col: Column name for the input text.
        label_col: Column name for the sentiment label.
        category_col: Optional column for product category (e.g., "coffee", "chocolate").

    Returns:
        DataFrame with columns: text, label, and optionally category.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif ext in (".tsv",):
        df = pd.read_csv(path, sep="\t")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use CSV, JSON, JSONL, TSV, or XLSX.")

    # Validate required columns exist
    if text_col not in df.columns:
        raise ValueError(
            f"Text column '{text_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Standardize column names
    rename = {text_col: "text", label_col: "label"}
    if category_col and category_col in df.columns:
        rename[category_col] = "category"

    df = df.rename(columns=rename)

    # Clean
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    print(f"Loaded {len(df)} samples from {path}")
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    if "category" in df.columns:
        print(f"Categories: {df['category'].value_counts().to_dict()}")

    return df


def train_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/val sets, optionally stratified by label."""
    if stratify:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, test_size=val_fraction, random_state=seed,
            stratify=df["label"],
        )
    else:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        split_idx = int(len(df) * (1 - val_fraction))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ── Chat Template Formatting ──────────────────────────────────────────

SYSTEM_PROMPT_DEFAULT = (
    "You are a sentiment classifier for consumer product reviews. "
    "Given a text response about a product, classify the sentiment. "
    "Respond with ONLY the sentiment label, nothing else."
)

SYSTEM_PROMPT_WITH_CATEGORY = (
    "You are a sentiment classifier for consumer product reviews. "
    "Given a text response about {category}, classify the sentiment. "
    "Note that sentiment is context-dependent — for example, 'bitter' might be "
    "positive for coffee but negative for chocolate. "
    "Respond with ONLY the sentiment label, nothing else."
)


def format_for_training(
    df: pd.DataFrame,
    labels: List[str],
    include_category: bool = False,
) -> List[Dict]:
    """
    Convert DataFrame rows into chat-template conversations for SFT.

    Each sample becomes:
        system: <classifier instructions with available labels>
        user: <the text to classify>
        assistant: <the label>
    """
    label_str = ", ".join(labels)
    conversations = []

    for _, row in df.iterrows():
        if include_category and "category" in row and pd.notna(row.get("category")):
            system = SYSTEM_PROMPT_WITH_CATEGORY.format(category=row["category"])
        else:
            system = SYSTEM_PROMPT_DEFAULT

        system += f"\n\nAvailable labels: {label_str}"

        conv = {
            "conversations": [
                {"role": "system", "content": system},
                {"role": "user", "content": str(row["text"])},
                {"role": "assistant", "content": str(row["label"])},
            ]
        }
        conversations.append(conv)

    return conversations


def format_for_inference(
    text: str,
    labels: List[str],
    category: Optional[str] = None,
) -> List[Dict]:
    """
    Format a single text for inference (no assistant response).
    Returns a messages list for tokenizer.apply_chat_template().
    """
    label_str = ", ".join(labels)

    if category:
        system = SYSTEM_PROMPT_WITH_CATEGORY.format(category=category)
    else:
        system = SYSTEM_PROMPT_DEFAULT

    system += f"\n\nAvailable labels: {label_str}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
