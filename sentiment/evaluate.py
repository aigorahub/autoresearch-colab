"""
Evaluation utilities for sentiment classification.

Runs the fine-tuned model on the validation set and computes
classification metrics (accuracy, macro F1, per-class F1).
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_model(
    model,
    tokenizer,
    val_data: List[Dict],
    labels: List[str],
    max_new_tokens: int = 16,
    batch_size: int = 1,
    category_col: bool = False,
) -> Dict:
    """
    Run inference on validation set and compute metrics.

    Args:
        model: The fine-tuned model (already in inference mode).
        tokenizer: The tokenizer.
        val_data: List of {"conversations": [...]} dicts (same format as training).
        labels: List of valid label strings.
        max_new_tokens: Max tokens to generate for each prediction.
        batch_size: Not used yet (sequential inference for reliability).
        category_col: Whether data includes category info.

    Returns:
        Dict with: accuracy, macro_f1, weighted_f1, per_class (dict),
                   predictions (list), y_true (list), y_pred (list),
                   n_invalid (int), eval_time (float)
    """
    t0 = time.time()

    y_true = []
    y_pred = []
    predictions = []  # (text_snippet, true_label, predicted_label, raw_output)
    n_invalid = 0
    label_set = set(lab.lower().strip() for lab in labels)

    for i, sample in enumerate(val_data):
        convs = sample["conversations"]

        # True label is the last assistant message
        true_label = convs[-1]["content"].strip()

        # Build input: everything except the assistant response
        messages = [msg for msg in convs if msg["role"] != "assistant"]

        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for deterministic eval
                temperature=1.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Extract the predicted label
        pred_label = raw_output.split("\n")[0].strip()

        # Normalize: check if prediction matches any known label
        pred_lower = pred_label.lower().strip()
        matched = False
        for lab in labels:
            if pred_lower == lab.lower().strip():
                pred_label = lab
                matched = True
                break

        if not matched:
            # Try partial match (model might output extra text)
            for lab in labels:
                if lab.lower() in pred_lower:
                    pred_label = lab
                    matched = True
                    break

        if not matched:
            n_invalid += 1
            pred_label = raw_output[:50]  # Keep raw for debugging

        y_true.append(true_label)
        y_pred.append(pred_label)

        text_snippet = convs[1]["content"][:80] if len(convs) > 1 else ""
        predictions.append((text_snippet, true_label, pred_label, raw_output))

        # Progress
        if (i + 1) % 25 == 0:
            print(f"  Evaluated {i+1}/{len(val_data)}...")

    eval_time = time.time() - t0

    # Compute metrics (only on valid predictions)
    valid_mask = [yp in [l.strip() for l in labels] for yp in y_pred]
    y_true_valid = [yt for yt, vm in zip(y_true, valid_mask) if vm]
    y_pred_valid = [yp for yp, vm in zip(y_pred, valid_mask) if vm]

    if len(y_true_valid) > 0:
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        macro_f1 = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
        report = classification_report(
            y_true_valid, y_pred_valid,
            output_dict=True, zero_division=0,
        )
    else:
        accuracy = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0
        report = {}

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": report,
        "predictions": predictions,
        "y_true": y_true,
        "y_pred": y_pred,
        "n_total": len(val_data),
        "n_valid": len(y_true_valid),
        "n_invalid": n_invalid,
        "eval_time": eval_time,
    }

    return results


def print_eval_summary(results: Dict):
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"Macro F1:    {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"Valid/Total: {results['n_valid']}/{results['n_total']}")
    if results['n_invalid'] > 0:
        print(f"Invalid predictions: {results['n_invalid']}")
    print(f"Eval time:   {results['eval_time']:.1f}s")

    # Per-class breakdown
    per_class = results.get("per_class", {})
    if per_class:
        print("\nPer-class F1:")
        for label, metrics in per_class.items():
            if isinstance(metrics, dict) and "f1-score" in metrics:
                if label not in ("accuracy", "macro avg", "weighted avg"):
                    print(f"  {label:20s}: F1={metrics['f1-score']:.4f}  "
                          f"(P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                          f"n={metrics['support']:.0f})")
    print("=" * 50)
