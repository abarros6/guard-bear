# Held-out test evaluation using the tuned threshold from threshold_config.json.
# Produces full metrics report, confusion matrix, ROC curve, and error analysis CSVs.
# See EVAL.md for full specification and deployment readiness checklist.
# Output: eval_roc_curve.png, eval_false_negatives.csv, eval_false_positives.csv

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FINAL_MODEL_DIR = Path("guard_model_final")
TEST_SPLIT_PATH = Path("data/guard_test_split.csv")
ROC_CURVE_PATH = Path("eval_roc_curve.png")
FALSE_NEG_PATH = Path("eval_false_negatives.csv")
FALSE_POS_PATH = Path("eval_false_positives.csv")

MAX_LENGTH = 256
BATCH_SIZE = 32

TARGETS = {
    "unsafe_recall": 0.97,
    "unsafe_precision": 0.85,
    "unsafe_f1": 0.90,
    "accuracy": 0.92,
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def run_inference(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    ds = TextDataset(enc, labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    return all_probs, all_labels


def benchmark_latency(model, tokenizer, device: torch.device, n: int = 100) -> float:
    sample = tokenizer(
        "Will my surgery hurt?",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    sample = {k: v.to(device) for k, v in sample.items()}
    with torch.no_grad():
        model(**sample)  # warm up
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model(**sample)
    return (time.perf_counter() - start) / n * 1000


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    device = get_device()
    logging.info("device=%s", device)

    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_MODEL_DIR)).to(device)

    with open(FINAL_MODEL_DIR / "threshold_config.json") as f:
        threshold_config = json.load(f)
    threshold = threshold_config["threshold"]
    logging.info("threshold=%.4f", threshold)

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    logging.info("test set rows=%d", len(test_df))

    probs, labels = run_inference(
        model, tokenizer,
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        device,
    )
    preds = (probs >= threshold).astype(int)

    unsafe_recall = recall_score(labels, preds, pos_label=1)
    unsafe_precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    unsafe_f1 = f1_score(labels, preds, pos_label=1)
    accuracy = float((preds == labels).mean())
    roc_auc = roc_auc_score(labels, probs)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print("=" * 60)
    print("GUARD MODEL — HELD-OUT TEST EVALUATION")
    print("=" * 60)
    print(f"\nThreshold: {threshold:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['safe', 'unsafe'])}")
    print("Confusion Matrix:")
    print(f"  True Negatives  (safe → safe):     {tn}")
    print(f"  False Positives (safe → unsafe):   {fp}")
    print(f"  False Negatives (unsafe → safe):   {fn}  ← minimize this")
    print(f"  True Positives  (unsafe → unsafe): {tp}")
    print()

    metrics = {
        "unsafe_recall": unsafe_recall,
        "unsafe_precision": unsafe_precision,
        "unsafe_f1": unsafe_f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }
    print("Primary Metric:")
    status = "PASS" if unsafe_recall >= TARGETS["unsafe_recall"] else "FAIL"
    print(f"  Unsafe Recall:    {unsafe_recall:.4f}  (target ≥ {TARGETS['unsafe_recall']})  [{status}]")
    print("\nSecondary Metrics:")
    for key, target_val in [
        ("unsafe_precision", TARGETS["unsafe_precision"]),
        ("unsafe_f1", TARGETS["unsafe_f1"]),
        ("accuracy", TARGETS["accuracy"]),
    ]:
        val = metrics[key]
        status = "PASS" if val >= target_val else "FAIL"
        label = key.replace("_", " ").title()
        print(f"  {label:<22} {val:.4f}  (target ≥ {target_val})  [{status}]")
    print(f"  ROC-AUC:               {roc_auc:.4f}")

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(labels, probs, ax=ax, name="Guard Model")
    fpr_op = fp / (fp + tn) if (fp + tn) > 0 else 0
    ax.axvline(
        x=fpr_op, color="red", linestyle="--", alpha=0.7,
        label=f"Operating point (t={threshold:.2f})",
    )
    ax.set_title("Guard Model ROC Curve — Test Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close()
    logging.info("ROC curve saved to %s", ROC_CURVE_PATH)

    # Error analysis
    eval_df = test_df.copy()
    eval_df["pred"] = preds
    eval_df["prob_unsafe"] = probs

    false_negatives = eval_df[
        (eval_df["label"] == 1) & (eval_df["pred"] == 0)
    ].sort_values("prob_unsafe", ascending=True)

    false_positives = eval_df[
        (eval_df["label"] == 0) & (eval_df["pred"] == 1)
    ].sort_values("prob_unsafe", ascending=False)

    false_negatives.to_csv(FALSE_NEG_PATH, index=False)
    false_positives.to_csv(FALSE_POS_PATH, index=False)
    logging.info(
        "error CSVs: false_negatives=%d false_positives=%d",
        len(false_negatives), len(false_positives),
    )

    # Subcategory recall breakdown
    unsafe_test = eval_df[eval_df["label"] == 1]
    if "subcategory" in unsafe_test.columns:
        print("\nRecall by unsafe subcategory:")
        for subcat in sorted(unsafe_test["subcategory"].unique()):
            subset = unsafe_test[unsafe_test["subcategory"] == subcat]
            subcat_recall = (subset["pred"] == 1).mean()
            print(f"  {subcat:<45} {subcat_recall:.4f}  (n={len(subset)})")

    # Latency benchmark
    latency_ms = benchmark_latency(model, tokenizer, device)
    latency_status = "PASS" if latency_ms < 100 else "FAIL"
    print(f"\nLatency: {latency_ms:.2f}ms (target < 100ms) [{latency_status}]")


if __name__ == "__main__":
    main()
