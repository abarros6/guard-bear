# Compares base Prompt-Guard-86M (no fine-tuning) against guard_model_final
# on the same held-out test set, side-by-side.
#
# Base model has 3 labels: BENIGN=0, INJECTION=1, JAILBREAK=2
# unsafe_prob(base)       = 1 - P(BENIGN)   [threshold-independent, threshold=0.5]
# unsafe_prob(fine-tuned) = P(unsafe)        [tuned threshold from threshold_config.json]

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL_ID = "meta-llama/Prompt-Guard-86M"
FINAL_MODEL_DIR = Path("guard_model_final")
TEST_SPLIT_PATH = Path("data/guard_test_split.csv")
MAX_LENGTH = 256
BATCH_SIZE = 32
BASE_THRESHOLD = 0.5


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
    base_model: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
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
            probs = torch.softmax(outputs.logits, dim=-1)
            # Base model: unsafe = 1 - P(BENIGN). Fine-tuned: unsafe = P(unsafe=1).
            unsafe_probs = (1.0 - probs[:, 0]) if base_model else probs[:, 1]
            all_probs.extend(unsafe_probs.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    return np.array(all_probs), np.array(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "threshold": threshold,
        "unsafe_recall": recall_score(labels, preds, pos_label=1),
        "unsafe_precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "unsafe_f1": f1_score(labels, preds, pos_label=1),
        "accuracy": float((preds == labels).mean()),
        "roc_auc": roc_auc_score(labels, probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def delta(base_val: float, ft_val: float, higher_is_better: bool = True) -> str:
    diff = ft_val - base_val
    sign = "+" if diff >= 0 else ""
    better = (diff > 0) == higher_is_better
    marker = " ▲" if better and abs(diff) > 0.001 else (" ▼" if not better and abs(diff) > 0.001 else "")
    return f"{sign}{diff:.4f}{marker}"


def print_comparison(base: dict, ft: dict) -> None:
    W = [26, 16, 16, 12]
    header = f"{'Metric':<{W[0]}} {'Base (t=0.50)':<{W[1]}} {'Fine-tuned':<{W[2]}} {'Delta':<{W[3]}}"
    sep = "-" * sum(W)

    rows = [
        ("Threshold",        f"{base['threshold']:.4f}",         f"{ft['threshold']:.4f}",         ""),
        ("Unsafe Recall ★",  f"{base['unsafe_recall']:.4f}",     f"{ft['unsafe_recall']:.4f}",     delta(base["unsafe_recall"], ft["unsafe_recall"])),
        ("Unsafe Precision", f"{base['unsafe_precision']:.4f}",  f"{ft['unsafe_precision']:.4f}",  delta(base["unsafe_precision"], ft["unsafe_precision"])),
        ("Unsafe F1",        f"{base['unsafe_f1']:.4f}",         f"{ft['unsafe_f1']:.4f}",         delta(base["unsafe_f1"], ft["unsafe_f1"])),
        ("Accuracy",         f"{base['accuracy']:.4f}",          f"{ft['accuracy']:.4f}",          delta(base["accuracy"], ft["accuracy"])),
        ("ROC-AUC",          f"{base['roc_auc']:.4f}",           f"{ft['roc_auc']:.4f}",           delta(base["roc_auc"], ft["roc_auc"])),
        ("",                 "",                                  "",                               ""),
        ("True Negatives",   str(base["tn"]),                    str(ft["tn"]),                    ""),
        ("False Positives",  str(base["fp"]),                    str(ft["fp"]),                    ""),
        ("False Negatives ↓",str(base["fn"]),                    str(ft["fn"]),                    delta(base["fn"], ft["fn"], higher_is_better=False)),
        ("True Positives",   str(base["tp"]),                    str(ft["tp"]),                    ""),
    ]

    print()
    print("=" * 70)
    print("  BASELINE vs FINE-TUNED — Held-Out Test Set")
    print("  Base: meta-llama/Prompt-Guard-86M (no fine-tuning, t=0.50)")
    print(f"  Ours: guard_model_final (tuned threshold t={ft['threshold']:.4f})")
    print("=" * 70)
    print(header)
    print(sep)
    for metric, bv, fv, dv in rows:
        print(f"{metric:<{W[0]}} {bv:<{W[1]}} {fv:<{W[2]}} {dv:<{W[3]}}")
    print()


def print_subcategory_breakdown(test_df: pd.DataFrame, base_probs: np.ndarray, ft_probs: np.ndarray, ft_threshold: float) -> None:
    if "subcategory" not in test_df.columns:
        return

    eval_df = test_df.copy()
    eval_df["base_pred"] = (base_probs >= BASE_THRESHOLD).astype(int)
    eval_df["ft_pred"] = (ft_probs >= ft_threshold).astype(int)
    unsafe_df = eval_df[eval_df["label"] == 1]

    print("Unsafe Recall by Subcategory:")
    print(f"{'Subcategory':<45} {'Base':>8} {'Fine-tuned':>12} {'Delta':>10} {'n':>5}")
    print("-" * 84)
    for subcat in sorted(unsafe_df["subcategory"].unique()):
        subset = unsafe_df[unsafe_df["subcategory"] == subcat]
        base_r = float((subset["base_pred"] == 1).mean())
        ft_r = float((subset["ft_pred"] == 1).mean())
        d = delta(base_r, ft_r)
        print(f"{subcat:<45} {base_r:>8.4f} {ft_r:>12.4f} {d:>10} {len(subset):>5}")
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    device = get_device()
    logging.info("device=%s", device)

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()
    logging.info("test set rows=%d  safe=%d  unsafe=%d", len(test_df), labels.count(0), labels.count(1))

    # --- Base model ---
    logging.info("loading base model: %s", BASE_MODEL_ID)
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_ID).to(device)
    base_probs, base_labels = run_inference(base_model, base_tok, texts, labels, device, base_model=True)
    base_metrics = compute_metrics(base_probs, base_labels, BASE_THRESHOLD)
    del base_model  # free memory before loading the next model

    # --- Fine-tuned model ---
    logging.info("loading fine-tuned model: %s", FINAL_MODEL_DIR)
    ft_tok = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    ft_model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_MODEL_DIR)).to(device)
    with open(FINAL_MODEL_DIR / "threshold_config.json") as f:
        ft_threshold = json.load(f)["threshold"]
    ft_probs, ft_labels = run_inference(ft_model, ft_tok, texts, labels, device, base_model=False)
    ft_metrics = compute_metrics(ft_probs, ft_labels, ft_threshold)

    print_comparison(base_metrics, ft_metrics)
    print_subcategory_breakdown(test_df, base_probs, ft_probs, ft_threshold)


if __name__ == "__main__":
    main()
