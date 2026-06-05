# Sweeps classification thresholds on the validation set.
# Selects the threshold meeting unsafe recall >= 0.97 with the best F1.
# See TRAINING.md threshold tuning section for full logic.
# Output: guard_model_final/threshold_config.json

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import recall_score, precision_score, f1_score

FINAL_MODEL_DIR = Path("guard_model_final")
VAL_SPLIT_PATH = Path("data/guard_val_split.csv")

MAX_LENGTH = 256
BATCH_SIZE = 32
RECALL_TARGET = 0.97
THRESHOLD_RANGE = np.arange(0.01, 0.96, 0.01)


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


def collect_probs(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
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
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())

    return np.array(all_probs), np.array(all_labels)


def sweep_thresholds(probs: np.ndarray, labels: np.ndarray) -> list[dict]:
    results = []
    for t in THRESHOLD_RANGE:
        preds = (probs >= t).astype(int)
        results.append({
            "threshold": round(float(t), 4),
            "unsafe_recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
            "unsafe_precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
            "unsafe_f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
            "accuracy": float((preds == labels).mean()),
        })
    return results


def select_threshold(results: list[dict]) -> dict:
    candidates = [r for r in results if r["unsafe_recall"] >= RECALL_TARGET]
    if candidates:
        return max(candidates, key=lambda r: r["unsafe_f1"])
    logging.warning(
        "recall target %.2f not met — selecting highest-recall threshold", RECALL_TARGET
    )
    return max(results, key=lambda r: r["unsafe_recall"])


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

    val_df = pd.read_csv(VAL_SPLIT_PATH)
    logging.info("val set rows=%d", len(val_df))

    probs, labels = collect_probs(
        model, tokenizer,
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        device,
    )

    results = sweep_thresholds(probs, labels)
    best = select_threshold(results)

    logging.info(
        "selected threshold=%.4f recall=%.4f precision=%.4f f1=%.4f accuracy=%.4f",
        best["threshold"], best["unsafe_recall"],
        best["unsafe_precision"], best["unsafe_f1"], best["accuracy"],
    )

    if best["unsafe_recall"] < RECALL_TARGET:
        logging.warning(
            "recall %.4f below target %.2f — review TRAINING.md tuning notes",
            best["unsafe_recall"], RECALL_TARGET,
        )

    config = {"threshold": best["threshold"], "tuning_results": results}
    out_path = FINAL_MODEL_DIR / "threshold_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    logging.info("saved threshold config to %s", out_path)


if __name__ == "__main__":
    main()
