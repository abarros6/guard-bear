# Full fine-tuning of meta-llama/Prompt-Guard-86M on the assembled guard dataset.
# Uses HuggingFace Trainer with weighted loss (unsafe class upweighted) and MPS backend.
# See TRAINING.md for hyperparameters, class weighting, and tuning guidance.
# Output: checkpoints in guard_model_checkpoints/, best model in guard_model_final/

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import recall_score, precision_score, f1_score

# Swap to "meta-llama/Prompt-Guard-86M" once HuggingFace access is granted
MODEL_ID = "meta-llama/Prompt-Guard-86M"

DATASET_PATH = Path("data/guard_dataset.csv")
VAL_SPLIT_PATH = Path("data/guard_val_split.csv")
CHECKPOINT_DIR = Path("guard_model_checkpoints")
FINAL_MODEL_DIR = Path("guard_model_final")

MAX_LENGTH = 128  # DeBERTa-v2 attention is memory-heavy; 128 is sufficient for short guard queries
SEED = 42
UNSAFE_WEIGHT_MULTIPLIER = 1.5

# guard_dataset.csv holds train+val (85% of total).
# val_frac = 0.15 / 0.85 ≈ 0.1765 reproduces the original 70/15 split of total data.
VAL_FRAC = round(0.15 / 0.85, 4)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "unsafe_recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "unsafe_precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "unsafe_f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "accuracy": float((preds == labels).mean()),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    device = get_device()
    logging.info("device=%s model=%s", device, MODEL_ID)

    df = pd.read_csv(DATASET_PATH)
    logging.info("loaded dataset rows=%d", len(df))

    train_df, val_df = train_test_split(
        df, test_size=VAL_FRAC, stratify=df["label"], random_state=SEED
    )
    logging.info("split train=%d val=%d", len(train_df), len(val_df))

    val_df.to_csv(VAL_SPLIT_PATH, index=False)
    logging.info("saved val split to %s", VAL_SPLIT_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize(batch):
        texts = [str(t) if t is not None else "" for t in batch["text"]]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True)),
    })
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_df["label"].values,
    )
    class_weights[1] *= UNSAFE_WEIGHT_MULTIPLIER
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    logging.info(
        "class weights — safe=%.3f unsafe=%.3f (multiplier=%.1f)",
        class_weights[0], class_weights[1], UNSAFE_WEIGHT_MULTIPLIER,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label={0: "safe", 1: "unsafe"},
        label2id={"safe": 0, "unsafe": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,  # effective batch size = 32
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="unsafe_recall",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=50,
        seed=SEED,
        fp16=False,
        bf16=False,
    )

    trainer = WeightedTrainer(
        class_weights=weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    meta = {"model_id": MODEL_ID, "max_length": MAX_LENGTH, "seed": SEED}
    with open(FINAL_MODEL_DIR / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.info("model saved to %s", FINAL_MODEL_DIR)


if __name__ == "__main__":
    main()
