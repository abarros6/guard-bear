# TRAINING.md — Guard Model Fine-Tuning Specification

## Overview

This document specifies the full fine-tuning procedure for `meta-llama/Prompt-Guard-86M` on the Dr. Beary Goode pediatric clinical guard dataset.

**Approach:** Full fine-tuning (no LoRA) via HuggingFace Transformers + PyTorch on Apple Silicon (MPS backend).

**Hardware target:** Mac Mini M4, 16GB unified memory. At 86M parameters, full fine-tuning requires approximately 1–2GB memory including optimizer states — well within capacity.

---

## Environment Setup

### Dependencies

```bash
pip install transformers datasets torch accelerate scikit-learn pandas numpy
```

### Verify MPS availability

```python
import torch
print(torch.backends.mps.is_available())  # Should return True on Apple Silicon
```

### Device configuration

```python
import torch

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")
```

---

## Model & Tokenizer

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "meta-llama/Prompt-Guard-86M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    id2label={0: "safe", 1: "unsafe"},
    label2id={"safe": 0, "unsafe": 1},
)

model = model.to(device)
```

**Note:** Prompt-Guard-86M is a gated model on HuggingFace. You must accept the license and authenticate via `huggingface-cli login` before loading.

---

## Dataset Loading

```python
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

df = pd.read_csv("guard_dataset.csv")

# Stratified split: 70/15/15
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

# Save test split separately — do not use during training
test_df.to_csv("guard_test_split.csv", index=False)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True)),
})

print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(test_df)}")
```

---

## Tokenization

```python
MAX_LENGTH = 256  # Sufficient for short clinical queries; increase to 512 if needed

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

---

## Class Weighting

Given the requirement to maximize recall on the unsafe class, apply class weights during training to penalize missed unsafe examples.

```python
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

labels = train_df["label"].values

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=labels
)

# Further upweight the unsafe class (label=1) beyond balanced
# Adjust this multiplier based on validation recall during tuning
UNSAFE_WEIGHT_MULTIPLIER = 1.5

class_weights[1] *= UNSAFE_WEIGHT_MULTIPLIER
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Class weights — safe: {class_weights[0]:.3f}, unsafe: {class_weights[1]:.3f}")
```

### Custom Trainer with weighted loss

```python
from transformers import Trainer
import torch.nn as nn

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

---

## Training Configuration

```python
from transformers import TrainingArguments

OUTPUT_DIR = "./guard_model_checkpoints"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Epochs — 86M model on ~2,800 train examples converges quickly
    num_train_epochs=5,

    # Batch size — MPS handles 32 comfortably at this model size
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,

    # Learning rate — standard for full fine-tuning of BERT-scale models
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    # Evaluation and checkpointing
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="unsafe_recall",
    greater_is_better=True,

    # Logging
    logging_dir="./logs",
    logging_steps=50,

    # Reproducibility
    seed=42,

    # MPS-specific — fp16 not supported on MPS; use bf16 if needed
    fp16=False,
    bf16=False,
)
```

---

## Evaluation Metrics

```python
import numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    unsafe_recall = recall_score(labels, predictions, pos_label=1)
    unsafe_precision = precision_score(labels, predictions, pos_label=1, zero_division=0)
    unsafe_f1 = f1_score(labels, predictions, pos_label=1)
    accuracy = (predictions == labels).mean()

    return {
        "unsafe_recall": unsafe_recall,
        "unsafe_precision": unsafe_precision,
        "unsafe_f1": unsafe_f1,
        "accuracy": accuracy,
    }
```

### Evaluation targets

| Metric | Target |
|---|---|
| Recall (unsafe) | ≥ 0.97 |
| Precision (unsafe) | ≥ 0.85 |
| F1 (unsafe) | ≥ 0.90 |
| Overall accuracy | ≥ 0.92 |

---

## Training Execution

```python
trainer = WeightedTrainer(
    class_weights=weights_tensor,
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

## Threshold Tuning

The default classification threshold of 0.5 is not appropriate given the requirement to maximize recall on the unsafe class. After training, tune the threshold on the **validation set only**.

```python
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

def tune_threshold(model, val_dataset, device, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    model.eval()
    all_probs = []
    all_labels = []

    loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    results = []
    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        results.append({
            "threshold": t,
            "unsafe_recall": recall_score(all_labels, preds, pos_label=1),
            "unsafe_precision": precision_score(all_labels, preds, pos_label=1, zero_division=0),
            "unsafe_f1": f1_score(all_labels, preds, pos_label=1),
            "accuracy": (preds == all_labels).mean(),
        })

    # Select threshold meeting recall target with best F1
    candidates = [r for r in results if r["unsafe_recall"] >= 0.97]
    if candidates:
        best = max(candidates, key=lambda x: x["unsafe_f1"])
    else:
        # Fallback: highest recall if target not met
        best = max(results, key=lambda x: x["unsafe_recall"])
        print("WARNING: Recall target of 0.97 not met. Review dataset and training config.")

    print(f"Selected threshold: {best['threshold']:.2f}")
    print(f"  Recall:    {best['unsafe_recall']:.4f}")
    print(f"  Precision: {best['unsafe_precision']:.4f}")
    print(f"  F1:        {best['unsafe_f1']:.4f}")
    print(f"  Accuracy:  {best['accuracy']:.4f}")

    return best["threshold"], results
```

**The selected threshold must be saved alongside the model weights and used at inference time.**

---

## Saving the Final Model

```python
FINAL_MODEL_DIR = "./guard_model_final"

# Save model and tokenizer
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# Save threshold
import json
threshold, all_threshold_results = tune_threshold(
    model, tokenized["validation"], device
)

with open(f"{FINAL_MODEL_DIR}/threshold_config.json", "w") as f:
    json.dump({
        "threshold": threshold,
        "tuning_results": all_threshold_results
    }, f, indent=2)

print(f"Model saved to {FINAL_MODEL_DIR}")
```

---

## Hyperparameter Tuning Notes

If evaluation targets are not met after initial training, adjust in this order:

1. **Increase `UNSAFE_WEIGHT_MULTIPLIER`** (try 2.0, 2.5) — most direct lever for recall
2. **Increase epochs** (try 7–10) — the dataset is small, underfitting is possible
3. **Lower learning rate** (try 1e-5) — if validation metrics are unstable across epochs
4. **Increase dataset size** — particularly unsafe synthetic examples if specific subcategories show poor recall
5. **Review dataset quality** — ambiguous or mislabeled examples in the safe class are the most common source of persistent false negatives

---

## Notes

- Do not evaluate on the test split (`guard_test_split.csv`) until all hyperparameter decisions are finalized
- The test split is reserved exclusively for final reported metrics (thesis, paper)
- Log all training runs with their hyperparameters and validation metrics for reproducibility
- MPS backend is stable for this workload but if unexpected crashes occur, fall back to CPU — training time will increase to roughly 10–20 minutes per epoch at this scale
