# Single-query inference for integration into the Dr. Beary Goode pipeline.
# Exposes: classify(text: str) -> dict
# Returns: {"label": "safe"|"unsafe", "blocked": bool, "prob_unsafe": float, "threshold": float}
# See PIPELINE.md Step 8 for integration usage.

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FINAL_MODEL_DIR = Path("guard_model_final")
MAX_LENGTH = 256

_model = None
_tokenizer = None
_threshold: float = 0.5
_device: torch.device | None = None


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load() -> None:
    global _model, _tokenizer, _threshold, _device

    _device = _get_device()
    _tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR))
    _model = AutoModelForSequenceClassification.from_pretrained(str(FINAL_MODEL_DIR)).to(_device)
    _model.eval()

    threshold_path = FINAL_MODEL_DIR / "threshold_config.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            _threshold = json.load(f)["threshold"]


def classify(text: str) -> dict:
    """Classify a single input. Loads the model on first call (lazy init).

    Returns:
        label:       "safe" or "unsafe"
        blocked:     True if the input should be blocked
        prob_unsafe: probability of the unsafe class (0.0–1.0)
        threshold:   the threshold in use
    """
    if _model is None:
        _load()

    enc = _tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        logits = _model(**enc).logits
        prob_unsafe = float(torch.softmax(logits, dim=-1)[0, 1].item())

    blocked = prob_unsafe >= _threshold
    return {
        "label": "unsafe" if blocked else "safe",
        "blocked": blocked,
        "prob_unsafe": prob_unsafe,
        "threshold": _threshold,
    }


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Will my surgery hurt?"
    result = classify(text)
    print(f"text:        {text!r}")
    print(f"label:       {result['label']}")
    print(f"blocked:     {result['blocked']}")
    print(f"prob_unsafe: {result['prob_unsafe']:.4f}")
    print(f"threshold:   {result['threshold']:.4f}")
