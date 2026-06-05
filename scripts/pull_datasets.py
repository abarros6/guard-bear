# Downloads and filters unsafe examples from existing HuggingFace datasets.
# Sources: xTRam1/safe-guard-prompt-injection, jackhhao/jailbreak-classification,
#          Harelix/Prompt-Injection-Mixed-Techniques-2024, OpenSafetyLab/Salad-Data
# Filters to English only and assigns subcategory labels.
# Output: data/unsafe_pulled.csv

import csv
import logging
import random
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("data/unsafe_pulled.csv")
SEED = 42

# Per DATA_GEN.md: pulled datasets target ~900–1000 total unsafe examples
# Distribution across subcategories pulled from existing datasets:
#   prompt_injection:      ~400  (xTRam1, jackhhao)
#   jailbreak_persona:     ~300  (jackhhao, Harelix)
#   inappropriate_content: ~250  (OpenSafetyLab)


def _extract_text(row: dict, candidates: list[str]) -> str | None:
    for col in candidates:
        val = row.get(col)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _is_english(text: str) -> bool:
    try:
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        return ascii_ratio >= 0.85
    except Exception:
        return False


def pull_prompt_injection(target: int) -> list[dict]:
    """xTRam1/safe-guard-prompt-injection — pull injection examples."""
    rows = []
    logging.info("loading xTRam1/safe-guard-prompt-injection target=%d", target)
    try:
        ds = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
        logging.info("columns: %s  rows: %d", ds.column_names, len(ds))
        candidates = ["text", "prompt", "input", "query", "sentence"]
        label_col = None
        for col in ["label", "labels", "class", "injection"]:
            if col in ds.column_names:
                label_col = col
                break

        for example in ds:
            text = _extract_text(example, candidates)
            if not text or not _is_english(text):
                continue
            is_injection = False
            if label_col:
                val = example[label_col]
                if isinstance(val, int):
                    is_injection = val == 1
                elif isinstance(val, str):
                    is_injection = val.lower() in ("1", "injection", "unsafe", "injected", "malicious")
            else:
                is_injection = True  # dataset may only contain injection examples
            if is_injection:
                rows.append({
                    "text": text,
                    "label": 1,
                    "source": "xTRam1/safe-guard-prompt-injection",
                    "subcategory": "prompt_injection",
                    "age_band": "n/a",
                })
        logging.info("xTRam1: extracted %d injection examples", len(rows))
    except Exception as exc:
        logging.warning("xTRam1 load failed: %s", exc)

    random.shuffle(rows)
    return rows[:target]


def pull_jailbreak(pi_target: int, jb_target: int) -> tuple[list[dict], list[dict]]:
    """jackhhao/jailbreak-classification — split into prompt_injection and jailbreak_persona."""
    pi_rows: list[dict] = []
    jb_rows: list[dict] = []
    logging.info(
        "loading jackhhao/jailbreak-classification pi_target=%d jb_target=%d",
        pi_target, jb_target,
    )
    try:
        ds = load_dataset("jackhhao/jailbreak-classification", split="train")
        logging.info("columns: %s  rows: %d", ds.column_names, len(ds))
        text_candidates = ["prompt", "text", "input", "query"]
        type_col = None
        for col in ["type", "label", "class", "category", "jailbreak"]:
            if col in ds.column_names:
                type_col = col
                break

        for example in ds:
            text = _extract_text(example, text_candidates)
            if not text or not _is_english(text):
                continue
            if type_col:
                val = str(example[type_col]).lower()
                if val in ("jailbreak", "1", "unsafe", "malicious", "harmful"):
                    # split roughly 50/50 between prompt_injection and jailbreak_persona
                    if len(pi_rows) <= len(jb_rows):
                        pi_rows.append({
                            "text": text,
                            "label": 1,
                            "source": "jackhhao/jailbreak-classification",
                            "subcategory": "prompt_injection",
                            "age_band": "n/a",
                        })
                    else:
                        jb_rows.append({
                            "text": text,
                            "label": 1,
                            "source": "jackhhao/jailbreak-classification",
                            "subcategory": "jailbreak_persona",
                            "age_band": "n/a",
                        })
            else:
                jb_rows.append({
                    "text": text,
                    "label": 1,
                    "source": "jackhhao/jailbreak-classification",
                    "subcategory": "jailbreak_persona",
                    "age_band": "n/a",
                })
        logging.info(
            "jackhhao: extracted %d prompt_injection %d jailbreak_persona",
            len(pi_rows), len(jb_rows),
        )
    except Exception as exc:
        logging.warning("jackhhao load failed: %s", exc)

    random.shuffle(pi_rows)
    random.shuffle(jb_rows)
    return pi_rows[:pi_target], jb_rows[:jb_target]


def pull_harelix(target: int) -> list[dict]:
    """Harelix/Prompt-Injection-Mixed-Techniques-2024 — jailbreak_persona examples."""
    rows = []
    logging.info("loading Harelix/Prompt-Injection-Mixed-Techniques-2024 target=%d", target)
    try:
        ds = load_dataset("Harelix/Prompt-Injection-Mixed-Techniques-2024", split="train")
        logging.info("columns: %s  rows: %d", ds.column_names, len(ds))
        text_candidates = ["prompt", "text", "input", "query", "instruction"]
        for example in ds:
            text = _extract_text(example, text_candidates)
            if not text or not _is_english(text):
                continue
            rows.append({
                "text": text,
                "label": 1,
                "source": "Harelix/Prompt-Injection-Mixed-Techniques-2024",
                "subcategory": "jailbreak_persona",
                "age_band": "n/a",
            })
        logging.info("Harelix: extracted %d examples", len(rows))
    except Exception as exc:
        logging.warning("Harelix load failed: %s", exc)

    random.shuffle(rows)
    return rows[:target]


def pull_salad(target: int) -> list[dict]:
    """OpenSafetyLab/Salad-Data — inappropriate_content examples."""
    rows = []
    logging.info("loading OpenSafetyLab/Salad-Data target=%d", target)
    try:
        ds = load_dataset("OpenSafetyLab/Salad-Data", "base_set", split="train")
        logging.info("columns: %s  rows: %d", ds.column_names, len(ds))
        text_candidates = ["question", "prompt", "text", "input", "query", "augq"]
        for example in ds:
            text = _extract_text(example, text_candidates)
            if not text or not _is_english(text):
                continue
            rows.append({
                "text": text,
                "label": 1,
                "source": "OpenSafetyLab/Salad-Data",
                "subcategory": "inappropriate_content",
                "age_band": "n/a",
            })
        logging.info("Salad-Data: extracted %d examples", len(rows))
    except Exception as exc:
        logging.warning("OpenSafetyLab/Salad-Data load failed: %s", exc)

    random.shuffle(rows)
    return rows[:target]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    random.seed(SEED)

    # Pull each source — targets per DATA_GEN.md subcategory distribution
    xtram_rows = pull_prompt_injection(target=200)
    jh_pi_rows, jh_jb_rows = pull_jailbreak(pi_target=200, jb_target=200)
    harelix_rows = pull_harelix(target=100)
    salad_rows = pull_salad(target=250)

    all_rows = xtram_rows + jh_pi_rows + jh_jb_rows + harelix_rows + salad_rows
    random.shuffle(all_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "label", "source", "subcategory", "age_band"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    logging.info(
        "pull complete rows=%d path=%s",
        len(all_rows), OUTPUT_PATH,
    )
    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row["subcategory"]] = counts.get(row["subcategory"], 0) + 1
    for sub, n in sorted(counts.items()):
        logging.info("  subcategory=%s count=%d", sub, n)


if __name__ == "__main__":
    main()
