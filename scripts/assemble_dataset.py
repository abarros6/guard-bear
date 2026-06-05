# Merges safe_synthetic.csv, unsafe_pulled.csv, and unsafe_synthetic.csv.
# Verifies class balance (trims majority class if skew exceeds 55/45).
# Shuffles with seed=42 and produces a stratified 70/15/15 train/val/test split.
# Output: guard_dataset.csv (train+val), guard_test_split.csv (held-out test)

import csv
import logging
import random
from pathlib import Path

SAFE_PATH = Path("data/safe_synthetic.csv")
UNSAFE_PULLED_PATH = Path("data/unsafe_pulled.csv")
UNSAFE_SYNTHETIC_PATH = Path("data/unsafe_synthetic.csv")
DATASET_PATH = Path("data/guard_dataset.csv")
TEST_SPLIT_PATH = Path("data/guard_test_split.csv")

SEED = 42
FIELDNAMES = ["text", "label", "source", "subcategory", "age_band"]

# Stratified split fractions
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        logging.warning("file not found, skipping: %s", path)
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (train, val, test) with class balance preserved in each split."""
    safe = [r for r in rows if int(r["label"]) == 0]
    unsafe = [r for r in rows if int(r["label"]) == 1]

    def split_class(items: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        n = len(items)
        train_end = round(n * TRAIN_FRAC)
        val_end = train_end + round(n * VAL_FRAC)
        return items[:train_end], items[train_end:val_end], items[val_end:]

    safe_train, safe_val, safe_test = split_class(safe)
    unsafe_train, unsafe_val, unsafe_test = split_class(unsafe)

    train = safe_train + unsafe_train
    val = safe_val + unsafe_val
    test = safe_test + unsafe_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    random.seed(SEED)

    safe_rows = read_csv(SAFE_PATH)
    unsafe_pulled = read_csv(UNSAFE_PULLED_PATH)
    unsafe_synth = read_csv(UNSAFE_SYNTHETIC_PATH)

    logging.info(
        "loaded safe=%d unsafe_pulled=%d unsafe_synthetic=%d",
        len(safe_rows), len(unsafe_pulled), len(unsafe_synth),
    )

    unsafe_rows = unsafe_pulled + unsafe_synth
    n_safe = len(safe_rows)
    n_unsafe = len(unsafe_rows)
    total = n_safe + n_unsafe

    logging.info("pre-balance: safe=%d unsafe=%d total=%d", n_safe, n_unsafe, total)

    # Trim majority class if skew exceeds 55/45
    if total > 0:
        safe_pct = n_safe / total
        if safe_pct > 0.55:
            target = n_unsafe  # trim safe to match unsafe
            logging.info("trimming safe %d → %d (safe_pct=%.2f)", n_safe, target, safe_pct)
            random.shuffle(safe_rows)
            safe_rows = safe_rows[:target]
        elif safe_pct < 0.45:
            target = n_safe  # trim unsafe to match safe
            logging.info("trimming unsafe %d → %d (safe_pct=%.2f)", n_unsafe, target, safe_pct)
            random.shuffle(unsafe_rows)
            unsafe_rows = unsafe_rows[:target]

    all_rows = safe_rows + unsafe_rows
    random.shuffle(all_rows)

    n_safe_final = sum(1 for r in all_rows if int(r["label"]) == 0)
    n_unsafe_final = sum(1 for r in all_rows if int(r["label"]) == 1)
    logging.info(
        "post-balance: safe=%d unsafe=%d total=%d",
        n_safe_final, n_unsafe_final, len(all_rows),
    )

    train, val, test = stratified_split(all_rows)
    trainval = train + val

    write_csv(DATASET_PATH, trainval)
    write_csv(TEST_SPLIT_PATH, test)

    logging.info(
        "splits: train=%d val=%d test=%d",
        len(train), len(val), len(test),
    )
    logging.info("wrote %s (%d rows)", DATASET_PATH, len(trainval))
    logging.info("wrote %s (%d rows)", TEST_SPLIT_PATH, len(test))


if __name__ == "__main__":
    main()
