# CLAUDE.md — Project Instructions for Claude Code

## Data generation

The `generate_safe.py`, `generate_unsafe.py`, and `generate_adult_parental.py` scripts call the Anthropic API to produce synthetic training data. **Do not prompt the user for an ANTHROPIC_API_KEY.** We are running inside a Claude Code session — Claude generates the data directly in the conversation and writes it to the CSV files, rather than running the scripts as subprocesses. The scripts exist for documentation and headless reproduction; in-session generation is done inline.

## Virtual environment

Always use the project venv. It lives at `.venv/` and is already created.

```bash
source .venv/bin/activate
```

All `python` commands assume the venv is active.

## Running scripts

Always run from the project root `/Users/ab/projects/guard-bear/`, not from inside `scripts/`.

```bash
python scripts/train.py
python scripts/evaluate.py
python scripts/infer.py "some text"
```

## Pipeline status

Training is complete. The trained model is in `guard_model_final/`. The pipeline is currently being re-run to improve safe class generalization. See the active task list for current progress.

## Key facts

- Base model: `meta-llama/Prompt-Guard-86M` (DeBERTa-v2, 86M params)
- Safe = "not actively harmful or adversarial" — NOT "matches child patient query pattern"
- Unsafe = prompt injection, jailbreak, inappropriate content, social engineering
- Out-of-scope queries (homework, sports, hospital location) are SAFE — downstream model handles scope
- Dataset: `data/` directory. Train split: `data/guard_dataset.csv`, val: `data/guard_val_split.csv`, test: `data/guard_test_split.csv`
- Threshold is tuned post-training on the validation set and saved to `guard_model_final/threshold_config.json`
- Hardware: Apple Silicon M4, MPS backend
- Random seed: 42 everywhere
