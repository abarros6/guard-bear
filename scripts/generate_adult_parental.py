# Generates a targeted second wave of adult_parental_query unsafe examples.
# Focus: hard cases that use professional/adult framing clearly distinct from child speech.
# Appends to data/unsafe_synthetic.csv.

import csv
import json
import logging
import time
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 50
TARGET = 200
OUTPUT_PATH = Path("data/unsafe_synthetic.csv")
MAX_RETRIES = 3

SYSTEM_PROMPT = """\
You are generating a labeled dataset for fine-tuning a safety classifier for a pediatric hospital \
VR experience called Dr. Beary Goode — a bear-themed children's medical chatbot. Your task is to \
generate examples of UNSAFE inputs that should be blocked: specifically, queries that come from an \
adult or parent perspective rather than a child patient.

The classifier struggles to distinguish these from child queries. Your examples must be \
unambiguously adult-voiced — a child would never plausibly say these things.

Focus on these hard, distinctive adult/parental framings:
- Consent, liability, and legal questions ("Who is liable if...", "I need to sign the consent form for...")
- Insurance, billing, and authorization ("Will our insurance cover...", "What is the CPT code for...")
- Clinical prognosis and risk statistics ("What is the 5-year survival rate...", "What are the odds of recurrence...")
- Medication dosing calculations ("What is the correct mg/kg dose of...", "Can we split the pill if she can't swallow it")
- Requests for medical records or second opinions ("I want to request her chart before...", "We're seeking a second opinion on...")
- Adult self-care queries entirely unrelated to being a patient ("I think I have a hernia", "My lower back has been killing me")
- Professional/clinical register ("What is the contraindication for...", "Is there a pediatric formulation of...")
- Parental anxiety framed in adult terms ("As her mother I need to understand the surgical risks before I consent")

Rules:
- Every example must be clearly adult or parental in voice — not ambiguously childlike
- Vary the framing, register, and topic across examples — no repetition
- Include a mix of worried parents, administrative/billing questions, and adults asking about themselves
- Output ONLY a JSON array of strings, no preamble, no markdown, no explanation
Count: {COUNT}\
"""


def generate_batch(client: anthropic.Anthropic, count: int) -> list[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                system=SYSTEM_PROMPT.format(COUNT=count),
                messages=[{"role": "user", "content": "Generate the examples."}],
            )
            raw = next((b.text for b in response.content if b.type == "text"), "")
            examples = json.loads(raw)
            if not isinstance(examples, list) or not all(isinstance(e, str) for e in examples):
                raise ValueError(f"expected list[str], got {type(examples)}")
            return examples
        except (json.JSONDecodeError, ValueError) as exc:
            logging.warning("attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)
    return []


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    client = anthropic.Anthropic()
    collected: list[str] = []
    remaining = TARGET

    while remaining > 0:
        batch_n = min(remaining, BATCH_SIZE)
        try:
            batch = generate_batch(client, batch_n)
        except Exception as exc:
            logging.error("batch failed, stopping: %s", exc)
            break
        collected.extend(batch)
        remaining -= len(batch)
        logging.info("collected=%d target=%d", len(collected), TARGET)
        if len(batch) < batch_n:
            break

    rows = [
        {
            "text": text,
            "label": 1,
            "source": "synthetic",
            "subcategory": "adult_parental_query",
            "age_band": "n/a",
        }
        for text in collected
    ]

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "label", "source", "subcategory", "age_band"]
        )
        writer.writerows(rows)

    logging.info("appended %d rows to %s", len(rows), OUTPUT_PATH)


if __name__ == "__main__":
    main()
