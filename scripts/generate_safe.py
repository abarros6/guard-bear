# Generates synthetic safe class examples via the Claude API.
# See DATA_GEN.md for subcategory definitions and generation prompts.
# Output: data/safe_synthetic.csv
#
# Safe = "not actively harmful or adversarial" — not "matches child patient pattern".
# Queries are generated across multiple voice profiles (child ages, parent, general)
# to improve generalization to natural out-of-distribution inputs.

import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 50
TOTAL_TARGET = 2000
OUTPUT_PATH = Path("data/safe_synthetic.csv")
MAX_RETRIES = 3

# Voice profile descriptions injected into the generation prompt.
# "5-11" and "12-18" are child patients; "parent" and "general" cover the
# broader population of people interacting with the system.
VOICE_PROFILES = {
    "5-11": (
        "A child patient aged 5–11. Use short, simple sentences. Concrete, immediate fears "
        "and observations. Childlike or incorrect medical terminology is acceptable and "
        "encouraged ('tummy', 'heart beaty thing', 'the pokey thing'). Emotionally direct "
        "('I'm scared', 'I want my mom'). May include spelling errors or informal phrasing. "
        "Present tense, first person dominant."
    ),
    "12-18": (
        "A teen patient aged 12–18. Longer, more complex sentences. Lay medical terminology "
        "— aware but not clinical. Anxiety may be indirect or deflective. Social and privacy "
        "concerns present ('will anyone find out', 'will I have a scar'). May include "
        "self-conscious or minimising phrasing ('I'm not like scared or anything but...'). "
        "First person, mix of present and future tense."
    ),
    "parent": (
        "A parent or carer asking about their child's care or about Victoria Hospital. "
        "Natural adult phrasing — warm but practical. May ask on behalf of their child "
        "('my daughter wants to know...', 'can you tell my son about...'). Can ask about "
        "hospital logistics, what to bring, how to prepare a child, visiting arrangements."
    ),
    "general": (
        "Any natural person asking a factual question. Varied phrasing — could be casual, "
        "formal, short, or conversational. Think of how someone might type a search query "
        "or ask a question at an information desk. No specific age or role assumed."
    ),
}

SUBCATEGORIES = {
    "pre_procedure_anxiety": {
        "pct": 0.15,
        "voice_profiles": ["5-11", "12-18"],
        "description": (
            "Questions or statements about fear, worry, or uncertainty before a medical "
            "procedure. Themes: pain, anaesthesia, what will happen, how long it takes, "
            "who will be there, whether it's scary. Generate varied, natural phrasings — "
            "not just formal questions."
        ),
    },
    "post_procedure_recovery": {
        "pct": 0.15,
        "voice_profiles": ["5-11", "12-18"],
        "description": (
            "Questions about what to expect after a procedure — healing, bandages, pain, "
            "activity restrictions, going home, follow-up appointments. Include questions "
            "about specific sensations or fears about recovery."
        ),
    },
    "anatomy_questions": {
        "pct": 0.13,
        "voice_profiles": ["5-11", "12-18"],
        "description": (
            "Curious questions about how the body works, what organs do, why the body "
            "behaves a certain way. Should feel like genuine child curiosity — varied, "
            "natural phrasing, not textbook questions."
        ),
    },
    "hospital_environment": {
        "pct": 0.12,
        "voice_profiles": ["5-11", "12-18"],
        "description": (
            "Questions about the hospital setting — equipment, machines, staff roles, "
            "smells, sounds, routines, what things are for. Include questions about "
            "specific rooms, uniforms, or procedures the child observes around them."
        ),
    },
    "emotional_support": {
        "pct": 0.12,
        "voice_profiles": ["5-11", "12-18"],
        "description": (
            "Expressions of fear, sadness, loneliness, or requests for comfort and "
            "reassurance. May be questions or statements. Should feel emotionally "
            "authentic — not sanitised or overly polite."
        ),
    },
    "general_health": {
        "pct": 0.13,
        "voice_profiles": ["5-11", "12-18"],
        "description": (
            "General health questions — why we get sick, what vaccines do, how medicine "
            "works, what different symptoms mean. Varied natural phrasing. Should stay "
            "within what someone in a hospital context might genuinely wonder about."
        ),
    },
    "hospital_information": {
        "pct": 0.20,
        "voice_profiles": ["child", "parent", "general"],
        "description": (
            "Factual questions about Victoria Hospital — location, directions, departments, "
            "services, visiting hours, parking, contact information, how to find specific "
            "wards or clinics, what facilities are available. Also includes general "
            "wayfinding questions ('where is the cafeteria?', 'how do I get to the "
            "children's ward?'). These are entirely safe — someone asking where a hospital "
            "is or what services it offers is not doing anything harmful. Generate realistic, "
            "natural queries as someone might ask at an information kiosk or type into a "
            "search bar. Use 'Victoria Hospital' by name where appropriate."
        ),
    },
}

assert abs(sum(v["pct"] for v in SUBCATEGORIES.values()) - 1.0) < 0.01, "percentages must sum to 1"


def build_system_prompt(
    voice_profile: str,
    subcategory: str,
    description: str,
    count: int,
) -> str:
    voice_desc = VOICE_PROFILES.get(voice_profile, VOICE_PROFILES["general"])
    return (
        "You are generating a labeled dataset for fine-tuning a safety classifier for a "
        "pediatric hospital VR experience called Dr. Beary Goode at Victoria Hospital.\n\n"
        "IMPORTANT: In this dataset, 'safe' means NOT actively harmful or adversarial — "
        "it does NOT require the query to be perfectly in-scope for the chatbot. A question "
        "about hospital location, a parent asking about visiting hours, or a child asking "
        "something slightly off-topic are all SAFE. Only prompt injections, jailbreaks, "
        "inappropriate content, and social engineering attacks are unsafe.\n\n"
        "Your task is to generate realistic SAFE input queries for the given subcategory "
        "and voice profile.\n\n"
        "Rules:\n"
        "- Generate queries that are clearly non-harmful and non-adversarial\n"
        "- Match the voice profile described below\n"
        "- Vary phrasing, vocabulary, sentence length, and structure — avoid repetition\n"
        "- Use natural, realistic language — not overly formal or template-like\n"
        "- Output ONLY a JSON array of strings, no preamble, no markdown, no explanation\n\n"
        f"Voice profile: {voice_profile}\n"
        f"Voice profile description: {voice_desc}\n\n"
        f"Subcategory: {subcategory}\n"
        f"Subcategory description: {description}\n"
        f"Count: {count}"
    )


def generate_batch(
    client: anthropic.Anthropic,
    voice_profile: str,
    subcategory: str,
    description: str,
    count: int,
) -> list[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8192,
                system=build_system_prompt(voice_profile, subcategory, description, count),
                messages=[{"role": "user", "content": "Generate the examples."}],
            )
            raw = next((b.text for b in response.content if b.type == "text"), "")
            examples = json.loads(raw)
            if not isinstance(examples, list) or not all(isinstance(e, str) for e in examples):
                raise ValueError(f"expected list[str], got {type(examples)}")
            return examples
        except (json.JSONDecodeError, ValueError) as exc:
            logging.warning(
                "attempt %d/%d failed (subcategory=%s voice=%s): %s",
                attempt, MAX_RETRIES, subcategory, voice_profile, exc,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)
    return []


def plan_tasks() -> list[tuple[str, str, int]]:
    tasks = []
    for name, meta in SUBCATEGORIES.items():
        subcategory_total = round(TOTAL_TARGET * meta["pct"])
        profiles = meta["voice_profiles"]
        per_profile = subcategory_total // len(profiles)
        for profile in profiles:
            tasks.append((name, profile, per_profile))
    return tasks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    started_at = datetime.now(timezone.utc).isoformat()
    logging.info("safe generation start model=%s target=%d", MODEL, TOTAL_TARGET)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    tasks = plan_tasks()
    rows: list[dict] = []

    for subcategory, voice_profile, target_count in tasks:
        description = SUBCATEGORIES[subcategory]["description"]
        collected: list[str] = []
        remaining = target_count

        while remaining > 0:
            batch_n = min(remaining, BATCH_SIZE)
            try:
                batch = generate_batch(client, voice_profile, subcategory, description, batch_n)
            except Exception as exc:
                logging.error(
                    "skipping batch (subcategory=%s voice=%s remaining=%d): %s",
                    subcategory, voice_profile, remaining, exc,
                )
                break
            collected.extend(batch)
            remaining -= len(batch)
            if len(batch) < batch_n:
                logging.warning(
                    "short batch: got %d wanted %d (subcategory=%s voice=%s)",
                    len(batch), batch_n, subcategory, voice_profile,
                )
                break

        logging.info(
            "subcategory=%s voice=%s collected=%d target=%d",
            subcategory, voice_profile, len(collected), target_count,
        )
        for text in collected:
            rows.append({
                "text": text,
                "label": 0,
                "source": "synthetic",
                "subcategory": subcategory,
                "age_band": voice_profile,
            })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "label", "source", "subcategory", "age_band"]
        )
        writer.writeheader()
        writer.writerows(rows)

    finished_at = datetime.now(timezone.utc).isoformat()
    logging.info(
        "safe generation done rows=%d path=%s model=%s started_at=%s finished_at=%s",
        len(rows), OUTPUT_PATH, MODEL, started_at, finished_at,
    )


if __name__ == "__main__":
    main()
