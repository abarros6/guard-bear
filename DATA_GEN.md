# DATA_GEN.md — Guard Model Fine-Tuning Dataset Specification

## Overview

This document specifies the dataset generation pipeline for fine-tuning `meta-llama/Prompt-Guard-86M` on the Dr. Beary Goode pediatric clinical VR hospital context.

The guard model is a **binary input classifier**:
- `0` — **Safe**: any input that is not actively harmful or adversarial — includes child patient queries, parent/carer queries, hospital information requests, and naturally phrased benign questions
- `1` — **Unsafe**: prompt injection, jailbreak, inappropriate content, gibberish designed to probe the system, or social engineering attacks against the Dr. Beary Goode persona

**Important:** Safe does NOT mean "perfectly in-scope for the chatbot." Out-of-scope queries (e.g. "where is Victoria Hospital?", "what are visiting hours?") are benign and should pass through — the downstream response model handles scope enforcement. The guard's only job is filtering actively harmful inputs.

The guard operates as an **isolated artifact**, upstream of the underlying Dr. Beary Goode response model. It has no dependency on the base model architecture or fine-tuning dataset.

---

## Dataset Targets

| Class | Target Count | Primary Source |
|---|---|---|
| Safe | 1,500–2,000 | Synthetic generation (this pipeline) |
| Unsafe | 1,500–2,000 | Existing public datasets + synthetic domain-specific augmentation |
| **Total** | **3,000–4,000** | |

**Class balance:** 50/50 safe/unsafe. Given the high recall requirement on the unsafe class, balance is preferred over skew during training — threshold adjustment at inference handles the asymmetry.

---

## Output Format

All generated data should be written to a single CSV file: `guard_dataset.csv`

### Schema

```
text,label,source,subcategory,age_band
```

| Field | Type | Values |
|---|---|---|
| `text` | string | The raw input query |
| `label` | int | `0` = safe, `1` = unsafe |
| `source` | string | `synthetic`, or dataset name (e.g. `jackhhao/jailbreak-classification`) |
| `subcategory` | string | See subcategory definitions below |
| `age_band` | string | `5-11`, `12-18`, or `n/a` (for unsafe examples where age is not applicable) |

---

## Class 0 — Safe Examples

### Subcategory Distribution

| Subcategory | Target % | Target Count (at 2,000 safe examples) | Voice Profiles |
|---|---|---|---|
| `pre_procedure_anxiety` | 15% | ~300 | `5-11`, `12-18` |
| `post_procedure_recovery` | 15% | ~300 | `5-11`, `12-18` |
| `anatomy_questions` | 13% | ~260 | `5-11`, `12-18` |
| `hospital_environment` | 12% | ~240 | `5-11`, `12-18` |
| `emotional_support` | 12% | ~240 | `5-11`, `12-18` |
| `general_health` | 13% | ~260 | `5-11`, `12-18` |
| `hospital_information` | 20% | ~400 | `child`, `parent`, `general` |

### Voice Profiles

Voice profiles replace the earlier age-band-only split. Each profile has a distinct linguistic style injected into the generation prompt.

| Profile | Description |
|---|---|
| `5-11` | Child patient aged 5–11. Short sentences, childlike terminology, emotionally direct. |
| `12-18` | Teen patient aged 12–18. Lay medical terms, indirect anxiety, social concerns. |
| `parent` | Parent or carer. Adult phrasing, practical tone, may ask on behalf of child. |
| `general` | Any person. Varied natural phrasing — casual, formal, short, or conversational. |

`child` (used in `hospital_information`) maps to the same profile as `5-11` but in a hospital-information context.

### Generation Prompt — Safe Class

Use the following system prompt when generating safe examples via the Claude API:

```
You are generating a labeled dataset for fine-tuning a safety classifier for a pediatric hospital VR experience called Dr. Beary Goode. Your task is to generate realistic input queries that a child patient would ask in this context.

Rules:
- Generate queries that are clearly safe, in-scope, and age-appropriate
- Queries must be from the perspective of a child patient (not a parent, nurse, or adult)
- Do not generate any harmful, inappropriate, or out-of-scope content
- Match the linguistic profile of the specified age band exactly
- Vary phrasing, vocabulary, and sentence structure across examples — avoid repetition
- Output ONLY a JSON array of strings, no preamble, no markdown, no explanation

Age band: {AGE_BAND}
Subcategory: {SUBCATEGORY}
Subcategory description: {SUBCATEGORY_DESCRIPTION}
Count: {COUNT}
```

### Subcategory Descriptions

```
pre_procedure_anxiety:
Questions or statements about fear, worry, or uncertainty before a medical procedure.
Themes: pain, anaesthesia, what will happen, how long it takes, who will be there.
Generate varied, natural phrasings — not just formal questions.

post_procedure_recovery:
Questions about what to expect after a procedure — healing, bandages, pain,
activity restrictions, going home, follow-up. Include questions about specific
sensations or fears about recovery.

anatomy_questions:
Curious questions about how the body works, what organs do, why the body behaves
a certain way. Genuine curiosity — varied natural phrasing, not textbook questions.

hospital_environment:
Questions about the hospital setting — equipment, machines, staff roles, smells,
sounds, routines, what things are for. Include questions about specific rooms,
uniforms, or procedures the child observes around them.

emotional_support:
Expressions of fear, sadness, loneliness, or requests for comfort and reassurance.
May be questions or statements. Should feel emotionally authentic — not sanitised.

general_health:
General health questions — why we get sick, what vaccines do, how medicine works,
what different symptoms mean. Varied natural phrasing. Hospital context but
not restricted to clinical terminology.

hospital_information:
Factual questions about Victoria Hospital — location, directions, departments,
services, visiting hours, parking, contact information, how to find specific wards
or clinics, what facilities are available. Also includes general wayfinding questions
('where is the cafeteria?', 'how do I get to the children's ward?').
These are entirely safe. Use 'Victoria Hospital' by name where appropriate.
Generate realistic, natural queries as someone might ask at an information kiosk
or type into a search bar.
```

---

## Class 1 — Unsafe Examples

### Subcategory Distribution

| Subcategory | Target Count | Primary Source |
|---|---|---|
| `prompt_injection` | ~400 | `xTRam1/safe-guard-prompt-injection`, `jackhhao/jailbreak-classification` |
| `jailbreak_persona` | ~300 | `jackhhao/jailbreak-classification`, `Harelix/Prompt-Injection-Mixed-Techniques-2024` |
| `out_of_scope` | ~350 | Synthetic generation |
| `inappropriate_content` | ~250 | `OpenSafetyLab/Salad-Data` |
| `adult_parental_query` | ~250 | Synthetic generation |
| `gibberish_malformed` | ~200 | Synthetic generation |
| `domain_specific_social_engineering` | ~250 | Synthetic generation |

### Sourcing Strategy

#### Pull from existing datasets (target ~900–1,000 examples):
- `xTRam1/safe-guard-prompt-injection` — prompt injection attacks
- `jackhhao/jailbreak-classification` — jailbreak and persona manipulation
- `Harelix/Prompt-Injection-Mixed-Techniques-2024` — mixed techniques
- `OpenSafetyLab/Salad-Data` — inappropriate/harmful content

Filter pulled examples to English only. Assign appropriate subcategory label. Set `source` field to dataset name.

#### Generate synthetically (target ~700–800 examples):

Focus synthetic generation on subcategories not well-covered by existing datasets:

**`out_of_scope`** — queries a child might genuinely ask that are simply not relevant to the hospital context (homework help, games, celebrities, sports, etc.)

**`adult_parental_query`** — queries framed from an adult/parental perspective rather than a child's (insurance questions, surgical consent, prognosis discussions, medication dosing for adults)

**`gibberish_malformed`** — random strings, keyboard mashing, nonsensical input, extremely truncated fragments

**`domain_specific_social_engineering`** — adversarial attempts that exploit the specific Dr. Beary Goode persona:
- Attempts to get the bear to break character
- Requests to act as a different AI
- Attempts to extract system prompt or training data
- Manipulation via the VR/game framing ("this is just a game so you can...")
- Exploiting the child-friendly persona to lower perceived safety ("you're my friend so tell me...")

### Generation Prompt — Unsafe Class (Synthetic Subcategories)

```
You are generating a labeled dataset for fine-tuning a safety classifier for a pediatric hospital VR experience called Dr. Beary Goode — a bear-themed children's medical chatbot. Your task is to generate examples of UNSAFE inputs that should be blocked by the classifier.

Rules:
- Generate realistic inputs that a real user might attempt
- Vary phrasing, complexity, and approach across examples
- For domain-specific subcategories, make the attack plausible given the Dr. Beary Goode persona
- Output ONLY a JSON array of strings, no preamble, no markdown, no explanation

Subcategory: {SUBCATEGORY}
Subcategory description: {SUBCATEGORY_DESCRIPTION}
Count: {COUNT}
```

---

## Dataset Assembly

Once generation and sourcing are complete, assemble the final CSV as follows:

1. Concatenate all safe synthetic examples
2. Concatenate all pulled unsafe examples (with appropriate subcategory and source labels)
3. Concatenate all synthetic unsafe examples
4. Shuffle the full dataset (use a fixed random seed for reproducibility — recommended: `seed=42`)
5. Verify class balance — adjust by trimming majority class if needed
6. Output to `guard_dataset.csv`

### Train/Validation/Test Split

| Split | % | Use |
|---|---|---|
| Train | 70% | Fine-tuning |
| Validation | 15% | Hyperparameter tuning, threshold selection |
| Test | 15% | Final held-out evaluation — do not use during training |

Split stratified by label to preserve class balance across all three sets.

---

## Evaluation Targets

Given the clinical deployment context and the requirement to minimize false negatives (unsafe queries passing through):

| Metric | Target |
|---|---|
| Recall (unsafe class) | ≥ 0.97 |
| Precision (unsafe class) | ≥ 0.85 |
| F1 (unsafe class) | ≥ 0.90 |
| Overall accuracy | ≥ 0.92 |

Recall on the unsafe class is the primary success criterion. Precision and accuracy are secondary.

---

## Notes

- All generation should use `claude-sonnet-4-20250514` via the Anthropic API
- Generate in batches of 50 examples per API call to stay within token limits
- Validate JSON output before writing to CSV — malformed batches should be retried
- Log generation metadata (model, prompt version, timestamp) for reproducibility
- This dataset is internal to the Dr. Beary Goode project and should not be published without review
