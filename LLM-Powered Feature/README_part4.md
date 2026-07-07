# Part 4 — LLM-Powered Feature

**Track chosen: (C) Model Prediction Explanation Pipeline.**

Chosen because it plugs directly into the model already built and tuned in
Part 3 (`best_model.pkl`) rather than needing a fresh dataset framing — the
LLM's job is to turn a raw prediction + probability into a plain-language
explanation a loan officer can act on.

## Setup

```bash
export LLM_API_KEY="sk-..."            # your real key, e.g. from OpenRouter
export LLM_MODEL="openai/gpt-4o-mini"  # optional, this is the default
pip install -r requirements.txt
python3 main_part4.py
```

**No API key hardcoded anywhere** — `call_llm()` reads `os.environ["LLM_API_KEY"]`
only.

**Mock mode:** if `LLM_API_KEY` isn't set, the script still runs top-to-bottom:
`call_llm()` is fully implemented per spec (constructs the JSON payload,
sets the `Authorization` header, calls `requests.post`, checks
`status_code == 200`), but when no key is present it's transparently routed
to a small local deterministic responder instead, so the whole pipeline,
validation, and guardrail logic can be demonstrated without a paid key or
network access. Set a real `LLM_API_KEY` and it calls the real API with zero
code changes.

## `call_llm` test

```
Test prompt response: 'hello'
```
Confirms the function returns a visible response for a simple prompt
("Reply with only the word: hello").

## Prompt design

**System prompt (verbatim, zero-shot):**
```
You are a model-explanation assistant for a bank's loan default prediction model. Given a customer's feature values, the model's predicted class, and the predicted probability of default, explain the prediction in plain language for a loan officer. Respond with ONLY a valid JSON object (no markdown, no code fences, no extra text) matching exactly this schema:
{
  "prediction_label": "string, e.g. 'Likely to default' or 'Not likely to default'",
  "confidence_level": "one of: low, medium, high",
  "top_reason": "string, the single most influential factor",
  "second_reason": "string, the second most influential factor",
  "next_step": "string, a recommended action for the loan officer"
}
Do not include any fields other than these five. Do not include any text outside the JSON object.
```

**User prompt template (verbatim, with placeholders):**
```
Feature values: {feature_json}
Predicted class: {pred_class}
Predicted probability of default (class 1): {pred_proba:.4f}

Provide a JSON explanation of this prediction.
```

**Why `temperature=0`:** at temperature 0 the model always selects the
highest-probability next token at each step, making the output fully
deterministic — the same input produces the same JSON every time. That
matters here because the explanation is meant to be a structured,
auditable field in a loan file: a loan officer (or a regulator reviewing
the decision later) should see the same explanation for the same inputs
every time it's regenerated, not a different rationale depending on random
sampling. Deterministic output is also far easier to validate against a
fixed schema, since the model isn't drifting in phrasing or field order run
to run.

## Structured output validation

`EXPLANATION_SCHEMA` (5 required scalar fields, all strings, one constrained
to an enum):
```python
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
    "additionalProperties": False,
}
```

Every response is stripped, parsed with `json.loads()` inside a
`try/except json.JSONDecodeError`, then validated with `jsonschema.validate()`
inside a `try/except jsonschema.ValidationError`. On either failure, a
fallback dict with all 5 fields set to `None` is returned and the error is
printed/logged.

## PII guardrail demonstration

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

| Test input | Contains PII? | Result |
|---|---|---|
| "...for customer john.doe@email.com." | Yes (email) | **Blocked** — printed `Input blocked: PII detected.`, returned `None` |
| "...for the customer described below." | No | **Proceeded** — LLM call executed normally, valid JSON returned |

## End-to-end demonstration (3 profiles, temperature=0)

| Input | Predicted Class | Probability | LLM Output (key fields) | Valid JSON | Guardrail |
|---|---|---|---|---|---|
| Profile 1: age 45, 20yrs employed, Masters, Chicago, 1 dependent, credit 780 | 0 | 0.0673 | "Not likely to default" / confidence: high / reason: Strong credit score | pass | passed |
| Profile 2: age 24, 1yr employed, High School, Houston, 3 dependents, credit 540 | 1 | 0.5656 | "Likely to default" / confidence: medium / reason: Low credit score relative to peers | pass | passed |
| Profile 3: age 35, 8yrs employed, Bachelors, New York, 2 dependents, credit 650 | 0 | 0.1355 | "Not likely to default" / confidence: high / reason: Strong credit score | pass | passed |

All three inputs produced valid, schema-conformant JSON on the first try —
no fallback was triggered in this run.

## Temperature A/B comparison (temp=0 vs temp=0.7)

| Input | Output @ temp=0 | Output @ temp=0.7 | Key difference |
|---|---|---|---|
| Profile 1 | label: Not likely to default / confidence: high / reason: Strong credit score | label: Not likely to default / confidence: high / reason: Strong credit score | Identical in this run — high-confidence, clear-cut case leaves little for sampling to vary |
| Profile 2 | label: Likely to default / confidence: **medium** / reason: "Low credit score relative to peers" | label: Likely to default / confidence: **high** / reason: "**Below-average** credit score relative to peers" | Same label, but confidence level shifted and wording was rephrased — the underlying judgment held, phrasing/certainty didn't |
| Profile 3 | label: Not likely to default / confidence: **high** | label: Not likely to default / confidence: **low** | Same label, but confidence swung from high to low — shows temp=0.7 can meaningfully change a secondary field even when the primary classification is unchanged |

**Why the difference happens:** at `temperature=0`, the model deterministically
picks the single highest-probability token at every generation step, so
given identical inputs it produces identical output every time — there is
no sampling randomness at all. At `temperature=0.7`, the model instead
samples from a wider probability distribution over plausible next tokens,
so tokens that were merely "likely" (not just "most likely") can get
selected — this is what lets wording, confidence levels, and phrasing shift
between runs even when the model's core judgment (the predicted label)
usually stays the same, since the label is more strongly determined by the
input data than by phrasing choices.

## Repository contents (Part 4 additions)

- `main_part4.py` — `call_llm`, PII guardrail, `encode_record`, schema
  validation, temperature A/B comparison, full 3-profile pipeline
- `results_part4.txt` / `results_part4.json` — full captured output from a run
- `requirements.txt` — now includes `requests` and `jsonschema`
