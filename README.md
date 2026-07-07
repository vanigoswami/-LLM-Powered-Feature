 LLM-Powered Feature

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
