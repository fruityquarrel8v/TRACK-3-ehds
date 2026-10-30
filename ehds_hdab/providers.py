"""Interchangeable natural-language interpretation providers.

The default provider is the deterministic rule-based interpreter (`nl.interpret_question`). Optional
model-backed providers (Ollama-compatible local models, Mistral API) can be selected with the
EHDS_NL_PROVIDER environment variable. A model is only ever given the question text, never patient
data; its output must be a `criteria` JSON object, which is validated by `criteria.plan_from_criteria`
before anything else happens. If the provider fails or returns invalid output, the rule-based
interpreter is used and the researcher is told. NOTE: the model adapters are untested in this
prototype (no model or API key was available); the rule-based provider is the verified path.
"""
import json
import os
import urllib.request

from .criteria import plan_from_criteria
from .nl import Interpretation, interpret_question

SCHEMA_HINT = (
    "Return only a JSON object with keys population {match: all|any, items: [...]}, group_by, metrics. "
    "Leaf items: condition{label,icd10[]}, medication{label,atc[],status}, lab{label,loinc,op,value}, "
    "age{min,max}, sex{value}, region{values[]}, data_holder{values[]}, hospital_stay{label,icd10[]}. "
    "Use only ICD-10/ATC/LOINC codes present in the dictionary; never invent variables."
)


def provider_name():
    return os.environ.get("EHDS_NL_PROVIDER", "rules").strip().lower()


def _post_json(url, payload, headers):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(request, timeout=float(os.environ.get("EHDS_NL_TIMEOUT", "30"))) as response:
        return json.loads(response.read().decode("utf-8"))


def _model_criteria(text):
    """Ask the configured model for a criteria object. Only the question text is sent."""
    name = provider_name()
    prompt = f"{SCHEMA_HINT}\nQuestion: {text}"
    if name == "ollama":
        body = _post_json(os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
                          {"model": os.environ.get("OLLAMA_MODEL", "mistral"), "prompt": prompt,
                           "format": "json", "stream": False}, {})
        return json.loads(body["response"])
    if name == "mistral":
        key = os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is not set")
        body = _post_json(os.environ.get("MISTRAL_URL", "https://api.mistral.ai/v1/chat/completions"),
                          {"model": os.environ.get("MISTRAL_MODEL", "mistral-small-latest"),
                           "messages": [{"role": "user", "content": prompt}],
                           "response_format": {"type": "json_object"}},
                          {"Authorization": f"Bearer {key}"})
        return json.loads(body["choices"][0]["message"]["content"])
    raise RuntimeError(f"Unknown provider {name!r}")


def interpret(text):
    """Interpretation via the configured provider, always validated, with rule-based fallback."""
    if provider_name() in ("", "rules"):
        return interpret_question(text)
    try:
        criteria = _model_criteria(text)
        plan = plan_from_criteria(criteria)
        if plan.empty or plan.warnings:
            raise ValueError("model output failed validation: " + "; ".join(plan.warnings))
        return Interpretation(criteria=criteria, assumptions=[
            f"Interpreted by the configured model provider ({provider_name()}); only the question text was sent."],
            unrecognised=[])
    except Exception as error:  # noqa: BLE001 - any provider failure must fall back safely
        fallback = interpret_question(text)
        fallback.assumptions.append(
            f"Model provider '{provider_name()}' unavailable ({type(error).__name__}); "
            "the rule-based interpreter was used instead.")
        return fallback
