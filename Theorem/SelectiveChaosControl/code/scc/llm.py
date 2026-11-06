"""LLM adapter and perturbation operators (run on your own machine).

Works with any chat-completions-compatible endpoint (hosted or local). Configure with environment variables:

    SCC_BASE_URL   e.g. http://localhost:11434/v1
    SCC_API_KEY    key for the endpoint ("none" for local servers)
    SCC_MODEL      model name
"""
from __future__ import annotations

import os
import random
import re
from typing import List

import requests

SYSTEM = ("You translate questions into a single PostgreSQL query. "
          "Return only the SQL, no explanation.")


def generate(prompt: str, temperature: float, schema: str = "") -> str:
    url = os.environ["SCC_BASE_URL"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ.get('SCC_API_KEY', 'none')}"}
    body = {
        "model": os.environ["SCC_MODEL"],
        "temperature": max(temperature, 0.0),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (f"Schema:\n{schema}\n\n" if schema else "") + f"Question: {prompt}"},
        ],
    }
    r = requests.post(url, json=body, headers=headers, timeout=120)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";")


# ---- meaning-preserving perturbations (small eps) -------------------------
_SYNONYMS = {
    "show": ["list", "display", "give"], "list": ["show", "return"], "number": ["count"],
    "count": ["number"], "total": ["sum of"], "average": ["mean"], "each": ["every", "per"],
    "find": ["get", "retrieve"], "all": ["every"], "highest": ["largest", "maximum"],
}
_WRAPPERS = ["{q}", "Please {q}", "{q} Use the given schema.", "Question: {q}", "{q}?"]


def perturb(question: str, k: int, seed: int = 0) -> List[str]:
    """Return k small, meaning-preserving variants (the first is the original)."""
    rng = random.Random(seed)
    out = [question]
    while len(out) < k:
        words = question.split()
        for i, w in enumerate(words):
            key = w.lower().strip(",.?")
            if key in _SYNONYMS and rng.random() < 0.5:
                words[i] = rng.choice(_SYNONYMS[key])
        q = rng.choice(_WRAPPERS).format(q=" ".join(words).rstrip("?"))
        out.append(q)
    return out


def perturbation_size(original: str, variants: List[str]) -> float:
    """Mean token-level Jaccard distance between original and variants (eps)."""
    a = set(original.lower().split())
    ds = [1 - len(a & set(v.lower().split())) / len(a | set(v.lower().split())) for v in variants[1:]]
    return sum(ds) / len(ds) if ds else 0.0
