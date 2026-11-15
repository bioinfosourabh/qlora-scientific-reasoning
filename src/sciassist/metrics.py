"""Evaluation metrics for generated answers.

* ROUGE-L F1 against the reference response (LCS over lower-cased word tokens; Lin, 2004).
* Prompt-echo rate: the answer repeats a long stretch of the paper context (the failure
  where the model "continues the paper" instead of answering).
* Run-on rate: the raw generation contains chat markers after the answer (the model
  starts new turns by itself).
* Structured mode: JSON validity (raw / after repair), schema validity, field coverage and
  grounding of evidence spans.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Dict, List, Optional

from .jsonrepair import parse_model_output
from .prompts import STOP_STRINGS, clean_response
from .structured import FIELDS, grounding, parse_pack

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def rouge_l(prediction: str, reference: str) -> float:
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for x in p:
        cur = [0] * (len(r) + 1)
        for j, y in enumerate(r, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def echoes_context(answer: str, context: str, n: int = 12) -> bool:
    """True if the answer contains any run of ``n`` consecutive context tokens."""
    a, c = tokenize(answer), tokenize(context)
    if len(a) < n or len(c) < n:
        return False
    grams = {tuple(c[i:i + n]) for i in range(len(c) - n + 1)}
    return any(tuple(a[i:i + n]) in grams for i in range(len(a) - n + 1))


def runs_on(raw: str) -> bool:
    return any(m in raw for m in STOP_STRINGS)


def score(task: str, raw: str, reference: str, context: str) -> Dict[str, Optional[float]]:
    answer = clean_response(raw)
    row: Dict[str, Optional[float]] = {
        "rougeL": rouge_l(answer, reference),
        "echoes_context": float(echoes_context(answer, context)),
        "runs_on": float(runs_on(raw)),
        "empty": float(not answer),
    }
    if task == "structured":
        pack, pr, errors = parse_pack(answer)
        obj = pr.obj if isinstance(pr.obj, dict) else {}
        row.update({
            "json_valid_raw": float(parse_model_output(answer).status == "valid"),
            "json_parsed": float(pr.ok),
            "schema_valid": float(pack is not None),
            "field_coverage": sum(f in obj for f in FIELDS) / len(FIELDS),
            "grounding": grounding(obj.get("evidence_spans") or [], context)
            if isinstance(obj.get("evidence_spans"), list) else None,
        })
    return row


def aggregate(rows: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    keys = sorted({k for r in rows for k in r})
    out: Dict[str, Optional[float]] = {"n": float(len(rows))}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = mean(vals) if vals else None
    return out
