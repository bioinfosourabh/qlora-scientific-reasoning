"""Structured "reasoning pack" output: schema, parsing with repair, grounding check."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .jsonrepair import ParseResult, parse_model_output


class ReasoningPack(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claims: List[str] = Field(min_length=1, description="Atomic, falsifiable statements")
    assumptions: List[str] = Field(default_factory=list, description="Unstated premises")
    limitations: List[str] = Field(default_factory=list, description="Scope restrictions or caveats")
    confidence: float = Field(ge=0.0, le=1.0, description="Certainty score 0-1")
    evidence_spans: List[str] = Field(default_factory=list, description="Phrases copied from the text")


FIELDS = list(ReasoningPack.model_fields)


def parse_pack(raw: str) -> Tuple[Optional[ReasoningPack], ParseResult, List[str]]:
    """Parse (with repair) and validate a model generation."""
    pr = parse_model_output(raw)
    if not pr.ok:
        return None, pr, [f"unparseable: {pr.error}"]
    try:
        return ReasoningPack.model_validate(pr.obj), pr, []
    except ValidationError as exc:
        return None, pr, [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]


def _norm(t: str) -> str:
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip().lower()


def grounding(spans: List[str], passage: str) -> Optional[float]:
    """Fraction of evidence spans that occur verbatim (case/whitespace-insensitive) in the passage."""
    spans = [s for s in spans if isinstance(s, str) and s.strip()]
    if not spans:
        return None
    p = _norm(passage)
    return sum(_norm(s).strip(" .\"'") in p for s in spans) / len(spans)
