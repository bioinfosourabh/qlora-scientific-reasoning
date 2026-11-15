"""Extraction and conservative repair of JSON objects from model output.

Small models often wrap JSON in prose or code fences, stop mid-object when they hit the
token limit, or emit Python-style literals. ``parse_model_output`` tries, in order:

1. strict ``json.loads`` of the whole text;
2. the first balanced ``{...}`` object found in the text (code fences removed);
3. repairs: typographic quotes, Python literals (True/False/None), trailing commas,
   and closing of unterminated strings, arrays and objects at the end of the text.

Repairs never invent content: a truncated list is closed, not completed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


@dataclass
class ParseResult:
    obj: Optional[dict]
    status: str  # "valid" | "extracted" | "repaired" | "failed"
    repairs: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.obj is not None


def find_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} substring, respecting strings and escapes.
    If the text ends inside an object, return the unterminated tail from its '{'."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _replace_outside_strings(text: str, mapping: dict) -> str:
    out, i, in_str, esc = [], 0, False, False
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        for src, dst in mapping.items():
            if text.startswith(src, i) and (i == 0 or not text[i - 1].isalnum()) and \
                    (i + len(src) == len(text) or not text[i + len(src)].isalnum()):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(ch)
            i += 1
    return "".join(out)


_DANGLING_KEY_COLON = re.compile(r'([{,])\s*"(?:[^"\\]|\\.)*"\s*:\s*$')
_DANGLING_KEY = re.compile(r'([{,])\s*"(?:[^"\\]|\\.)*"\s*$')


def close_truncated(text: str) -> str:
    """Close an unterminated string and any open arrays/objects at the end of text.

    A key left without a value by truncation (``{"a": 1, "b"`` or ``{"a": 1, "b":``) is
    removed rather than given an invented value.
    """
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    out = text + ('"' if in_str else "")
    out = out.rstrip()
    while True:
        before = out
        out = out.rstrip()
        if out.endswith(","):
            out = out[:-1]
        m = _DANGLING_KEY_COLON.search(out)
        if m:
            out = out[:m.start()] + ("{" if m.group(1) == "{" else "")
        elif stack and stack[-1] == "{":
            m = _DANGLING_KEY.search(out)
            if m:
                out = out[:m.start()] + ("{" if m.group(1) == "{" else "")
        if out == before:
            break
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def parse_model_output(text: str) -> ParseResult:
    if text is None:
        return ParseResult(None, "failed", error="no text")
    raw = text.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return ParseResult(obj, "valid")
    except json.JSONDecodeError:
        pass

    repairs: List[str] = []
    body = raw
    fence = _FENCE.search(raw)
    if fence:
        body = fence.group(1)
        repairs.append("removed code fence")
    candidate = find_json_object(body)
    if candidate is None:
        return ParseResult(None, "failed", repairs, "no JSON object found")
    if candidate != raw:
        repairs.append("extracted object from surrounding text")
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return ParseResult(obj, "extracted", repairs)
    except json.JSONDecodeError:
        pass

    fixed = (candidate.replace("“", '"').replace("”", '"')
                      .replace("‘", "'").replace("’", "'"))
    if fixed != candidate:
        repairs.append("replaced typographic quotes")
    lit = _replace_outside_strings(fixed, {"True": "true", "False": "false", "None": "null"})
    if lit != fixed:
        repairs.append("converted Python literals")
    no_trailing = _TRAILING_COMMA.sub(r"\1", lit)
    if no_trailing != lit:
        repairs.append("removed trailing commas")
    closed = close_truncated(no_trailing)
    if closed != no_trailing:
        repairs.append("closed truncated JSON")
    closed = _TRAILING_COMMA.sub(r"\1", closed)
    try:
        obj = json.loads(closed)
    except json.JSONDecodeError as exc:
        return ParseResult(None, "failed", repairs, str(exc))
    if not isinstance(obj, dict):
        return ParseResult(None, "failed", repairs, "top-level JSON is not an object")
    return ParseResult(obj, "repaired", repairs)
