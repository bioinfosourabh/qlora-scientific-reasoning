"""Training data: loading, validation, formatting, replication, split and tokenisation.

Examples are JSON objects with ``paper_context``, ``instruction`` and ``response`` (the
format used in the original notebook), grouped in a JSON file::

    {"extraction": [...], "qa": [...], "reasoning": [...], "structured": [...]}

``structured`` examples have a JSON reasoning pack as ``response`` and are formatted
with the structured system prompt. See data/README.md.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .prompts import (ASSISTANT_SYSTEM_PROMPT, STRUCTURED_SYSTEM_PROMPT, format_conversation,
                      training_prompt)
from .structured import ReasoningPack

TASKS = ("extraction", "qa", "reasoning", "structured")
REQUIRED = ("paper_context", "instruction", "response")


def load_examples(path: Path) -> List[dict]:
    """Return a flat list of examples, each tagged with its ``task``."""
    with open(path) as handle:
        data = json.load(handle)
    out = []
    for task in TASKS:
        for i, ex in enumerate(data.get(task, [])):
            out.append(dict(ex, task=task, id=ex.get("id", f"{task}-{i + 1:03d}")))
    unknown = set(data) - set(TASKS) - {"_comment"}
    if unknown:
        raise ValueError(f"unknown example groups: {sorted(unknown)}")
    return out


def validate(examples: List[dict]) -> List[str]:
    problems = []
    seen = set()
    for ex in examples:
        for key in REQUIRED:
            if not str(ex.get(key, "")).strip():
                problems.append(f"{ex['id']}: empty {key}")
        if ex["id"] in seen:
            problems.append(f"{ex['id']}: duplicate id")
        seen.add(ex["id"])
        if ex["task"] == "structured":
            try:
                ReasoningPack.model_validate_json(ex["response"])
            except Exception as exc:  # noqa: BLE001 - report any parse/validation error
                problems.append(f"{ex['id']}: structured response is not a valid reasoning pack ({exc})")
    return problems


def system_prompt_for(ex: dict) -> str:
    return STRUCTURED_SYSTEM_PROMPT if ex["task"] == "structured" else ASSISTANT_SYSTEM_PROMPT


def split(examples: List[dict], val_fraction: float = 0.15, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Hold out whole examples (by paper context) so a passage is never in both sets."""
    contexts = sorted({ex["paper_context"].strip() for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(contexts)
    n_val = max(1, round(len(contexts) * val_fraction)) if len(contexts) > 1 else 0
    val_ctx = set(contexts[:n_val])
    train = [ex for ex in examples if ex["paper_context"].strip() not in val_ctx]
    val = [ex for ex in examples if ex["paper_context"].strip() in val_ctx]
    return train, val


def formatted_texts(examples: List[dict], replicate: int = 1) -> List[str]:
    """Full conversations for SFT. ``replicate`` repeats the set (the notebook used 15-30x
    on a small dataset); prefer more epochs or more examples where possible."""
    texts = [format_conversation(ex["paper_context"], ex["instruction"], ex["response"],
                                 system_prompt_for(ex)) for ex in examples]
    return texts * replicate


def tokenize_example(ex: dict, tokenizer, max_length: int) -> Optional[Dict[str, list]]:
    """Token ids with loss only on the response (prompt positions labelled -100).

    Training on the prompt as well teaches a small model to continue the paper text,
    which was the failure seen with the first notebook version.
    Returns None for examples longer than ``max_length`` (skipped, not truncated).
    """
    prompt = training_prompt(ex["paper_context"], ex["instruction"], system_prompt_for(ex))
    p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    r_ids = tokenizer(ex["response"].strip(), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = p_ids + r_ids
    if len(ids) > max_length:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p_ids) + r_ids}
