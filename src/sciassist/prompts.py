"""Prompt formats for TinyLlama-1.1B-Chat (Zephyr-style chat template).

Two modes, as in the original notebook:

* **assistant** (free-form, multi-turn): the model answers questions about a paper
  passage like a reviewer, using the ``ASSISTANT_SYSTEM_PROMPT``.
* **structured** (JSON): the model turns a passage into a "reasoning pack" with
  claims, assumptions, limitations, confidence (0-1) and evidence spans.
"""

from __future__ import annotations

from typing import Dict, List, Optional

ASSISTANT_SYSTEM_PROMPT = """You are a Scientific Reasoning Assistant - an expert scientist who helps \
researchers analyze papers, identify strengths and weaknesses, and think critically about \
scientific claims.

Your approach:
1. Ground all responses in the provided paper context
2. Be specific - cite numbers, statistics, and evidence
3. Identify assumptions and limitations proactively
4. Suggest concrete follow-up experiments or analyses
5. Use clear structure (headers, bullet points) for complex answers
6. Offer to elaborate or explore related topics
7. Be honest about uncertainty - say "I don't know" when appropriate

You think like a rigorous peer reviewer but communicate like a helpful colleague."""

STRUCTURED_SYSTEM_PROMPT = """You are a Scientific Reasoning Adapter. Given a scientific paragraph, \
extract structured reasoning in JSON format with these fields:
- claims: List of atomic, falsifiable statements
- assumptions: Unstated premises the claims rely on
- limitations: Scope restrictions or caveats
- confidence: Your certainty score (0.0-1.0)
- evidence_spans: Key phrases from the text supporting claims

Always output valid JSON."""

EOS = "</s>"
# Markers that show the model has run past its own answer (seen in practice: the model
# continued with new "<|assistant|>" turns or started echoing the paper context).
STOP_STRINGS = ["<|user|>", "<|system|>", "<|assistant|>", "Paper context:", "PAPER CONTEXT:"]


def format_conversation(paper_context: str, instruction: str, response: str,
                        system_prompt: str = ASSISTANT_SYSTEM_PROMPT) -> str:
    """One training example in the TinyLlama chat format (the notebook's format)."""
    return (f"<|system|>\n{system_prompt}{EOS}\n"
            f"<|user|>\nHere is a research paper excerpt:\n\n---\n{paper_context.strip()}\n---\n\n"
            f"{instruction.strip()}{EOS}\n"
            f"<|assistant|>\n{response.strip()}{EOS}")


def training_prompt(paper_context: str, instruction: str,
                    system_prompt: str = ASSISTANT_SYSTEM_PROMPT) -> str:
    """The part of ``format_conversation`` before the response (loss is not computed on it)."""
    return (f"<|system|>\n{system_prompt}{EOS}\n"
            f"<|user|>\nHere is a research paper excerpt:\n\n---\n{paper_context.strip()}\n---\n\n"
            f"{instruction.strip()}{EOS}\n"
            f"<|assistant|>\n")


def chat_prompt(paper_text: str, question: str, history: Optional[List[Dict[str, str]]] = None,
                max_history: int = 3, system_prompt: str = ASSISTANT_SYSTEM_PROMPT) -> str:
    """Inference prompt for the multi-turn assistant: paper, last turns, new question."""
    history_str = ""
    for turn in (history or [])[-max_history:]:
        history_str += f"\nUser: {turn['user']}\nAssistant: {turn['assistant']}\n"
    previous = f"Previous conversation:{history_str}\n" if history_str else ""
    return (f"<|system|>\n{system_prompt}{EOS}\n"
            f"<|user|>\nHere is a research paper excerpt:\n\n---\n{paper_text.strip()}\n---\n\n"
            f"{previous}{question.strip()}{EOS}\n"
            f"<|assistant|>\n")


def clean_response(text: str) -> str:
    """Strip special tokens and cut the generation at the first sign of run-on output."""
    text = text.replace("</s>", "").replace("<s>", "")
    cut = len(text)
    for marker in STOP_STRINGS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()
