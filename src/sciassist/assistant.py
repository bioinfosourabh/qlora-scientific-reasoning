"""Interactive assistant: load a paper, ask questions, keep a short conversation history.

Generation details that fixed problems seen with the first notebook version:

* decode only the newly generated tokens (decoding the whole sequence echoed the prompt);
* stop when the model starts a new chat turn (``StopOnStrings``) and cut any run-on text
  (``clean_response``);
* low temperature (0.3) and a mild repetition penalty.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from .documents import ProcessedPaper, from_pdf, from_text
from .prompts import STOP_STRINGS, STRUCTURED_SYSTEM_PROMPT, chat_prompt, clean_response, training_prompt

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model(adapter_dir: Optional[str] = None, base_model: str = BASE_MODEL, load_in_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or base_model)
    kwargs = {"device_map": "auto"}
    if load_in_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def make_stopping_criteria(tokenizer, prompt_len: int, stops: List[str] = STOP_STRINGS):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnStrings(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            text = tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=False)
            return any(s in text for s in stops)

    return StoppingCriteriaList([StopOnStrings()])


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 600, temperature: float = 0.3,
             max_input_tokens: int = 1500) -> str:
    """Return the raw generated continuation (prompt tokens excluded)."""
    import torch

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    inputs = {k: enc[k].to(model.device) for k in ("input_ids", "attention_mask")}
    n = inputs["input_ids"].shape[1]
    sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9} if temperature > 0 \
        else {"do_sample": False}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling,
                             repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=make_stopping_criteria(tokenizer, n))
    return tokenizer.decode(out[0][n:], skip_special_tokens=True)


class ScientificReasoningAssistant:
    def __init__(self, model, tokenizer, max_context_chars: int = 3000):
        self.model, self.tokenizer = model, tokenizer
        self.max_context_chars = max_context_chars
        self.paper: Optional[ProcessedPaper] = None
        self.history: List[Dict[str, str]] = []

    def load_paper(self, text: Optional[str] = None, pdf_path: Optional[str] = None) -> ProcessedPaper:
        self.paper = from_pdf(pdf_path) if pdf_path else from_text(text or "")
        self.history = []
        return self.paper

    def chat(self, question: str, max_new_tokens: int = 600) -> str:
        if self.paper is None:
            raise RuntimeError("load a paper first")
        prompt = chat_prompt(self.paper.get_context(self.max_context_chars), question, self.history)
        answer = clean_response(generate(self.model, self.tokenizer, prompt, max_new_tokens))
        self.history.append({"user": question, "assistant": answer})
        return answer

    def reasoning_pack(self, max_new_tokens: int = 500) -> str:
        """Structured mode: raw JSON text for the loaded paper (parse with structured.parse_pack)."""
        if self.paper is None:
            raise RuntimeError("load a paper first")
        prompt = training_prompt(self.paper.get_context(self.max_context_chars),
                                 "Extract the structured reasoning for this text.",
                                 STRUCTURED_SYSTEM_PROMPT)
        return clean_response(generate(self.model, self.tokenizer, prompt, max_new_tokens, temperature=0.0))

    def reset(self) -> None:
        self.history = []

    def export_markdown(self) -> str:
        title = self.paper.title if self.paper else "paper"
        lines = [f"# Reasoning session: {title}", f"_{datetime.now():%Y-%m-%d %H:%M}_", ""]
        for i, t in enumerate(self.history, start=1):
            lines += [f"## Q{i}. {t['user']}", "", t["assistant"], ""]
        return "\n".join(lines)
