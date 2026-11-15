"""QLoRA supervised fine-tuning (transformers + peft + bitsandbytes + trl).

    python -m sciassist.train --config configs/train_tinyllama_qlora.yaml

Needs a CUDA GPU. ``--smoke-test`` swaps TinyLlama for a tiny randomly initialised Llama
and a tokenizer trained on the examples, disables quantization and runs a few steps on
CPU; it checks the code path end to end and its loss values mean nothing.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from .data import load_examples, split, tokenize_example, validate
from .prompts import ASSISTANT_SYSTEM_PROMPT, format_conversation

log = logging.getLogger("sciassist.train")


def _filter_kwargs(cls, kwargs: Dict) -> Dict:
    """Keep arguments accepted by ``cls``; maps renamed arguments across library versions."""
    params = inspect.signature(cls.__init__).parameters
    renames = {"max_seq_length": "max_length", "eval_strategy": "evaluation_strategy"}
    for old, new in renames.items():
        if old in kwargs and old not in params and new in params:
            kwargs[new] = kwargs.pop(old)
    dropped = sorted(k for k in kwargs if k not in params)
    if dropped:
        log.warning("ignoring arguments not supported by %s: %s", cls.__name__, dropped)
    return {k: v for k, v in kwargs.items() if k in params}


def _dataset(rows: List[dict], tokenizer, max_len: int, replicate: int = 1):
    from datasets import Dataset

    toks = [t for t in (tokenize_example(r, tokenizer, max_len) for r in rows) if t is not None]
    if len(toks) < len(rows):
        log.warning("skipped %d examples longer than %d tokens", len(rows) - len(toks), max_len)
    return Dataset.from_list(toks * replicate)


def smoke_tokenizer(texts: List[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(texts, trainer=trainers.BpeTrainer(vocab_size=600,
                                                              special_tokens=["<unk>", "<s>", "</s>"]))
    tok.post_processor = processors.TemplateProcessing(single="<s> $A",
                                                       special_tokens=[("<s>", tok.token_to_id("<s>"))])
    tok.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                   unk_token="<unk>", pad_token="</s>")


def check_chat_template(tokenizer) -> None:
    """Warn if our prompt format differs from the tokenizer's own chat template."""
    if not getattr(tokenizer, "chat_template", None):
        return
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"}]
    theirs = tokenizer.apply_chat_template(msgs, tokenize=False).strip()
    ours = "<|system|>\nS</s>\n<|user|>\nU</s>\n<|assistant|>\nA</s>"
    if theirs != ours:
        log.warning("chat template differs from the expected format:\n%r\n%r", theirs, ours)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning of TinyLlama")
    p.add_argument("--config", type=Path, default=Path("configs/train_tinyllama_qlora.yaml"))
    p.add_argument("--examples", type=Path, help="override data.examples")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--max-steps", type=int)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq)
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(a.config.read_text())
    mcfg, qcfg, lcfg, dcfg = cfg["model"], cfg["quantization"], cfg["lora"], cfg["data"]
    tcfg = dict(cfg["training"])
    out_dir = Path(a.output_dir or tcfg.pop("output_dir"))
    tcfg.pop("output_dir", None)

    examples = load_examples(a.examples or Path(dcfg["examples"]))
    problems = validate(examples)
    if problems:
        raise SystemExit("invalid examples:\n" + "\n".join(problems))
    train_rows, val_rows = split(examples, dcfg.get("val_fraction", 0.15))
    log.info("examples: %d train, %d validation", len(train_rows), len(val_rows))

    if a.smoke_test:
        from transformers import LlamaConfig, LlamaForCausalLM

        tokenizer = smoke_tokenizer([format_conversation(e["paper_context"], e["instruction"],
                                                         e["response"], ASSISTANT_SYSTEM_PROMPT)
                                     for e in examples])
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=2048,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id))
        tcfg.update(fp16=False, optim="adamw_torch", gradient_checkpointing=False,
                    eval_strategy="no", save_strategy="no", logging_steps=1,
                    gradient_accumulation_steps=1)
        max_steps = a.max_steps or 3
    else:
        if not torch.cuda.is_available():
            raise SystemExit("QLoRA training needs a CUDA GPU (use --smoke-test on CPU).")
        tokenizer = AutoTokenizer.from_pretrained(mcfg["base_model"])
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        check_chat_template(tokenizer)
        bnb = BitsAndBytesConfig(load_in_4bit=qcfg["load_in_4bit"],
                                 bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
                                 bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
                                 bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]))
        model = AutoModelForCausalLM.from_pretrained(mcfg["base_model"], quantization_config=bnb,
                                                     device_map="auto", torch_dtype=torch.float16)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=tcfg.get(
            "gradient_checkpointing", True))
        max_steps = a.max_steps or -1
    tokenizer.padding_side = "right"
    model = get_peft_model(model, LoraConfig(**lcfg))
    model.print_trainable_parameters()

    max_len = mcfg.get("max_seq_length", 1024)
    train_ds = _dataset(train_rows, tokenizer, max_len, dcfg.get("replicate", 1))
    val_ds = _dataset(val_rows, tokenizer, max_len) if val_rows else None
    if val_ds is None:
        tcfg["eval_strategy"] = "no"
    args = SFTConfig(**_filter_kwargs(SFTConfig, dict(
        tcfg, output_dir=str(out_dir), max_steps=max_steps, max_seq_length=max_len, packing=False,
        remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True})))
    kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
              data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    params = inspect.signature(SFTTrainer.__init__).parameters
    kw["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**kw)
    trainer.train()

    adapter = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    (out_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (out_dir / "split.json").write_text(json.dumps({"train": [e["id"] for e in train_rows],
                                                    "val": [e["id"] for e in val_rows]}, indent=2))
    log.info("adapter saved to %s", adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
