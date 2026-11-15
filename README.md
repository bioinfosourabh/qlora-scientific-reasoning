# qlora-scientific-reasoning

A small language-model assistant for critical reading of scientific papers. A LoRA
adapter is trained with QLoRA on TinyLlama-1.1B-Chat so that, given a passage from a paper,
the model:

* answers questions about it the way a reviewer would (claims, assumptions,
  limitations, adequacy of the design, experiments that would strengthen it), over
  several turns of conversation; and
* in structured mode, returns a JSON "reasoning pack": `claims`, `assumptions`,
  `limitations`, `confidence` (0-1) and `evidence_spans` copied from the text.

The project is sized for the free Google Colab tier (one T4 GPU, limited disk): the base
model is loaded in 4 bit, only the adapter is trained and saved, and sequences are at most
1,024 tokens.

## Overview

```
paper text / PDF --> documents.py (PyMuPDF text, abstract, sections, 1,500-char chunks with 200 overlap)
                         |
                         v
prompts.py  <|system|> ... </s> <|user|> excerpt + history + question </s> <|assistant|>
                         |
TinyLlama-1.1B-Chat (4-bit NF4) + LoRA adapter (r=16, alpha=32, q/k/v/o)
                         |
generate: decode new tokens only, stop at new chat turns, clean run-on text
                         |
free-form answer  or  JSON reasoning pack (parse, repair, validate, check evidence spans)
```

## Method

**Model and fine-tuning.** QLoRA (Dettmers et al., 2023): the base weights are quantised
to 4-bit NormalFloat with double quantisation and fp16 compute (the T4 has no bfloat16),
and low-rank adapters (Hu et al., 2021) with r = 16, alpha = 32 and dropout 0.05 are
trained on the attention projections. Batch size 2 with gradient accumulation 4, learning
rate 2e-4 with cosine decay and 10% warm-up, paged 8-bit AdamW, gradient checkpointing.
Training uses `transformers`, `peft`, `bitsandbytes` and `trl` (`SFTTrainer`).

**Prompt format.** TinyLlama-1.1B-Chat-v1.0 uses the Zephyr-style template
`<|system|>\n...</s>\n<|user|>\n...</s>\n<|assistant|>\n`; `prompts.format_conversation`
reproduces it (checked against the tokenizer's chat template at training time).

**Lessons from the first runs, and what changed.** The first notebook version showed three
failure modes, and the code addresses each:

| Observed | Cause | Change |
|---|---|---|
| The answer started with the whole prompt | the full output sequence was decoded | decode only tokens after the prompt (`assistant.generate`) |
| The model continued the paper text instead of answering | too few examples and epochs; loss on the prompt tokens | 10 epochs, more and more varied examples, loss on the response only (`data.tokenize_example`) |
| After answering, the model wrote new `<|assistant|>` turns on its own | no stop condition | stopping criteria on chat markers plus `prompts.clean_response` |

**Evaluation** (`sciassist.evaluate`), on examples held out by passage: ROUGE-L F1 against
the reference answer (Lin, 2004), the rate of answers that echo the passage, the rate of
run-on generations, and for structured mode JSON validity before and after repair, schema
validity, field coverage and the fraction of evidence spans found verbatim in the passage.

### References

- Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. QLoRA: Efficient finetuning of
  quantized LLMs. NeurIPS 2023. arXiv:2305.14314
- Hu EJ, et al. LoRA: Low-rank adaptation of large language models. ICLR 2022. arXiv:2106.09685
- Zhang P, Zeng G, Wang T, Lu W. TinyLlama: An open-source small language model.
  arXiv:2401.02385 (2024)
- Lin C-Y. ROUGE: A package for automatic evaluation of summaries. Text Summarization
  Branches Out (ACL workshop), 2004.

## Install

```bash
git clone https://github.com/bioinfosourabh/qlora-scientific-reasoning.git
cd qlora-scientific-reasoning
pip install -r requirements.txt && pip install -e .   # GPU machine
pip install -r requirements-cpu.txt && pip install -e .   # CPU: data, metrics, tests only
```

`requirements.txt` pins a stack that was installed and run together (CPU smoke test,
below). The first notebook version used transformers 4.36, peft 0.7, trl 0.7 and
bitsandbytes 0.41; those versions no longer install cleanly on current Colab images.

## Quick start

On Colab, open `notebooks/Scientific_Reasoning_Assistant_Colab.ipynb` and run the cells.
From a shell with a GPU:

```bash
python -m sciassist.train --config configs/train_tinyllama_qlora.yaml        # writes outputs/.../adapter
python -m sciassist.evaluate --adapter outputs/sci-reasoning-assistant/adapter \
    --examples data/examples.json --split outputs/sci-reasoning-assistant/split.json --out outputs/eval
python -m sciassist.app --adapter outputs/sci-reasoning-assistant/adapter --share   # Gradio
```

```python
from sciassist.assistant import ScientificReasoningAssistant, load_model
model, tok = load_model("outputs/sci-reasoning-assistant/adapter")
a = ScientificReasoningAssistant(model, tok)
a.load_paper(pdf_path="paper.pdf")           # or text="..."
print(a.chat("What are the main limitations of this study?"))
print(a.chat("What experiments would strengthen these findings?"))
```

**Training data.** The examples used in the original notebook are not part of this
repository. Put them, or your own, in `data/examples.json` in the format described in
`data/README.md` (`data/examples.template.json` shows the structure).

**What was run here.** No GPU or Hugging Face download was available while this
repository was prepared, so TinyLlama has not been trained from this code yet. What was
run: the unit tests, and the training and generation code paths end to end on CPU with a
tiny randomly initialised Llama model and a tokenizer trained on placeholder text:

```bash
python -m sciassist.train --examples examples_placeholder.json --smoke-test --max-steps 3
```

```text
INFO sciassist.train: examples: 6 train, 2 validation
trainable params: 16,384 || all params: 162,624 || trainable%: 10.0748
{'loss': 6.3224, ..., 'epoch': 0.03}
...
INFO sciassist.train: adapter saved to /tmp/smoke_out/adapter
```

These numbers only show that the pipeline runs; they say nothing about model quality.

## Results

To be filled after training on a GPU with the full example set. Held-out examples,
greedy decoding:

| Metric | Base TinyLlama-Chat | Fine-tuned adapter |
|---|---|---|
| ROUGE-L F1 vs reference | - | - |
| Echoes paper context (rate) | - | - |
| Runs on into new turns (rate) | - | - |
| Structured: JSON valid without repair | - | - |
| Structured: schema valid after repair | - | - |
| Structured: evidence spans found in passage | - | - |

## Inputs and outputs

| Item | Format |
|---|---|
| Training examples | `data/examples.json`: groups `extraction`, `qa`, `reasoning`, `structured`; each item `paper_context`, `instruction`, `response` |
| Paper for inference | plain text or PDF (PyMuPDF) |
| Adapter | `outputs/sci-reasoning-assistant/adapter/` (LoRA weights + tokenizer) |
| Split | `outputs/.../split.json` (example ids used for training and validation) |
| Evaluation | `outputs/eval/metrics.json`, `metrics.md`, `predictions.jsonl` |
| Session export | Markdown (`ScientificReasoningAssistant.export_markdown`) |

Structured-mode schema (`structured.ReasoningPack`):
`claims: list[str] (>= 1)`, `assumptions: list[str]`, `limitations: list[str]`,
`confidence: float in [0, 1]`, `evidence_spans: list[str]`.

## Configuration

`configs/train_tinyllama_qlora.yaml` holds the model, quantisation, LoRA, data
(`examples`, `val_fraction`, `replicate`) and training settings. Training arguments are
passed to `trl.SFTConfig`; arguments renamed between library versions
(`max_seq_length`/`max_length`, `eval_strategy`/`evaluation_strategy`) are mapped
automatically.

## Testing

```bash
pip install -r requirements-cpu.txt && pip install -e .
pytest      # 15 tests, CPU only
```

Tests cover the prompt format, response cleaning, PDF-free document processing and
chunking, example loading, validation and passage-level splitting, label masking,
JSON extraction and repair, the reasoning-pack schema and grounding check, and the metrics
and evaluation CLI. CI runs them on every push; GPU training is not run in CI.

## Limitations

- TinyLlama-1.1B follows instructions weakly; answers can be generic, wrong or not
  grounded in the passage, and must be checked against the paper.
- With a small dataset the adapter mostly learns the answer format; it does not add
  domain knowledge.
- The context is limited to about 3,000 characters of the paper (abstract first); long
  papers are truncated, not summarised.
- ROUGE-L against one reference answer is a weak measure of reasoning quality; expert
  review of a sample of answers is needed for a real assessment.

## License

MIT, see `LICENSE`. TinyLlama is distributed under Apache-2.0 by its authors.

## Author

Author: Sourabh Kumar · bioinfosourabh@gmail.com · https://bioinfosourabh.netlify.app · ORCID https://orcid.org/0009-0009-5840-9830
