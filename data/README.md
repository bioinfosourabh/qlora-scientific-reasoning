# Training data

`examples.json` holds the 18 training examples from the original notebook
(`EXTRACTION_EXAMPLES`: 5, `QA_EXAMPLES`: 6, `REASONING_EXAMPLES`: 7). Each `paper_context` is an
invented study summary written for training, not text from a real paper. The first runs showed
that about 30 or more distinct examples are needed (see below), so add your own in this format:

```json
{
  "extraction": [{"paper_context": "...", "instruction": "What are the main limitations of this study?", "response": "..."}],
  "qa":         [{"paper_context": "...", "instruction": "Is the sample size adequate?", "response": "..."}],
  "reasoning":  [{"paper_context": "...", "instruction": "What experiments would strengthen these findings?", "response": "..."}],
  "structured": [{"paper_context": "...", "instruction": "Extract the structured reasoning for this text.",
                  "response": "{\"claims\": [...], \"assumptions\": [...], \"limitations\": [...], \"confidence\": 0.7, \"evidence_spans\": [...]}"}]
}
```

* `extraction`, `qa`, `reasoning`: free-form answers, trained with the assistant system prompt.
* `structured`: `response` must be a JSON reasoning pack (validated by `sciassist.data.validate`);
  `evidence_spans` should be copied verbatim from `paper_context`.
* Each example may carry an `id`; otherwise one is generated.

Guidance from the first training runs:

* Use only text you are allowed to reuse (your own summaries, open-licence abstracts).
  Label invented passages as such.
* 30 or more distinct examples were needed before TinyLlama answered the question
  instead of continuing the paper text; vary instructions, not only passages.
* Keep responses short enough that prompt + response fits in `max_seq_length` (1024
  tokens); longer examples are skipped.
* Check the file with `python -c "from sciassist.data import load_examples, validate; print(validate(load_examples('data/examples.json')))"`.

`examples.template.json` shows the structure with placeholder text.
