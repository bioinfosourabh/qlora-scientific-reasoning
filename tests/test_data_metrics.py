import json

from sciassist.data import formatted_texts, load_examples, split, tokenize_example, validate
from sciassist.jsonrepair import parse_model_output
from sciassist.metrics import aggregate, echoes_context, rouge_l, runs_on, score
from sciassist.prompts import STRUCTURED_SYSTEM_PROMPT
from sciassist.structured import grounding, parse_pack


class FakeTokenizer:
    """Whitespace tokenizer with BOS/EOS, enough to test label masking."""
    bos_token_id, eos_token_id = 1, 2

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 10) for w in text.split()]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}


def test_load_validate_split(examples_file):
    ex = load_examples(examples_file)
    assert len(ex) == 5 and {e["task"] for e in ex} == {"extraction", "qa", "reasoning", "structured"}
    assert validate(ex) == []
    train, val = split(ex, val_fraction=0.25, seed=1)
    tr_ctx = {e["paper_context"] for e in train}
    assert val and not tr_ctx & {e["paper_context"] for e in val}     # no passage in both
    assert len(formatted_texts(train, replicate=3)) == 3 * len(train)
    structured = [t for t in formatted_texts(ex) if STRUCTURED_SYSTEM_PROMPT in t]
    assert len(structured) == 1


def test_validate_catches_bad_examples(examples_file, tmp_path):
    data = json.loads(examples_file.read_text())
    data["structured"][0]["response"] = '{"claims": [], "confidence": 2}'
    data["qa"][0]["response"] = " "
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    problems = validate(load_examples(bad))
    assert any("empty response" in p for p in problems)
    assert any("not a valid reasoning pack" in p for p in problems)


def test_tokenize_masks_prompt(examples_file):
    ex = load_examples(examples_file)[0]
    tok = tokenize_example(ex, FakeTokenizer(), max_length=10_000)
    n_resp = len(ex["response"].split()) + 1
    assert tok["labels"][-n_resp:] == tok["input_ids"][-n_resp:]
    assert all(l == -100 for l in tok["labels"][:-n_resp])
    assert tok["input_ids"][-1] == FakeTokenizer.eos_token_id
    assert tokenize_example(ex, FakeTokenizer(), max_length=5) is None


def test_json_repair_and_pack():
    raw = 'Here you go:\n```json\n{"claims": ["A rises."], "assumptions": [], "limitations": ["n=1",],' \
          ' "confidence": 0.6, "evidence_spans": ["A rises"'
    pack, pr, errors = parse_pack(raw)
    assert pack is not None and pr.status == "repaired" and not errors
    assert parse_model_output('{"a": True}').obj == {"a": True}
    assert parse_model_output("no json").status == "failed"
    assert parse_pack('{"claims": ["x"], "confidence": 1.5}')[0] is None
    assert grounding(["alpha VALUE increased", "not there"], "The alpha value increased.") == 0.5


def test_metrics(examples_file):
    assert rouge_l("the cat sat", "the cat sat") == 1.0 and rouge_l("", "x") == 0.0
    ctx = " ".join(f"w{i}" for i in range(30))
    assert echoes_context("prefix " + " ".join(f"w{i}" for i in range(5, 20)), ctx)
    assert not echoes_context("short answer", ctx)
    assert runs_on("ok <|assistant|> more") and not runs_on("ok")
    ex = load_examples(examples_file)
    s = score("structured", ex[-1]["response"], ex[-1]["response"], ex[-1]["paper_context"])
    assert s["schema_valid"] == 1.0 and s["grounding"] == 1.0 and s["rougeL"] == 1.0
    f = score("qa", "One dataset.<|user|> again", "One dataset.", "ctx")
    assert f["rougeL"] == 1.0 and f["runs_on"] == 1.0
    agg = aggregate([s, f])
    assert agg["n"] == 2 and agg["runs_on"] == 0.5 and agg["schema_valid"] == 1.0
