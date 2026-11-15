import json

from sciassist.evaluate import main


def test_evaluate_from_predictions(examples_file, tmp_path):
    from sciassist.data import load_examples

    ex = load_examples(examples_file)
    preds = tmp_path / "p.jsonl"
    with open(preds, "w") as fh:
        for e in ex:
            out = e["response"] if e["task"] != "qa" else e["paper_context"] + " <|user|>"
            fh.write(json.dumps({"id": e["id"], "output": out}) + "\n")
    assert main(["--examples", str(examples_file), "--predictions", str(preds), "--out", str(tmp_path / "ev")]) == 0
    m = json.loads((tmp_path / "ev" / "metrics.json").read_text())
    assert m["qa"]["echoes_context"] == 0.0 or m["qa"]["runs_on"] == 1.0
    assert m["structured"]["schema_valid"] == 1.0
    assert "| ROUGE-L F1 vs reference |" in (tmp_path / "ev" / "metrics.md").read_text()
