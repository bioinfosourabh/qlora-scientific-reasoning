"""Generate answers for the held-out examples and score them.

    python -m sciassist.evaluate --adapter outputs/sci-reasoning-assistant/adapter \\
        --examples data/examples.json --split outputs/sci-reasoning-assistant/split.json \\
        --out outputs/eval
    python -m sciassist.evaluate --predictions preds.jsonl --examples data/examples.json --out outputs/eval

``--predictions`` scores existing generations (JSONL with ``id`` and ``output``) without a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from .data import load_examples, system_prompt_for
from .metrics import aggregate, score
from .prompts import training_prompt

LABELS = [("rougeL", "ROUGE-L F1 vs reference"), ("echoes_context", "Echoes paper context"),
          ("runs_on", "Runs on into new turns"), ("empty", "Empty answer"),
          ("json_valid_raw", "JSON valid without repair"), ("json_parsed", "JSON parsed after repair"),
          ("schema_valid", "Schema valid"), ("field_coverage", "Field coverage"),
          ("grounding", "Evidence spans found in passage")]


def evaluate(examples: List[dict], outputs: Dict[str, str]) -> Dict[str, object]:
    rows = []
    for ex in examples:
        r = score(ex["task"], outputs[ex["id"]], ex["response"], ex["paper_context"])
        rows.append(dict(r, id=ex["id"], task=ex["task"]))
    summary = {"all": aggregate(rows)}
    for task in sorted({r["task"] for r in rows}):
        summary[task] = aggregate([r for r in rows if r["task"] == task])
    return {"summary": summary, "rows": rows}


def markdown(summary: Dict[str, Dict[str, Optional[float]]]) -> str:
    groups = list(summary)
    lines = ["| Metric | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|",
             "| n | " + " | ".join(str(int(summary[g]["n"])) for g in groups) + " |"]
    for key, label in LABELS:
        vals = [summary[g].get(key) for g in groups]
        if any(v is not None for v in vals):
            lines.append(f"| {label} | " + " | ".join("n/a" if v is None else f"{v:.3f}" for v in vals) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--examples", type=Path, required=True)
    p.add_argument("--split", type=Path, help="split.json written by training (uses 'val')")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter")
    src.add_argument("--predictions", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    examples = load_examples(a.examples)
    if a.split:
        val = set(json.loads(a.split.read_text())["val"])
        examples = [e for e in examples if e["id"] in val]
    if a.predictions:
        outputs = {}
        with open(a.predictions) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    outputs[r["id"]] = r["output"]
    else:
        from .assistant import generate, load_model

        model, tok = load_model(a.adapter)
        outputs = {e["id"]: generate(model, tok, training_prompt(e["paper_context"], e["instruction"],
                                                                 system_prompt_for(e)),
                                     temperature=0.0) for e in examples}
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "predictions.jsonl", "w") as fh:
        for k, v in outputs.items():
            fh.write(json.dumps({"id": k, "output": v}) + "\n")
    res = evaluate(examples, outputs)
    (a.out / "metrics.json").write_text(json.dumps(res["summary"], indent=2) + "\n")
    table = markdown(res["summary"])
    (a.out / "metrics.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
