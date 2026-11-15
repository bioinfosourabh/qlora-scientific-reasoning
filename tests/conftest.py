import json

import pytest

# Neutral placeholder text: these tests check formatting and parsing logic only.
CTX = ["Alpha text one. The alpha value increased in group A. Only group A was measured.",
       "Beta text two. Method B was compared with method C on one dataset.",
       "Gamma text three. Gamma was observed in a single run without replication.",
       "Delta text four. Delta correlated with epsilon in the sample."]


@pytest.fixture
def examples_file(tmp_path):
    data = {
        "extraction": [{"paper_context": CTX[0], "instruction": "What are the main limitations?",
                        "response": "Only group A was measured, so the comparison is missing."}],
        "qa": [{"paper_context": CTX[1], "instruction": "How many datasets were used?",
                "response": "One dataset."},
               {"paper_context": CTX[1], "instruction": "What was compared?",
                "response": "Method B with method C."}],
        "reasoning": [{"paper_context": CTX[2], "instruction": "Is the result reliable?",
                       "response": "Not yet: it comes from a single run without replication."}],
        "structured": [{"paper_context": CTX[3], "instruction": "Extract the structured reasoning for this text.",
                        "response": json.dumps({"claims": ["Delta correlates with epsilon."],
                                                "assumptions": ["The sample is representative."],
                                                "limitations": ["Correlation only."], "confidence": 0.4,
                                                "evidence_spans": ["Delta correlated with epsilon"]})}],
    }
    path = tmp_path / "examples.json"
    path.write_text(json.dumps(data))
    return path
