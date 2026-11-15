import pytest

from sciassist.documents import create_chunks, extract_abstract, from_text, split_sections
from sciassist.prompts import (ASSISTANT_SYSTEM_PROMPT, chat_prompt, clean_response,
                               format_conversation, training_prompt)


def test_format_conversation_matches_tinyllama_template():
    t = format_conversation("Some context.", "Question?", "Answer.")
    assert t.startswith("<|system|>\nYou are a Scientific Reasoning Assistant")
    assert "</s>\n<|user|>\nHere is a research paper excerpt:\n\n---\nSome context.\n---\n\nQuestion?</s>\n" in t
    assert t.endswith("<|assistant|>\nAnswer.</s>")
    assert t.startswith(training_prompt("Some context.", "Question?"))


def test_chat_prompt_keeps_last_three_turns():
    hist = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(5)]
    p = chat_prompt("Paper.", "new question", hist)
    assert "q1" not in p and "q2" in p and "q4" in p
    assert p.endswith("new question</s>\n<|assistant|>\n")
    assert "Previous conversation" not in chat_prompt("Paper.", "q")
    assert ASSISTANT_SYSTEM_PROMPT in p


@pytest.mark.parametrize("raw, expected", [
    ("Answer text.</s>", "Answer text."),
    ("First answer. <|assistant|> Certainly, more text", "First answer."),
    ("Answer\n<|user|>\nnext question", "Answer"),
    ("Answer. Paper context: copied text", "Answer."),
    ("<s> plain", "plain"),
])
def test_clean_response(raw, expected):
    assert clean_response(raw) == expected


def test_documents():
    text = ("Title line\nAbstract\nWe studied alpha in beta.\n\nIntroduction\nIntro text.\n"
            "Methods\nWe did things.\nResults\nAlpha rose.\n")
    assert extract_abstract(text) == "We studied alpha in beta."
    sec = split_sections(text)
    assert sec["methods"] == "We did things." and sec["results"] == "Alpha rose."
    paper = from_text(text)
    assert paper.get_context(60).startswith("ABSTRACT:\nWe studied alpha in beta.")
    assert len(paper.get_context(60)) <= 60


def test_chunking_overlap_and_coverage():
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    chunks = create_chunks(text, chunk_size=500, overlap=100)
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[0][:20] == text[:20] and chunks[-1][-20:] == text[-20:]
    assert chunks[1][:30] in chunks[0]            # consecutive chunks overlap
    with pytest.raises(ValueError):
        create_chunks(text, 100, 100)
