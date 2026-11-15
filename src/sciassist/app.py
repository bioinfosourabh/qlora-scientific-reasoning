"""Gradio interface for the fine-tuned assistant (needs a GPU and a trained adapter).

    python -m sciassist.app --adapter outputs/sci-reasoning-assistant/adapter [--share]
"""

from __future__ import annotations

import argparse
import json


def build(assistant):
    import gradio as gr

    from .structured import parse_pack

    def load(text, pdf):
        paper = assistant.load_paper(text=text, pdf_path=pdf.name if pdf is not None else None)
        return f"Loaded: {paper.title} ({len(paper.full_text)} characters)", []

    def ask(question, chat):
        answer = assistant.chat(question)
        return "", chat + [(question, answer)]

    def pack():
        pk, pr, errors = parse_pack(assistant.reasoning_pack())
        return json.dumps(pk.model_dump() if pk else pr.obj, indent=2), "\n".join(errors) or "valid"

    with gr.Blocks(title="Scientific Reasoning Assistant") as demo:
        gr.Markdown("## Scientific Reasoning Assistant (TinyLlama + QLoRA)\n"
                    "Answers are model generations and can be wrong; check them against the paper.")
        with gr.Row():
            text = gr.Textbox(label="Paste abstract or text", lines=8)
            pdf = gr.File(label="or upload a PDF", file_types=[".pdf"])
        status = gr.Markdown()
        chat = gr.Chatbot(label="Conversation")
        q = gr.Textbox(label="Question", placeholder="What are the main limitations of this study?")
        with gr.Row():
            load_btn, pack_btn, reset_btn = gr.Button("Load paper"), gr.Button("Reasoning pack (JSON)"), gr.Button("Reset")
        pack_out, pack_err = gr.Code(label="Reasoning pack", language="json"), gr.Textbox(label="Validation")
        load_btn.click(load, [text, pdf], [status, chat])
        q.submit(ask, [q, chat], [q, chat])
        pack_btn.click(pack, None, [pack_out, pack_err])
        reset_btn.click(lambda: (assistant.reset(), [])[1], None, chat)
    return demo


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Gradio app")
    p.add_argument("--adapter", required=True)
    p.add_argument("--share", action="store_true")
    a = p.parse_args(argv)
    from .assistant import ScientificReasoningAssistant, load_model

    model, tok = load_model(a.adapter)
    build(ScientificReasoningAssistant(model, tok)).launch(share=a.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
