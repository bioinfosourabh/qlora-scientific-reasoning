"""Paper ingestion: PDF/text extraction, abstract detection, sections and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

SECTION_PATTERNS = [
    r"(?i)^\s*(abstract)[:\s]*$",
    r"(?i)^\s*(introduction)[:\s]*$",
    r"(?i)^\s*(background)[:\s]*$",
    r"(?i)^\s*(methods?|materials?\s*(and|&)\s*methods?)[:\s]*$",
    r"(?i)^\s*(results?)[:\s]*$",
    r"(?i)^\s*(discussion)[:\s]*$",
    r"(?i)^\s*(conclusions?)[:\s]*$",
    r"(?i)^\s*(references?|bibliography)[:\s]*$",
]


@dataclass
class ProcessedPaper:
    title: str = ""
    full_text: str = ""
    abstract: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_context(self, max_chars: int = 3000) -> str:
        """Context for the model, abstract first, truncated to ``max_chars``."""
        if self.abstract:
            context = f"ABSTRACT:\n{self.abstract}\n\n"
            remaining = max_chars - len(context)
            if remaining > 500 and self.full_text:
                context += f"PAPER CONTENT:\n{self.full_text[:remaining]}"
            return context[:max_chars]
        return self.full_text[:max_chars]


def extract_abstract(text: str, max_chars: int = 2500) -> str:
    m = re.search(r"(?is)\babstract\b[:\s]*\n?(.+?)(?:\n\s*\n|\n\s*(?:1\.?\s*)?introduction\b|\bkeywords?\b)",
                  text)
    return m.group(1).strip()[:max_chars] if m else ""


def split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {}
    current = None
    for line in text.splitlines():
        header = next((re.match(p, line) for p in SECTION_PATTERNS if re.match(p, line)), None)
        if header:
            current = header.group(1).lower()
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def create_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Fixed-size character chunks with overlap, broken at sentence ends where possible."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            stop = text.rfind(". ", start + chunk_size // 2, end)
            if stop != -1:
                end = stop + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def from_text(text: str, title: str = "Uploaded paper") -> ProcessedPaper:
    return ProcessedPaper(title=title, full_text=text.strip(), abstract=extract_abstract(text),
                          sections=split_sections(text), chunks=create_chunks(text),
                          metadata={"source": "text_input"})


def from_pdf(pdf_path: str) -> ProcessedPaper:
    """Extract text with PyMuPDF; the title is the first span with a large font."""
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        full_text = "\n".join(page.get_text() for page in doc)
        title = ""
        for block in doc[0].get_text("dict")["blocks"] if len(doc) else []:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("size", 0) > 14 and span.get("text", "").strip():
                        title = title or span["text"].strip()
        pages = len(doc)
    paper = from_text(full_text, title or pdf_path)
    paper.metadata.update({"source": pdf_path, "pages": pages})
    return paper
