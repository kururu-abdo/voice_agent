from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass(frozen=True)
class TextPage:
    text: str
    page: int | None = None


def extract_text(path: Path) -> list[TextPage]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        pages = []
        for index, page in enumerate(PdfReader(path).pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(TextPage(text=text, page=index + 1))
        return pages
    if suffix == ".docx":
        text = "\n".join(p.text for p in DocxDocument(path).paragraphs if p.text.strip())
        return [TextPage(text=text)] if text.strip() else []

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [TextPage(text=text)] if text else []


def split_text(text: str, chunk_size: int = 1_200, overlap: int = 180) -> list[str]:
    """Split on paragraph boundaries, falling back to hard limits."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        remaining = paragraph
        while remaining:
            room = chunk_size - len(current) - (2 if current else 0)
            if room <= 0:
                chunks.append(current)
                current = current[-overlap:]
                continue
            piece, remaining = remaining[:room], remaining[room:]
            current = f"{current}\n\n{piece}" if current else piece
            if len(current) >= chunk_size:
                chunks.append(current)
                current = current[-overlap:] if remaining else ""
    if current.strip():
        chunks.append(current.strip())
    return chunks
