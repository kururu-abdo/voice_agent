from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import get_settings
from src.documents import SUPPORTED_EXTENSIONS
from src.graph import RAGService
from src.vector_store import KnowledgeBase

app = FastAPI(title="Company RAG API", version="0.1.0")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)
    return cleaned or "document"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    settings = get_settings()
    filename = safe_filename(file.filename or "document")
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    destination = settings.upload_path / f"{uuid4().hex}-{filename}"
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    destination.write_bytes(data)
    try:
        chunks = KnowledgeBase(settings).ingest(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {"filename": filename, "chunks_indexed": chunks}


@app.post("/ask")
async def ask(payload: Question) -> dict[str, object]:
    result = await RAGService().ask(payload.question)
    return {"answer": result["answer"], "sources": result.get("sources", [])}
