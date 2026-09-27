from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import Settings, get_settings
from src.documents import SUPPORTED_EXTENSIONS, extract_text, split_text


class KnowledgeBase:
    def __init__(self, settings: Settings | None = None, collection: Collection | None = None):
        self.settings = settings or get_settings()
        if collection is not None:
            self.collection = collection
            self.embedder = None
            return
        if not self.settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings")
        client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self.embedder = GoogleGenerativeAIEmbeddings(
            model=self.settings.embedding_model,
            google_api_key=self.settings.google_api_key,
        )
        self.collection = client.get_or_create_collection(
            "company_knowledge_gemini",
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, path: Path) -> int:
        pages = extract_text(path)
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        for page in pages:
            for index, chunk in enumerate(split_text(page.text)):
                chunk_id = f"{file_hash}:{page.page or 0}:{index}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadata: dict[str, Any] = {"source": path.name, "file_hash": file_hash}
                if page.page is not None:
                    metadata["page"] = page.page
                metadatas.append(metadata)
        if documents:
            if self.embedder is None:
                raise RuntimeError("An embedding model is required to ingest documents")
            embeddings = self.embedder.embed_documents(documents)
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        return len(documents)

    def ingest_directory(self, directory: Path) -> int:
        """Index every supported document under a directory."""
        if not directory.exists():
            return 0
        total = 0
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                total += self.ingest(path)
        return total

    def ensure_seed_documents(self, directory: Path) -> int:
        """Seed an empty deployment database from bundled company documents."""
        if self.collection.count() > 0:
            return 0
        return self.ingest_directory(directory)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query.strip() or self.collection.count() == 0:
            return []
        if self.embedder is None:
            raise RuntimeError("An embedding model is required to search documents")
        query_embedding = self.embedder.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k or self.settings.rag_top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        matches = []
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            relevance = 1.0 - float(distance)
            if relevance >= self.settings.rag_min_relevance:
                matches.append({"text": text, "metadata": metadata, "relevance": relevance})
        return matches
