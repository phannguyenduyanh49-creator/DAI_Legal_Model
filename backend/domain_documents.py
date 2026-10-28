"""Domain-scoped plain-text document ingestion into isolated Qdrant collections."""

from __future__ import annotations

import re
import uuid
from pathlib import PurePath
from typing import Any, Callable

from qdrant_client.http import models

from backend.domain_registry import DomainRegistryService, registry_service


MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
ALLOWED_EXTENSIONS = {".txt", ".md"}


class DomainDocumentError(ValueError):
    pass


def domain_document_collection(domain_id: str) -> str:
    normalized = str(domain_id or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", normalized):
        raise DomainDocumentError("Invalid domain identifier")
    return f"domain_documents_{normalized}"


def retrieval_collections_for_domain(
    domain_id: str,
    collection_exists: Callable[[str], bool],
    registry: DomainRegistryService = registry_service,
) -> list[str]:
    config = registry.get_domain_config(domain_id)
    if not config.enabled:
        return []
    candidates = []
    if config.knowledge_collection:
        candidates.append(config.knowledge_collection)
    candidates.append(domain_document_collection(config.domain_id))
    return [
        collection
        for collection in dict.fromkeys(candidates)
        if collection_exists(collection)
    ]


def chunk_document(text: str, *, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    """Split text by approximate character budget while preserving offsets."""
    size = max(256, min(int(chunk_size) * 4, 8192))
    overlap = max(0, min(int(chunk_overlap) * 4, size // 2))
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary < 0:
                boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        value = text[start:end].strip()
        if value:
            chunks.append((start, end, value))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def decode_document(filename: str, content: bytes) -> tuple[str, str]:
    safe_name = PurePath(str(filename or "")).name.strip()
    extension = PurePath(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise DomainDocumentError("Only UTF-8 .txt and .md documents are supported")
    if not content:
        raise DomainDocumentError("Document is empty")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise DomainDocumentError("Document exceeds the 2 MiB limit")
    try:
        text = content.decode("utf-8-sig").replace("\x00", "").strip()
    except UnicodeDecodeError as exc:
        raise DomainDocumentError("Document must be valid UTF-8") from exc
    if not text:
        raise DomainDocumentError("Document contains no usable text")
    if len(text) > MAX_DOCUMENT_BYTES:
        raise DomainDocumentError("Document exceeds the text limit")
    return safe_name, text


class DomainDocumentService:
    def __init__(self, qdrant: Any, embedder: Any, registry: DomainRegistryService = registry_service):
        self.qdrant = qdrant
        self.embedder = embedder
        self.registry = registry

    def upload(self, domain_id: str, filename: str, content: bytes) -> dict[str, Any]:
        try:
            config = self.registry.get_domain_config(domain_id)
        except KeyError as exc:
            raise DomainDocumentError("Unknown domain") from exc
        if not config.enabled:
            raise DomainDocumentError("Domain is disabled")
        title, text = decode_document(filename, content)
        chunk_size = config.chunking_strategy.chunk_size
        chunk_overlap = config.chunking_strategy.chunk_overlap
        chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        collection = domain_document_collection(config.domain_id)
        vectors = self.embedder.encode(
            [chunk for _, _, chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if len(vectors) != len(chunks):
            raise RuntimeError("Document embedding returned an incomplete vector set")

        if not self.qdrant.collection_exists(collection):
            self.qdrant.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=int(self.embedder.get_sentence_embedding_dimension()),
                    distance=models.Distance.COSINE,
                ),
            )

        document_id = str(uuid.uuid4())
        points = []
        for index, ((start, end, chunk), vector) in enumerate(zip(chunks, vectors)):
            evidence_id = f"domain:{config.domain_id}:{document_id}:{index + 1}"
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                payload={
                    "domain_id": config.domain_id,
                    "document_id": document_id,
                    "evidence_id": evidence_id,
                    "source_type": "domain_upload",
                    "source_id": document_id,
                    "title": title,
                    "procedure_name": title,
                    "location": f"Characters {start + 1}-{end}",
                    "chunk_index": index + 1,
                    "text": chunk,
                    "full_text": chunk,
                },
            ))
        self.qdrant.upsert(collection_name=collection, points=points, wait=True)
        return {
            "status": "indexed",
            "domain_id": config.domain_id,
            "document_id": document_id,
            "title": title,
            "chunk_count": len(points),
        }
