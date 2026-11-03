"""PostgreSQL source-of-truth repository and transactional indexing outbox."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import create_engine, text


def create_search_database_engine():
    """Use an explicitly configured URL; never inherit Catalog's legacy default."""
    database_url = os.getenv("SEARCH_DATABASE_URL") or os.getenv("CATALOG_DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "An explicitly configured SEARCH_DATABASE_URL or CATALOG_DATABASE_URL is required for PostgreSQL search storage"
        )
    if not str(database_url).startswith(("postgresql://", "postgresql+psycopg2://")):
        raise RuntimeError("Search storage requires a PostgreSQL database URL")
    return create_engine(database_url, pool_pre_ping=True)


@dataclass(frozen=True)
class StoredDocument:
    document_key: str
    domain_id: str
    source_id: str
    document_id: str
    chunk_index: int
    title: str
    content: str
    source_type: str
    record_id: str | None
    metadata: dict[str, Any]

    @classmethod
    def from_payload(cls, document_key: str, payload: dict[str, Any]) -> "StoredDocument":
        domain_id = str(payload.get("domain_id") or "").strip().lower()
        source_id = str(payload.get("source_id") or "").strip()
        document_id = str(payload.get("document_id") or "").strip()
        # Search each retrieved chunk; retain any larger source text separately
        # in PostgreSQL metadata so re-indexing does not turn chunks into duplicates.
        content = str(payload.get("text") or payload.get("full_text") or "").strip()
        if not document_key or not domain_id or not source_id or not document_id or not content:
            raise ValueError(
                "Search documents require a key, domain, source ID, document ID, and nonempty content"
            )
        fields = payload.get("record_fields")
        record_id = payload.get("record_id")
        if not record_id and isinstance(fields, dict):
            record_id = fields.get("record_id")
        try:
            chunk_index = int(payload.get("chunk_index") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("chunk_index must be an integer") from exc
        return cls(
            document_key=document_key,
            domain_id=domain_id,
            source_id=source_id,
            document_id=document_id,
            chunk_index=chunk_index,
            title=str(payload.get("title") or payload.get("record_name") or "").strip(),
            content=content,
            source_type=str(payload.get("source_type") or "document"),
            record_id=str(record_id) if record_id else None,
            metadata={
                key: value for key, value in payload.items()
                if key not in {
                    "domain_id", "source_id", "document_id", "chunk_index", "title",
                    "text", "source_type", "record_id",
                }
            },
        )

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload.update({
            "domain_id": self.domain_id,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "text": self.content,
            "full_text": str(self.metadata.get("full_text") or self.content),
            "source_type": self.source_type,
        })
        if self.record_id:
            payload["record_id"] = self.record_id
        return payload


class PostgresDocumentRepository:
    """Persist documents and pending index events in one PostgreSQL transaction."""

    def __init__(self, engine=None):
        self.engine = engine if engine is not None else create_search_database_engine()

    def health(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def store_and_queue(self, documents: Iterable[StoredDocument]) -> int:
        rows = list(documents)
        if not rows:
            return 0
        upsert = text("""
            INSERT INTO domain_search_documents
                (document_key, domain_id, source_id, document_id, chunk_index,
                 title, content, source_type, record_id, metadata, updated_at)
            VALUES
                (:document_key, :domain_id, :source_id, :document_id, :chunk_index,
                 :title, :content, :source_type, :record_id, CAST(:metadata AS JSONB), NOW())
            ON CONFLICT (document_key) DO UPDATE SET
                domain_id = EXCLUDED.domain_id,
                source_id = EXCLUDED.source_id,
                document_id = EXCLUDED.document_id,
                chunk_index = EXCLUDED.chunk_index,
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                source_type = EXCLUDED.source_type,
                record_id = EXCLUDED.record_id,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """)
        queue = text("""
            INSERT INTO domain_search_outbox (document_key, event_type, created_at)
            VALUES (:document_key, 'index', NOW())
            ON CONFLICT (document_key) WHERE processed_at IS NULL DO NOTHING
        """)
        with self.engine.begin() as connection:
            for document in rows:
                values = {
                    "document_key": document.document_key,
                    "domain_id": document.domain_id,
                    "source_id": document.source_id,
                    "document_id": document.document_id,
                    "chunk_index": document.chunk_index,
                    "title": document.title,
                    "content": document.content,
                    "source_type": document.source_type,
                    "record_id": document.record_id,
                    "metadata": json.dumps(document.metadata, ensure_ascii=False),
                }
                connection.execute(upsert, values)
                connection.execute(queue, {"document_key": document.document_key})
        return len(rows)

    def pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        statement = text("""
            SELECT o.id AS event_id, d.document_key, d.domain_id, d.source_id,
                   d.document_id, d.chunk_index, d.title, d.content, d.source_type,
                   d.record_id, d.metadata
            FROM domain_search_outbox o
            JOIN domain_search_documents d USING (document_key)
            WHERE o.processed_at IS NULL AND o.available_at <= NOW()
            ORDER BY o.id
            LIMIT :limit
        """)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement, {"limit": limit}).mappings()]

    def mark_indexed(self, event_id: int) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("""
                UPDATE domain_search_outbox
                SET processed_at = NOW(), last_error = NULL
                WHERE id = :event_id
            """), {"event_id": event_id})

    def mark_failed(self, event_id: int, message: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("""
                UPDATE domain_search_outbox
                SET attempts = attempts + 1,
                    last_error = :message,
                    available_at = NOW() + LEAST(POWER(2, attempts), 300) * INTERVAL '1 second'
                WHERE id = :event_id
            """), {"event_id": event_id, "message": message[:1000]})

    def get_by_keys(self, document_keys: list[str]) -> dict[str, StoredDocument]:
        if not document_keys:
            return {}
        statement = text("""
            SELECT document_key, domain_id, source_id, document_id, chunk_index,
                   title, content, source_type, record_id, metadata
            FROM domain_search_documents WHERE document_key = ANY(:keys)
        """)
        with self.engine.connect() as connection:
            rows = connection.execute(statement, {"keys": document_keys}).mappings()
            return {
                row["document_key"]: StoredDocument(
                    document_key=row["document_key"], domain_id=row["domain_id"],
                    source_id=row["source_id"], document_id=row["document_id"],
                    chunk_index=row["chunk_index"], title=row["title"] or "",
                    content=row["content"], source_type=row["source_type"],
                    record_id=row["record_id"], metadata=dict(row["metadata"] or {}),
                )
                for row in rows
            }
