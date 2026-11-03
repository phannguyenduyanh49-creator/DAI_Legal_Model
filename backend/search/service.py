"""Document ingest, outbox synchronization, and domain-isolated search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.search.elasticsearch import ElasticsearchClient
from backend.search.repository import PostgresDocumentRepository, StoredDocument


class DocumentIndexingPending(RuntimeError):
    """PostgreSQL committed the document, but Elasticsearch has not confirmed it."""


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class DomainSearchService:
    def __init__(self, repository=None, elasticsearch=None):
        self.repository = repository or PostgresDocumentRepository()
        self.elasticsearch = elasticsearch or ElasticsearchClient()

    def index_documents(self, rows: list[tuple[str, dict[str, Any]]]) -> int:
        documents = [StoredDocument.from_payload(key, payload) for key, payload in rows]
        self.repository.store_and_queue(documents)
        indexed = self.sync_pending(max(100, len(documents)))
        missing = [doc.document_key for doc in documents if doc.document_key not in indexed]
        if missing:
            raise DocumentIndexingPending(
                f"PostgreSQL stored {len(documents)} document chunks; Elasticsearch indexing is pending for {len(missing)}."
            )
        return len(documents)

    def sync_pending(self, limit: int = 100) -> set[str]:
        pending = self.repository.pending_outbox(limit)
        if not pending:
            return set()
        request_docs = [{
            "document_key": row["document_key"],
            "domain_id": row["domain_id"],
            "source_id": row["source_id"],
            "document_id": row["document_id"],
            "record_id": row.get("record_id"),
            "chunk_index": row["chunk_index"],
            "title": row["title"] or "",
            "content": row["content"],
            "source_type": row["source_type"],
            "metadata": {
                key: value for key, value in (row.get("metadata") or {}).items()
                # PostgreSQL retains the complete source payload and hydrates
                # search hits from it. Keep verbose source fields out of the
                # flattened ES metadata mapping; long record_fields values can
                # exceed Lucene's term limits and reject an otherwise valid
                # document. The searchable content is already indexed above.
                if key not in {"full_text", "text", "content", "record_fields"}
            },
        } for row in pending]
        try:
            self.elasticsearch.ensure_index()
            outcomes = self.elasticsearch.bulk_index(request_docs)
            if len(outcomes) != len(pending):
                raise RuntimeError("Elasticsearch returned an incomplete bulk result")
        except Exception as exc:
            for event in pending:
                self.repository.mark_failed(event["event_id"], type(exc).__name__)
                print(
                    "[SEARCH OUTBOX]",
                    "status=failed",
                    "event_id=", event["event_id"],
                    "exception_type=", type(exc).__name__,
                    flush=True,
                )
            return set()
        indexed = set()
        for event, success in zip(pending, outcomes):
            if success:
                self.repository.mark_indexed(event["event_id"])
                indexed.add(event["document_key"])
            else:
                self.repository.mark_failed(event["event_id"], "elasticsearch_item_error")
                print(
                    "[SEARCH OUTBOX]",
                    "status=failed",
                    "event_id=", event["event_id"],
                    "reason=elasticsearch_item_error",
                    flush=True,
                )
        return indexed

    def search(
        self,
        query: str,
        domain_id: str,
        *,
        limit: int = 10,
        min_score: float | None = None,
        offset: int = 0,
        metadata_filters: dict[str, str | int | bool] | None = None,
    ) -> list[SearchHit]:
        if not self.elasticsearch.health():
            raise RuntimeError("Elasticsearch is unavailable")
        hits = self.elasticsearch.search(
            query,
            domain_id,
            limit=limit,
            min_score=min_score,
            offset=offset,
            metadata_filters=metadata_filters,
        )
        keys = [str(hit.get("_id") or "") for hit in hits]
        stored = self.repository.get_by_keys(keys)
        results = []
        for hit in hits:
            key = str(hit.get("_id") or "")
            document = stored.get(key)
            if document is None:
                continue
            if document.domain_id != domain_id:
                continue
            results.append(SearchHit(
                id=key,
                score=float(hit.get("_score") or 0.0),
                payload=document.to_payload(),
            ))
        return results

    def health(self) -> dict[str, str]:
        return {
            "postgres": "available" if self.repository.health() else "unavailable",
            "elasticsearch": "available" if self.elasticsearch.health() else "unavailable",
        }
