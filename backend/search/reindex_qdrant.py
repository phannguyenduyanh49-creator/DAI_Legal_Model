"""Copy domain-tagged Qdrant payloads to PostgreSQL and Elasticsearch.

Dry-run is the default. This script never deletes or changes Qdrant data.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter

from qdrant_client import QdrantClient

from backend.domain_registry import registry_service
from backend.domain_documents import (
    DomainDocumentError,
    _is_instruction_sheet,
    _sheet_domain_from_headers,
    _sheet_domain_from_name,
)
from backend.search.service import DomainSearchService
from backend.search.repository import StoredDocument


def registered_collection_domains() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for domain_id in registry_service.list_enabled_domains():
        config = registry_service.get_domain_config(domain_id)
        if config.knowledge_collection:
            pairs.add((config.knowledge_collection, domain_id))
        pairs.add((f"domain_documents_{domain_id}", domain_id))
    return sorted(pairs)


def _expected_domains_by_collection() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for collection, domain_id in registered_collection_domains():
        result.setdefault(collection, set()).add(domain_id)
    return result


def collect_records(client: QdrantClient):
    counts: Counter[str] = Counter()
    records: list[StoredDocument] = []
    content_signatures: set[tuple[str, str]] = set()
    document_chunk_keys: set[tuple[str, str, str, int]] = set()
    for collection, expected_domains in _expected_domains_by_collection().items():
        try:
            if not client.collection_exists(collection):
                continue
        except Exception as exc:
            counts[f"collection_check_error:{type(exc).__name__}"] += 1
            continue
        offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                counts[f"scanned_collection:{collection}"] += 1
                sheet_name = str(payload.get("sheet_name") or "").strip()
                payload_domain = str(payload.get("domain_id") or "").strip().lower()
                counts[f"payload_domain:{payload_domain or 'missing'}"] += 1
                if sheet_name:
                    counts[f"worksheet:{sheet_name}"] += 1
                if sheet_name and _is_instruction_sheet(sheet_name):
                    counts["excluded_instruction_sheet"] += 1
                    continue

                record_fields = payload.get("record_fields")
                record_fields = record_fields if isinstance(record_fields, dict) else {}
                try:
                    name_domain = _sheet_domain_from_name(sheet_name, registry_service) if sheet_name else None
                    header_domain = _sheet_domain_from_headers(
                        [{"fields": record_fields}], registry_service
                    ) if record_fields else None
                except DomainDocumentError:
                    counts["ambiguous_worksheet_domain"] += 1
                    continue
                if name_domain and header_domain and name_domain != header_domain:
                    counts["worksheet_name_header_conflict"] += 1
                    continue

                inferred_domain = name_domain or header_domain
                # An explicit worksheet alias/schema is stronger than the old
                # collection tag for known mixed-workbook ingestion defects.
                target_domain = inferred_domain or payload_domain
                if not target_domain:
                    counts["missing_domain_metadata"] += 1
                    continue
                try:
                    target_config = registry_service.get_domain_config(target_domain)
                except KeyError:
                    counts["unregistered_domain"] += 1
                    continue
                if not target_config.enabled:
                    counts[f"disabled_domain:{target_domain}"] += 1
                    continue
                if inferred_domain and payload_domain and inferred_domain != payload_domain:
                    counts[f"remapped_domain:{payload_domain}:{inferred_domain}"] += 1
                    if payload.get("source_type") == "domain_upload":
                        document_id = str(payload.get("document_id") or "")
                        source_id = str(payload.get("source_id") or "")
                        chunk_index = payload.get("chunk_index")
                        old_evidence_id = payload.get("evidence_id")
                        if (
                            not document_id
                            or source_id != document_id
                            or not isinstance(chunk_index, int)
                            or chunk_index < 1
                            or old_evidence_id != (
                                f"domain:{payload_domain}:{document_id}:{chunk_index}"
                            )
                        ):
                            counts["unrepairable_upload_identity"] += 1
                            continue
                        payload["evidence_id"] = (
                            f"domain:{target_domain}:{document_id}:{chunk_index}"
                        )
                        counts[f"rewritten_evidence_id:{target_domain}"] += 1
                elif not inferred_domain and payload_domain not in expected_domains:
                    counts[f"out_of_scope_domain:{payload_domain or 'missing'}"] += 1
                    continue

                # Do not guess the domain for legacy knowledge collections whose
                # points lack domain_id and whose Registry collection is shared.
                if not inferred_domain and not payload_domain:
                    counts["missing_domain_metadata"] += 1
                    continue

                payload["domain_id"] = target_domain
                payload["record_id"] = payload.get("record_id") or record_fields.get("record_id")
                if not payload.get("source_id"):
                    counts["missing_source_id"] += 1
                    continue
                if not payload.get("document_id"):
                    counts["missing_document_id"] += 1
                    continue
                if not str(payload.get("text") or payload.get("full_text") or "").strip():
                    counts["missing_content"] += 1
                    continue
                try:
                    records.append(StoredDocument.from_payload(
                        f"{collection}:{point.id}", payload
                    ))
                except (TypeError, ValueError):
                    counts["missing_source_document_or_content"] += 1
                    continue
                counts[f"accepted:{target_domain}"] += 1
                text = str(payload.get("text") or payload.get("full_text") or "").strip()
                chunk_identity = (
                    target_domain,
                    records[-1].source_id,
                    records[-1].document_id,
                    records[-1].chunk_index,
                )
                if chunk_identity in document_chunk_keys:
                    counts[f"duplicate_document_chunk:{target_domain}"] += 1
                document_chunk_keys.add(chunk_identity)
                signature = hashlib.sha256(text.encode("utf-8")).hexdigest()
                signature_key = (target_domain, signature)
                if signature_key in content_signatures:
                    counts[f"duplicate_content:{target_domain}"] += 1
                content_signatures.add(signature_key)
            if next_offset is None:
                break
            offset = next_offset
    return records, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex Qdrant payloads without deleting source data")
    parser.add_argument("--apply", action="store_true", help="Persist to PostgreSQL and Elasticsearch")
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 1000:
        parser.error("--batch-size must be between 1 and 1000")

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    documents, counts = collect_records(client)
    print(f"Scanned eligible domain collections; validated records: {len(documents)}")
    for key, value in sorted(counts.items()):
        print(f"{key}={value}")
    if not args.apply:
        print("Dry run only. Review counts and payload provenance, then pass --apply to index.")
        return

    store = DomainSearchService()
    if not store.repository.health() or not store.elasticsearch.cluster_health():
        raise SystemExit("PostgreSQL and Elasticsearch must both be healthy before applying")
    store.elasticsearch.ensure_index()
    for start in range(0, len(documents), args.batch_size):
        batch = documents[start:start + args.batch_size]
        rows = [(doc.document_key, doc.to_payload()) for doc in batch]
        store.index_documents(rows)
        print(f"Indexed validated batch {start // args.batch_size + 1}: {len(batch)} records")
    print("Reindex complete; Qdrant source records were not modified or deleted.")


if __name__ == "__main__":
    main()
