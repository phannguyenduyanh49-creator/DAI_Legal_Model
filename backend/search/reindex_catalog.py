"""Dry-run or index existing PostgreSQL Catalog records into the search store."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.catalog.models import Procedure
from backend.domain_registry import registry_service
from backend.evidence.normalizer import procedure_to_evidence
from backend.search.repository import StoredDocument, create_search_database_engine
from backend.search.service import DomainSearchService


def catalog_domains() -> list[str]:
    """Use Registry collection ownership; do not hardcode legal domain IDs."""
    return [
        domain_id
        for domain_id in registry_service.list_enabled_domains()
        if (
            (config := registry_service.get_domain_config(domain_id)).enabled
            and config.catalog_enabled
            and config.knowledge_collection
        )
    ]


def catalog_record_for_domain(procedure: Procedure, domain_id: str) -> StoredDocument:
    evidence = procedure_to_evidence(procedure)
    payload = {
        "domain_id": domain_id,
        "source_id": evidence.source_id,
        "document_id": procedure.formality_id,
        "chunk_index": 0,
        "title": evidence.title,
        "text": evidence.content,
        "full_text": evidence.content,
        "source_type": evidence.source_type,
        "record_id": procedure.formality_id,
        "formality_id": procedure.formality_id,
        "evidence_id": evidence.evidence_id,
        "source_url": evidence.source_url,
        "source": procedure.source,
        "code": procedure.code,
        "category_name": procedure.category_name,
        "record_fields": evidence.fields,
        "completeness": evidence.completeness,
        "metadata": evidence.metadata,
    }
    return StoredDocument.from_payload(
        f"catalog:{domain_id}:{procedure.formality_id}", payload
    )


def collect_catalog_documents(engine=None) -> tuple[list[StoredDocument], Counter[str]]:
    engine = engine or create_search_database_engine()
    records: list[StoredDocument] = []
    counts: Counter[str] = Counter()
    domains = catalog_domains()
    statement = (
        select(Procedure)
        .where(Procedure.status == "active")
        .order_by(Procedure.formality_id)
    )
    with Session(engine) as session:
        for procedure in session.execute(statement).scalars().yield_per(500):
            for domain_id in domains:
                try:
                    records.append(catalog_record_for_domain(procedure, domain_id))
                except (TypeError, ValueError):
                    counts[f"invalid_catalog_record:{domain_id}"] += 1
                    continue
                counts[f"candidate:{domain_id}"] += 1
    return records, counts


def summarize(records: list[StoredDocument], counts: Counter[str]) -> None:
    content_by_domain: dict[str, set[str]] = {}
    duplicate_counts: Counter[str] = Counter()
    for record in records:
        digest = hashlib.sha256(record.content.encode("utf-8")).hexdigest()
        seen = content_by_domain.setdefault(record.domain_id, set())
        if digest in seen:
            duplicate_counts[record.domain_id] += 1
        seen.add(digest)
    counts.update({
        f"duplicate_content:{domain_id}": count
        for domain_id, count in duplicate_counts.items()
    })
    print(f"Validated PostgreSQL Catalog search records: {len(records)}")
    for key, value in sorted(counts.items()):
        print(f"{key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex the existing PostgreSQL Catalog")
    parser.add_argument("--apply", action="store_true", help="Persist Catalog records and index them")
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1000:
        parser.error("--batch-size must be between 1 and 1000")

    engine = create_search_database_engine()
    try:
        records, counts = collect_catalog_documents(engine)
        summarize(records, counts)
        if not args.apply:
            print("Dry run only. No PostgreSQL or Elasticsearch records were changed.")
            return

        store = DomainSearchService()
        if not store.repository.health() or not store.elasticsearch.cluster_health():
            raise SystemExit("PostgreSQL and Elasticsearch must both be healthy before applying")
        store.elasticsearch.ensure_index()
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            rows = [(record.document_key, record.to_payload()) for record in batch]
            store.index_documents(rows)
            print(f"Indexed Catalog batch {start // args.batch_size + 1}: {len(batch)} records")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
