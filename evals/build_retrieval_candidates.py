"""Create a source-linked, explicitly unreviewed retrieval annotation queue.

No labels are promoted to human gold by this script. It reads the migrated
PostgreSQL search store and emits one JSONL row per requested domain slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.search.repository import create_search_database_engine


SPECS = {
    "legal_ai": (20, None),
    "administrative_law": (20, None),
    "healthcare_facility": (10, "facility_service_mock"),
    "healthcare_knowledge": (10, "health_knowledge"),
    "education": (15, None),
    "tourism": (10, None),
    "shopping": (10, None),
    "food": (5, None),
}


def _fields(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    fields = metadata.get("record_fields") or {}
    return fields if isinstance(fields, dict) else {}


def _query_for(domain: str, record: dict[str, Any], record_type: str) -> tuple[str | None, str]:
    fields = _fields(record)
    if domain == "healthcare" and record_type == "HC-F":
        specialty = str(fields.get("chuyen_khoa") or "").strip()
        district = str(fields.get("quan_huyen") or "").strip()
        if specialty and district:
            return f"Tìm dịch vụ {specialty} tại {district}", "structured fields: chuyen_khoa, quan_huyen"
        return None, "missing structured specialty or district"
    if domain == "healthcare" and record_type == "HC-K":
        topic = str(fields.get("chu_de") or "").strip()
        return (f"Thông tin về {topic}", "structured field: chu_de") if topic else (None, "missing chu_de")
    if domain == "education":
        course = str(fields.get("ten_khoa_hoc") or "").strip()
        return (course, "structured field: ten_khoa_hoc") if course else (None, "missing ten_khoa_hoc")
    if domain == "shopping":
        name = str(fields.get("ten_san_pham") or "").strip()
        brand = str(fields.get("thuong_hieu") or "").strip()
        query = " ".join(value for value in (brand, name) if value)
        return (query, "structured fields: thuong_hieu, ten_san_pham") if query else (None, "missing product name")
    if domain in {"legal_ai", "administrative_law"}:
        title = str(record.get("title") or "").strip()
        return (title, "exact Catalog title") if title else (None, "missing Catalog title")
    if domain == "tourism":
        match = re.search(r"(?:^|\n)\s*Tiêu đề:\s*([^\r\n]+)", str(record.get("content") or ""), re.IGNORECASE)
        if match:
            return match.group(1).strip(), "exact source heading: Tiêu đề"
        return None, "no stable source heading; human must author query from the inspected record"
    return None, "no indexed source records for this domain"


def _record_type(domain: str, record: dict[str, Any]) -> str:
    fields = _fields(record)
    if domain == "healthcare":
        raw = str(fields.get("loai_ban_ghi") or "")
        return {"facility_service_mock": "HC-F", "health_knowledge": "HC-K"}.get(raw, "unknown")
    return str(record.get("source_type") or "unknown")


def _query_basis_fields(domain: str, record_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    if domain == "healthcare" and record_type == "HC-F":
        names = ("chuyen_khoa", "quan_huyen")
    elif domain == "healthcare" and record_type == "HC-K":
        names = ("chu_de",)
    elif domain == "education":
        names = ("ten_khoa_hoc",)
    elif domain == "shopping":
        names = ("thuong_hieu", "ten_san_pham")
    else:
        return {}
    return {name: fields[name] for name in names if name in fields}


def build_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_domain.setdefault(record["domain_id"], []).append(record)

    output: list[dict[str, Any]] = []
    used: set[str] = set()
    used_catalog_records: set[tuple[str, str]] = set()
    for bucket, (count, filter_type) in SPECS.items():
        domain = "healthcare" if bucket.startswith("healthcare_") else bucket
        candidates = by_domain.get(domain, [])
        if filter_type:
            candidates = [row for row in candidates if _record_type(domain, row) == ("HC-F" if filter_type == "facility_service_mock" else "HC-K")]
        candidates = [row for row in candidates if row["document_key"] not in used]
        if domain in {"legal_ai", "administrative_law"}:
            candidates = [row for row in candidates if (row["source_id"], row["document_id"]) not in used_catalog_records]
        # Prefer records from which a neutral query can be copied directly;
        # the fallback remains an explicit human-authored annotation slot.
        candidates.sort(key=lambda row: _query_for(domain, row, _record_type(domain, row))[0] is None)
        for index in range(count):
            row = candidates[index] if index < len(candidates) else None
            query = None
            query_basis = ""
            if row:
                used.add(row["document_key"])
                if domain in {"legal_ai", "administrative_law"}:
                    used_catalog_records.add((row["source_id"], row["document_id"]))
                record_type = _record_type(domain, row)
                query, query_basis = _query_for(domain, row, record_type)
            else:
                record_type = "HC-F" if filter_type == "facility_service_mock" else "HC-K" if filter_type == "health_knowledge" else "unknown"
                query_basis = "no matching indexed source record; requires data/annotation"

            source = None
            auto_suggestions = []
            if row:
                fields = _fields(row)
                metadata = row.get("metadata") or {}
                source = {
                    "document_key": row["document_key"],
                    "domain_id": row["domain_id"],
                    "document_id": row["document_id"],
                    "record_id": row.get("record_id") or fields.get("record_id"),
                    "source_id": row["source_id"],
                    "source_title": row.get("title"),
                    "source_url": metadata.get("source_url") or metadata.get("url_nguon"),
                    "source_type": row["source_type"],
                    "content_sha256": hashlib.sha256(str(row.get("content") or "").encode("utf-8")).hexdigest(),
                    "query_basis_fields": _query_basis_fields(domain, record_type, fields),
                }
                if query:
                    auto_suggestions.append({
                        "document_key": row["document_key"],
                        "record_id": source["record_id"],
                        "suggested_relevance": 2,
                        "suggestion_status": "automatic_unreviewed",
                        "basis": query_basis,
                    })
            if domain in {"legal_ai", "administrative_law"}:
                query_basis += "; domain assignment is provisional because both Registry domains share the same Catalog collection"

            identity_seed = f"candidate-v2|{bucket}|{index + 1}|{source['document_key'] if source else ''}|{query or ''}"
            query_id = "Q-" + hashlib.sha256(identity_seed.encode("utf-8")).hexdigest()[:12]
            output.append({
                "sample_id": query_id,
                "query_id": query_id,
                "coordinator_bucket": bucket,
                "query": query,
                "expected_domain": domain,
                "expected_domain_status": "proposed_from_source_record; requires human review",
                "expected_record_type": record_type,
                "expected_record_type_status": "proposed_from_source_metadata; requires human review",
                "relevant_document_ids": [],
                "relevant_record_ids": [],
                "relevance_labels": {},
                "relevant_evidence": {},
                "expected_evidence": [],
                "expected_answer_points": [],
                "expected_behavior": "requires human annotation",
                "annotation_status": "blocked_no_source_data" if not row else "needs_human_annotation",
                "human_approved": False,
                "review_status": "unreviewed",
                "reviewers": [],
                "adjudicator": None,
                "source_provenance": {"status": "candidate_only_not_verified", "references": []},
                "auto_candidate_judgments": auto_suggestions,
                "source_record_for_inspection": source,
                "annotation_notes": query_basis,
                "decision_options": ["relevant", "partially_relevant", "irrelevant", "insufficient_information"],
                "review": {"annotators": [], "adjudicator": None, "review_date": None},
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evals/golden_candidates.jsonl")
    args = parser.parse_args()
    engine = create_search_database_engine()
    try:
        with engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(text("""
                SELECT document_key, domain_id, source_id, document_id, record_id,
                       title, content, source_type, metadata
                FROM domain_search_documents
                WHERE domain_id = ANY(:domains)
                ORDER BY domain_id, record_id NULLS LAST, document_id
            """), {"domains": ["legal_ai", "administrative_law", "healthcare", "education", "tourism", "shopping", "food"]}).mappings()]
    finally:
        engine.dispose()

    candidates = build_rows(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates), encoding="utf-8")
    query_count = sum(bool(row["query"]) for row in candidates)
    status_counts: dict[str, int] = {}
    for row in candidates:
        status_counts[row["annotation_status"]] = status_counts.get(row["annotation_status"], 0) + 1
    print(f"candidate_slots={len(candidates)} query_candidates={query_count} human_approved=0")
    for status, count in sorted(status_counts.items()):
        print(f"{status}={count}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
