"""Validate independent human reviews and export approved retrieval gold JSONL.

Candidate suggestions are never read as labels. Gold is emitted only when every
query candidate has two independent reviews, an approved adjudication, and
source IDs/content hashes that still match PostgreSQL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from backend.domain_registry import registry_service
from backend.search.repository import create_search_database_engine


LABEL_FOR_STATE = {"irrelevant": 0, "partially_relevant": 1, "relevant": 2}
QUERY_STATES = {"answerable", "insufficient_information", "invalid_query"}
REVIEW_STATES = set(LABEL_FOR_STATE) | {"insufficient_information"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
        rows.append(row)
    return rows


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def validate_judgment(judgment: dict[str, Any]) -> None:
    state = judgment.get("state")
    label = judgment.get("label")
    if state not in REVIEW_STATES:
        raise ValueError("judgment state must be relevant, partially_relevant, irrelevant, or insufficient_information")
    if state == "insufficient_information":
        if label is not None:
            raise ValueError("insufficient_information must use a null relevance label, not 0")
    elif type(label) is not int or label != LABEL_FOR_STATE[state]:
        raise ValueError(f"state {state!r} must use relevance label {LABEL_FOR_STATE.get(state)}")
    if not str(judgment.get("note") or "").strip():
        raise ValueError("each document judgment requires an explanatory note")
    _timestamp(judgment.get("source_checked_at"), "source_checked_at")
    for key in ("document_key", "domain_id", "document_id", "source_id", "content_sha256"):
        if not str(judgment.get(key) or "").strip():
            raise ValueError(f"judgment requires {key}")


def validate_review_records(
    candidates: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    query_candidates = {row["query_id"]: row for row in candidates if row.get("query")}
    reviews_by_query: dict[str, list[dict[str, Any]]] = {}
    for review in reviews:
        query_id = str(review.get("query_id") or "")
        if query_id not in query_candidates:
            raise ValueError(f"review references unknown/non-query candidate {query_id!r}")
        if review.get("review_status") != "submitted":
            raise ValueError(f"{query_id}: review_status must be submitted")
        annotator = str(review.get("annotator_id") or "").strip()
        if not annotator:
            raise ValueError(f"{query_id}: annotator_id is required")
        _timestamp(review.get("annotated_at"), "annotated_at")
        if review.get("query_status") not in QUERY_STATES:
            raise ValueError(f"{query_id}: invalid query_status")
        if not str(review.get("expected_domain") or "").strip():
            raise ValueError(f"{query_id}: independently assessed expected_domain is required")
        if review.get("expected_domain") not in registry_service.list_enabled_domains():
            raise ValueError(f"{query_id}: expected_domain is not an enabled Registry domain")
        judgments = review.get("document_judgments")
        if not isinstance(judgments, list) or not judgments:
            raise ValueError(f"{query_id}: document_judgments must be a nonempty list")
        seen_keys = set()
        for judgment in judgments:
            validate_judgment(judgment)
            if judgment["document_key"] in seen_keys:
                raise ValueError(f"{query_id}: duplicate document judgment")
            seen_keys.add(judgment["document_key"])
        if not str(review.get("notes") or "").strip():
            raise ValueError(f"{query_id}: review notes are required")
        if "auto_candidate_judgments" in review or "suggested_relevance" in review:
            raise ValueError("automatic candidate suggestions must not be copied into human review records")
        reviews_by_query.setdefault(query_id, []).append(review)

    adjudication_by_query: dict[str, dict[str, Any]] = {}
    for adjudication in adjudications:
        query_id = str(adjudication.get("query_id") or "")
        if query_id not in query_candidates:
            raise ValueError(f"adjudication references unknown/non-query candidate {query_id!r}")
        if query_id in adjudication_by_query:
            raise ValueError(f"{query_id}: multiple adjudications are not allowed")
        if adjudication.get("adjudication_status") not in {"approved", "disputed"}:
            raise ValueError(f"{query_id}: adjudication_status must be approved or disputed")
        if not str(adjudication.get("adjudicator_id") or "").strip():
            raise ValueError(f"{query_id}: adjudicator_id is required")
        _timestamp(adjudication.get("adjudicated_at"), "adjudicated_at")
        if adjudication.get("query_status") not in QUERY_STATES:
            raise ValueError(f"{query_id}: invalid adjudicated query_status")
        if not str(adjudication.get("expected_domain") or "").strip():
            raise ValueError(f"{query_id}: adjudicated expected_domain is required")
        if adjudication.get("expected_domain") not in registry_service.list_enabled_domains():
            raise ValueError(f"{query_id}: adjudicated domain is not enabled in Registry")
        if not str(adjudication.get("expected_record_type") or "").strip():
            raise ValueError(f"{query_id}: adjudicated expected_record_type is required")
        if not str(adjudication.get("notes") or "").strip():
            raise ValueError(f"{query_id}: adjudication notes are required")
        judgments = adjudication.get("document_judgments")
        if not isinstance(judgments, list) or not judgments:
            raise ValueError(f"{query_id}: adjudicated document_judgments must be nonempty")
        for judgment in judgments:
            validate_judgment(judgment)
        adjudication_by_query[query_id] = adjudication

    for query_id in query_candidates:
        query_reviews = reviews_by_query.get(query_id, [])
        annotators = [row["annotator_id"] for row in query_reviews]
        if len(annotators) != 2:
            raise ValueError(f"{query_id}: requires exactly two independent annotator reviews")
        if len(annotators) != len(set(annotators)):
            raise ValueError(f"{query_id}: an annotator submitted more than one review")
        if len({tuple(sorted(j["document_key"] for j in r["document_judgments"])) for r in query_reviews}) != 1:
            raise ValueError(f"{query_id}: independent reviews must judge the same pooled document set")
        adjudication = adjudication_by_query.get(query_id)
        if adjudication is None:
            raise ValueError(f"{query_id}: missing adjudication")
        if adjudication["adjudication_status"] != "approved":
            raise ValueError(f"{query_id}: disputed adjudication blocks gold export")
        if adjudication["adjudicator_id"] in set(annotators):
            raise ValueError(f"{query_id}: adjudicator must be independent from annotators")
        if {j["document_key"] for j in adjudication["document_judgments"]} != {
            j["document_key"] for j in query_reviews[0]["document_judgments"]
        }:
            raise ValueError(f"{query_id}: adjudication must cover the complete reviewed document pool")
    extra_queries = set(reviews_by_query) | set(adjudication_by_query)
    if extra_queries - set(query_candidates):
        raise ValueError("annotation files contain rows outside the candidate query set")
    return reviews_by_query, adjudication_by_query


def verify_postgres_sources(
    candidates: list[dict[str, Any]],
    reviews_by_query: dict[str, list[dict[str, Any]]] | None = None,
    adjudications: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    references: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        source = candidate.get("source_record_for_inspection")
        if source:
            references[source["document_key"]] = {
                **source,
                "domain_id": source.get("domain_id") or candidate.get("expected_domain"),
            }
    for records in (reviews_by_query or {}).values():
        for review in records:
            for judgment in review["document_judgments"]:
                references[judgment["document_key"]] = judgment
    for adjudication in (adjudications or {}).values():
        for judgment in adjudication["document_judgments"]:
            references[judgment["document_key"]] = judgment

    if not references:
        return {}, []
    engine = create_search_database_engine()
    try:
        with engine.connect() as connection:
            records = connection.execute(text("""
                SELECT document_key,domain_id,source_id,document_id,record_id,title,content,source_type,metadata
                FROM domain_search_documents WHERE document_key=ANY(:keys)
            """), {"keys": list(references)}).mappings()
            actual = {row["document_key"]: dict(row) for row in records}
    finally:
        engine.dispose()

    errors: list[str] = []
    for key, reference in references.items():
        row = actual.get(key)
        if row is None:
            errors.append(f"missing_postgres_document:{key}")
            continue
        for field in ("source_id", "document_id", "record_id"):
            expected = reference.get(field)
            if expected != row.get(field):
                errors.append(f"postgres_{field}_mismatch:{key}")
        digest = hashlib.sha256(str(row["content"]).encode("utf-8")).hexdigest()
        expected_digest = reference.get("content_sha256")
        if expected_digest and digest != expected_digest:
            errors.append(f"postgres_content_hash_mismatch:{key}")
        reference_domain = reference.get("domain_id")
        if reference_domain and reference_domain != row["domain_id"]:
            errors.append(f"postgres_domain_mismatch:{key}")
    return actual, errors


def build_gold(
    candidates: list[dict[str, Any]],
    reviews_by_query: dict[str, list[dict[str, Any]]],
    adjudications: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    gold = []
    for candidate in candidates:
        query = candidate.get("query")
        if not query:
            continue
        query_id = candidate["query_id"]
        adjudication = adjudications[query_id]
        adjudicated_judgments = adjudication["document_judgments"]
        relevant = {j["document_key"]: j["label"] for j in adjudicated_judgments if j["label"] is not None}
        references = []
        for judgment in adjudicated_judgments:
            row = source_records[judgment["document_key"]]
            metadata = row.get("metadata") or {}
            references.append({
                "document_key": row["document_key"],
                "domain_id": row["domain_id"],
                "source_id": row["source_id"],
                "document_id": row["document_id"],
                "record_id": row.get("record_id"),
                "title": row.get("title"),
                "source_url": metadata.get("source_url") or metadata.get("url_nguon") or metadata.get("source"),
                "content_sha256": hashlib.sha256(str(row["content"]).encode("utf-8")).hexdigest(),
                "checked_at": judgment["source_checked_at"],
            })
        reviews = reviews_by_query[query_id]
        gold.append({
            "sample_id": query_id,
            "query": str(adjudication.get("query") or query),
            "expected_domain": adjudication["expected_domain"],
            "expected_behavior": adjudication["query_status"],
            "relevant_evidence": relevant,
            "expected_answer_points": list(adjudication.get("expected_answer_points") or []),
            "review_status": "expert_confirmed",
            "reviewers": [row["annotator_id"] for row in reviews],
            "adjudicator": adjudication["adjudicator_id"],
            "adjudicated_at": adjudication["adjudicated_at"],
            "source_provenance": {"references": references},
            "annotation_notes": adjudication["notes"],
        })
    return gold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="evals/golden_candidates.jsonl")
    parser.add_argument("--reviews", help="JSONL with independent annotator submissions")
    parser.add_argument("--adjudications", help="JSONL with adjudicator decisions")
    parser.add_argument("--output", default="evals/gold_reviewed.jsonl")
    parser.add_argument("--audit-only", action="store_true", help="Check candidate source records only")
    args = parser.parse_args()
    candidates = read_jsonl(args.candidates)
    if args.audit_only:
        records, errors = verify_postgres_sources(candidates)
        slots = [row for row in candidates if row.get("query")]
        missing = [row["query_id"] for row in slots if not row.get("source_record_for_inspection")]
        print(f"query_candidates={len(slots)} candidate_source_references={len(records)} missing_source_reference={len(missing)}")
        print(f"source_validation_errors={len(errors)}")
        for error in errors:
            print(error)
        return 0 if not errors and not missing else 2
    if not args.reviews or not args.adjudications:
        parser.error("--reviews and --adjudications are required unless --audit-only is used")

    reviews = read_jsonl(args.reviews)
    adjudications = read_jsonl(args.adjudications)
    try:
        reviews_by_query, adjudication_by_query = validate_review_records(candidates, reviews, adjudications)
        source_records, errors = verify_postgres_sources(candidates, reviews_by_query, adjudication_by_query)
        if errors:
            raise ValueError("PostgreSQL source validation failed: " + ", ".join(errors[:10]))
        gold = build_gold(candidates, reviews_by_query, adjudication_by_query, source_records)
        for row in gold:
            if not row["relevant_evidence"] and row["expected_behavior"] == "answerable":
                raise ValueError(f"{row['sample_id']}: answerable query has no labeled relevant evidence")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold), encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"Gold export blocked: {exc}")
        return 2
    print(f"gold_samples={len(gold)} approved_labels={sum(len(row['relevant_evidence']) for row in gold)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
