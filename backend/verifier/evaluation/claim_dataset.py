"""Version and validate unreviewed claim–Evidence candidates without inventing gold labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_ID = "dai-legal-claim-evidence-v1"
LABELS = {"supported", "contradicted", "not_supported", "insufficient_context"}


def candidate_from_catalog(row: dict) -> dict:
    """Retain exact Catalog text and provenance; leave unproven fields empty."""
    source_record_id = row["source_record_id"]
    return {
        "dataset_id": DATASET_ID,
        "sample_id": row["sample_id"],
        "question": None,
        "claim": row["claim"],
        "evidence_excerpt": row["evidence_text"],
        "evidence_id": row["evidence_id"],
        "source_document_id": None,
        "source_metadata": {
            "domain_id": row["domain"],
            "source_record_id": source_record_id,
            "source_field": row.get("source_field"),
            "source_url": row.get("source_url"),
        },
        "label": None,
        "rationale": None,
        "reviewer": None,
        "review_status": "candidate_unreviewed",
    }


def validate(rows: list[dict]) -> dict:
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str, str]] = set()
    source_by_evidence_id: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    for index, row in enumerate(rows, 1):
        key = row.get("sample_id")
        if not isinstance(key, str) or not key.strip() or key in seen_ids:
            errors.append(f"row {index}: missing or duplicate sample_id")
        else:
            seen_ids.add(key)
        if row.get("dataset_id") != DATASET_ID:
            errors.append(f"row {index}: wrong dataset_id")
        for field in ("claim", "evidence_excerpt", "evidence_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                errors.append(f"row {index}: missing {field}")
        metadata = row.get("source_metadata")
        if not isinstance(metadata, dict):
            errors.append(f"row {index}: missing source_metadata")
            continue
        record_id = metadata.get("source_record_id")
        domain_id = metadata.get("domain_id")
        if not isinstance(record_id, str) or not record_id.strip() or not isinstance(domain_id, str) or not domain_id.strip():
            errors.append(f"row {index}: missing record/domain provenance")
        if isinstance(row.get("evidence_id"), str) and row["evidence_id"].startswith("catalog:") and row["evidence_id"] != f"catalog:{record_id}":
            errors.append(f"row {index}: Evidence ID contradicts source record")
        evidence_id = row.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id.strip():
            provenance = (str(domain_id), str(record_id))
            previous = source_by_evidence_id.setdefault(evidence_id, provenance)
            if previous != provenance:
                errors.append(f"row {index}: Evidence ID has conflicting domain/source metadata")
        pair = (str(row.get("claim", "")).strip(), str(row.get("evidence_excerpt", "")).strip(), str(row.get("evidence_id", "")))
        if pair in seen_pairs:
            errors.append(f"row {index}: duplicate claim–Evidence pair")
        seen_pairs.add(pair)
        status = row.get("review_status")
        label = row.get("label")
        if status == "candidate_unreviewed":
            if label is not None or row.get("reviewer") is not None or row.get("rationale") is not None:
                errors.append(f"row {index}: candidate has an unapproved label or reviewer")
        elif status == "reviewed":
            if label not in LABELS or not all(isinstance(row.get(field), str) and row[field].strip() for field in ("question", "source_document_id", "rationale", "reviewer")):
                errors.append(f"row {index}: reviewed row lacks label or audit fields")
        else:
            errors.append(f"row {index}: invalid review_status")
    return {
        "dataset_id": DATASET_ID,
        "total": len(rows),
        "reviewed": sum(row.get("review_status") == "reviewed" for row in rows),
        "candidate_unreviewed": sum(row.get("review_status") == "candidate_unreviewed" for row in rows),
        "errors": errors,
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-catalog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.from_catalog:
        if not args.output or args.output.exists():
            parser.error("--output must name a new path")
        rows = [candidate_from_catalog(row) for row in load_jsonl(args.from_catalog)]
    elif args.validate:
        rows = load_jsonl(args.validate)
    else:
        parser.error("provide --from-catalog and --output, or --validate")
    report = validate(rows)
    if report["errors"]:
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 1
    if args.from_catalog:
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        report["sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
