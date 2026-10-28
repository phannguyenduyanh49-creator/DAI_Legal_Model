"""Export unlabeled claim/Evidence candidates from the Direct Catalog.

This is a read-only export. Candidate claims are verbatim Catalog field text;
they are not gold labels and must be reviewed before they enter evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from sqlalchemy import select

from backend.catalog.database import SessionLocal
from backend.catalog.models import Procedure


FIELD_NAMES = (
    "description",
    "execution_steps",
    "requirements_conditions",
    "profile_components",
    "fees",
    "results",
    "legal_basis",
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỵ0-9])")


def _candidate_texts(value: str) -> list[str]:
    """Keep source wording intact; split only on conservative sentence edges."""
    value = " ".join(value.split()).strip()
    if not value:
        return []
    parts = [part.strip(" \t\r\n-•") for part in _SENTENCE_SPLIT.split(value)]
    return [part for part in parts if part]


def _candidate_rows(records: Iterable[Procedure], limit: int) -> list[dict[str, str]]:
    # Build category buckets, then round-robin them so one category does not
    # dominate the export. One pair per source record prevents near-duplicate
    # fields from a single procedure filling the candidate set.
    buckets: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for record_index, record in enumerate(records):
        category = (record.category_name or record.target_type or "uncategorized").strip()
        preferred_fields = FIELD_NAMES[record_index % len(FIELD_NAMES):] + FIELD_NAMES[:record_index % len(FIELD_NAMES)]
        for field_name in preferred_fields:
            raw_value = getattr(record, field_name, None)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            text = _candidate_texts(raw_value)[0]
            if not text:
                continue
            source_id = str(record.formality_id)
            buckets[category].append(
                {
                    "sample_id": f"catalog-candidate-{source_id}-{field_name}",
                    "domain": "legal_ai",
                    "claim": text,
                    "evidence_id": f"catalog:{source_id}",
                    "evidence_text": raw_value.strip(),
                    "gold_label": "",
                    "review_status": "unreviewed",
                    "source_record_id": source_id,
                    "source_field": field_name,
                    "source_url": str(record.source_url or ""),
                    "notes": (
                        "Unlabeled candidate. Claim is verbatim Catalog text; "
                        "review and confirm against the source record before evaluation."
                    ),
                }
            )
            break

    ordered_categories = sorted(buckets)
    selected: list[dict[str, str]] = []
    while ordered_categories and len(selected) < limit:
        remaining_categories = []
        for category in ordered_categories:
            if buckets[category] and len(selected) < limit:
                selected.append(buckets[category].popleft())
            if buckets[category]:
                remaining_categories.append(category)
        ordered_categories = remaining_categories
    return selected


def extract_candidates(limit: int = 80) -> list[dict[str, str]]:
    if not 50 <= limit <= 100:
        raise ValueError("limit must be between 50 and 100")
    statement = (
        select(Procedure)
        .where(Procedure.status == "active")
        .order_by(Procedure.formality_id)
    )
    with SessionLocal() as session:
        records = session.execute(statement).scalars().all()
    return _candidate_rows(records, limit)


def write_jsonl(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80, choices=range(50, 101))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/verifier/evaluation/catalog_candidates.jsonl"),
    )
    args = parser.parse_args()
    rows = extract_candidates(args.limit)
    write_jsonl(rows, args.output)
    print(f"exported={len(rows)} output={args.output}")
    print("gold_labels=empty review_status=unreviewed; candidates are not benchmark labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
