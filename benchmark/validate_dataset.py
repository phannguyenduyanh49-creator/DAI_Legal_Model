"""Validate operational benchmark inputs; this never creates relevance labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata


TARGET_CATEGORIES = {
    "legal_rag": 25,
    "administrative_law": 20,
    "hybrid_retrieval": 15,
    "citation_grounding": 15,
    "mcp_permission": 5,
    "insufficient_evidence": 10,
    "multi_domain_routing": 5,
    "security_prompt_injection": 5,
}


def _query_key(query: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", query).casefold()).strip()


def validate_dataset(dataset: dict, *, target_100: bool = False) -> dict:
    errors: list[str] = []
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        return {"errors": ["cases must be an array"], "case_count": 0}
    ids: set[str] = set()
    queries: set[tuple[str, str]] = set()
    categories = Counter()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            errors.append(f"case {index}: must be an object")
            continue
        case_id = case.get("case_id")
        query = case.get("query")
        domain = case.get("expected_domain")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            errors.append(f"case {index}: missing or duplicate case_id")
        else:
            ids.add(case_id)
        if not isinstance(query, str) or not query.strip():
            errors.append(f"case {index}: missing query")
        elif not isinstance(domain, str) or not domain.strip():
            errors.append(f"case {index}: missing expected_domain")
        else:
            key = (domain, _query_key(query))
            if key in queries:
                errors.append(f"case {index}: duplicate query in domain {domain}")
            queries.add(key)
        category = case.get("category", case.get("group"))
        if not isinstance(category, str) or not category.strip():
            errors.append(f"case {index}: missing category/group")
        else:
            categories[category] += 1
        if case.get("review_status") == "reviewed" and not case.get("reviewer"):
            errors.append(f"case {index}: reviewed without reviewer")
    if target_100:
        if len(cases) != 100:
            errors.append(f"target requires 100 cases; found {len(cases)}")
        for category, expected in TARGET_CATEGORIES.items():
            if categories[category] != expected:
                errors.append(f"{category}: expected {expected}, found {categories[category]}")
    return {
        "dataset_id": dataset.get("dataset_id"),
        "case_count": len(cases),
        "categories": dict(categories),
        "reviewed_case_count": sum(isinstance(case, dict) and case.get("review_status") == "reviewed" for case in cases),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--target-100", action="store_true")
    args = parser.parse_args()
    report = validate_dataset(json.loads(args.dataset.read_text(encoding="utf-8")), target_100=args.target_100)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
