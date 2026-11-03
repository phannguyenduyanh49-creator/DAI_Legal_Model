"""Reproducible, fail-closed benchmark for retrieval and manually reviewed answers.

This tool consumes JSONL only. It never calls a model or substitutes expected data
for missing runtime results. See QUALITY_BENCHMARK.md for the input contract.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    content = Path(path).read_text(encoding="utf-8-sig")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        rows = parsed
    elif parsed is not None:
        raise ValueError(f"{path}: expected JSONL objects or a JSON array")
    else:
        rows = []
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: each JSONL value must be an object")
            rows.append(row)
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every sample must be a JSON object")
    return rows


def validate_gold(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("gold dataset is empty; quality scores are unavailable")
    seen = set()
    for row in rows:
        sid = row.get("sample_id")
        if not sid or sid in seen:
            raise ValueError("gold sample_id values must be present and unique")
        seen.add(sid)
        reviewers = row.get("reviewers", [])
        provenance = row.get("source_provenance")
        if row.get("review_status") != "expert_confirmed":
            raise ValueError(f"{sid}: review_status must be expert_confirmed")
        if len(set(reviewers)) < 2 or not row.get("adjudicator"):
            raise ValueError(f"{sid}: require two independent reviewers and an adjudicator")
        if not isinstance(provenance, dict) or not provenance.get("references"):
            raise ValueError(f"{sid}: source_provenance.references is required")
        if not row.get("expected_domain"):
            raise ValueError(f"{sid}: expected_domain is required")
        if not isinstance(row.get("relevant_evidence"), dict):
            raise ValueError(f"{sid}: relevant_evidence must map document IDs to graded relevance")
        if not isinstance(row.get("expected_answer_points"), list):
            raise ValueError(f"{sid}: expected_answer_points is required")


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _ndcg(retrieved: list[str], grades: dict[str, Any], k: int) -> float:
    def dcg(ids: list[str]) -> float:
        return sum((2 ** float(grades.get(doc_id, 0)) - 1) / math.log2(rank + 2)
                   for rank, doc_id in enumerate(ids[:k]))
    ideal = sorted(grades, key=lambda doc_id: float(grades[doc_id]), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(retrieved) / denominator if denominator else 0.0


def evaluate(gold: list[dict[str, Any]], results: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    validate_gold(gold)
    gold_by_id = {row["sample_id"]: row for row in gold}
    result_by_id = {row.get("sample_id"): row for row in results}
    if len(result_by_id) != len(results) or set(result_by_id) != set(gold_by_id):
        raise ValueError("result sample IDs must exactly match gold IDs with no duplicates")

    recalls, precisions, reciprocal_ranks, ndcgs, domain_hits = [], [], [], [], []
    stage_times: dict[str, list[float]] = {}
    answer_stats = {key: [] for key in (
        "factual_correctness", "evidence_coverage", "citation_accuracy",
        "completeness", "unsupported_claim_rate",
    )}
    domain_stats: dict[str, dict[str, list[float]]] = {}
    answer_reviewed = True
    for sid, reference in gold_by_id.items():
        result = result_by_id[sid]
        relevant = {doc: float(grade) for doc, grade in reference["relevant_evidence"].items() if float(grade) > 0}
        ranked = result.get("retrieved_evidence_ids", [])
        if not isinstance(ranked, list) or len(ranked) != len(set(ranked)):
            raise ValueError(f"{sid}: retrieved_evidence_ids must be a deduplicated ranked list")
        top = ranked[:k]
        hits = sum(doc in relevant for doc in top)
        recalls.append(hits / len(relevant) if relevant else 0.0)
        precisions.append(hits / k if k else 0.0)
        reciprocal_ranks.append(next((1 / rank for rank, doc in enumerate(ranked, 1) if doc in relevant), 0.0))
        ndcgs.append(_ndcg(ranked, relevant, k))
        domain_hits.append(float(result.get("selected_domain") == reference["expected_domain"]))
        per_domain = domain_stats.setdefault(reference["expected_domain"], {
            "recall": [], "precision": [], "mrr": [], "ndcg": [], "domain_accuracy": [],
        })
        per_domain["recall"].append(recalls[-1])
        per_domain["precision"].append(precisions[-1])
        per_domain["mrr"].append(reciprocal_ranks[-1])
        per_domain["ndcg"].append(ndcgs[-1])
        per_domain["domain_accuracy"].append(domain_hits[-1])

        for stage, milliseconds in result.get("latency_ms", {}).items():
            if isinstance(milliseconds, (int, float)) and milliseconds >= 0:
                stage_times.setdefault(stage, []).append(float(milliseconds))

        review = result.get("answer_review")
        if not isinstance(review, dict) or review.get("review_status") != "expert_confirmed":
            answer_reviewed = False
            continue
        required = ("supported_claims", "unsupported_claims", "contradicted_claims",
                    "cited_claims", "accurate_citations", "expected_points_present", "expected_points_total")
        if any(not isinstance(review.get(field), (int, float)) for field in required):
            raise ValueError(f"{sid}: incomplete expert answer_review")
        claims = review["supported_claims"] + review["unsupported_claims"] + review["contradicted_claims"]
        cited = review["cited_claims"]
        answer_stats["factual_correctness"].append(review["supported_claims"] / claims if claims else 0.0)
        answer_stats["unsupported_claim_rate"].append(review["unsupported_claims"] / claims if claims else 0.0)
        answer_stats["evidence_coverage"].append(review["supported_claims"] / claims if claims else 0.0)
        answer_stats["citation_accuracy"].append(review["accurate_citations"] / cited if cited else 0.0)
        total_points = review["expected_points_total"]
        answer_stats["completeness"].append(review["expected_points_present"] / total_points if total_points else 0.0)

    return {
        "sample_count": len(gold), "top_k": k,
        "retrieval": {
            f"recall@{k}": _mean(recalls), f"precision@{k}": _mean(precisions),
            "mrr": _mean(reciprocal_ranks), f"ndcg@{k}": _mean(ndcgs),
            "domain_accuracy": _mean(domain_hits),
        },
        "answer": ({name: _mean(values) for name, values in answer_stats.items()}
                   if answer_reviewed else None),
        "answer_metrics_status": "expert_reviewed" if answer_reviewed else "unavailable: missing expert-confirmed answer review",
        "by_domain": {
            domain: {
                f"recall@{k}": _mean(values["recall"]),
                f"precision@{k}": _mean(values["precision"]),
                "mrr": _mean(values["mrr"]),
                f"ndcg@{k}": _mean(values["ndcg"]),
                "domain_accuracy": _mean(values["domain_accuracy"]),
                "sample_count": len(values["recall"]),
            }
            for domain, values in sorted(domain_stats.items())
        },
        "latency_ms": {stage: {"count": len(values), "mean": _mean(values),
                                "p50": statistics.median(values),
                                "p95": sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]}
                       for stage, values in sorted(stage_times.items())},
        "limitations": ["Metrics describe only the supplied samples and labels.",
                        "No model-based judge or legal-correctness inference is performed."],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--improved", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", help="Optional output JSON path; stdout otherwise")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        gold = read_jsonl(args.gold)
        baseline = evaluate(gold, read_jsonl(args.baseline), args.top_k)
        improved = evaluate(gold, read_jsonl(args.improved), args.top_k)
        payload = {"protocol": "paired expert-reviewed benchmark v1", "gold_samples": len(gold),
                   "baseline": baseline, "improved": improved,
                   "retrieval_delta": {key: improved["retrieval"][key] - baseline["retrieval"][key]
                                       for key in baseline["retrieval"]}}
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(serialized, encoding="utf-8")
        else:
            sys.stdout.write(serialized)
    except (OSError, ValueError) as exc:
        print(f"Benchmark unavailable: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
