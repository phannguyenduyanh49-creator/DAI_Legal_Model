"""Bounded, content-free candidate telemetry for retrieval investigations."""

from __future__ import annotations

from typing import Any

from backend.rag.relevance import finite_retrieval_score


def retrieval_candidate_summary(hit: Any, decision: str, *, relevance: dict | None = None) -> dict:
    payload = getattr(hit, "payload", None) or {}
    fields = payload.get("record_fields")
    fields = fields if isinstance(fields, dict) else {}
    result = {
        "candidate_id": str(getattr(hit, "id", "unknown"))[:100],
        "record_id": str(payload.get("record_id") or fields.get("record_id") or "")[:100],
        "source_id": str(payload.get("source_id") or payload.get("formality_id") or "")[:100],
        "document_id": str(payload.get("document_id") or "")[:100],
        "domain_id": str(payload.get("domain_id") or "")[:60],
        "retrieval_score": (
            round(score, 6)
            if (score := finite_retrieval_score(getattr(hit, "score", None))) is not None
            else None
        ),
        "decision": decision,
    }
    if relevance is not None:
        result["query_coverage"] = round(float(relevance.get("query_coverage") or 0), 4)
        result["title_precision"] = round(float(relevance.get("title_precision") or 0), 4)
    return result


def reranker_candidate_summary(candidate: dict, threshold: float) -> dict:
    item = candidate["item"]
    fields = item.get("record_fields")
    fields = fields if isinstance(fields, dict) else {}
    return {
        "candidate_id": str(item.get("point_id") or "unknown")[:100],
        "record_id": str(item.get("record_id") or fields.get("record_id") or "")[:100],
        "source_id": str(item.get("source_id") or "")[:100],
        "rank": candidate["rank"],
        "normalized_score": round(float(candidate["score"]), 6),
        "raw_score": None,  # normalize=True does not return raw logits.
        "threshold": threshold,
        "selected": bool(candidate["selected"]),
        "rejection_reason": candidate["rejection_reason"],
    }
