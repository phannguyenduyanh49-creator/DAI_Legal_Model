"""Validated, explicit search-backend cutover configuration."""

from __future__ import annotations

import os
import math


def resolve_search_config(environ=None) -> tuple[str, float | None]:
    environ = os.environ if environ is None else environ
    backend = str(environ.get("SEARCH_BACKEND", "qdrant")).strip().lower()
    if backend not in {"qdrant", "elasticsearch", "hybrid"}:
        raise ValueError("SEARCH_BACKEND must be 'qdrant', 'elasticsearch' or 'hybrid'")
    if backend == "qdrant":
        return backend, None
    raw_min_score = environ.get("ELASTICSEARCH_MIN_SCORE")
    # Hybrid uses BM25 rank, not a calibrated BM25 acceptance score. The
    # existing reranker and evidence gates still decide answer eligibility.
    if backend == "hybrid" and not str(raw_min_score or "").strip():
        return backend, None
    if raw_min_score is None or not str(raw_min_score).strip():
        raise ValueError(
            "ELASTICSEARCH_MIN_SCORE must be calibrated and configured before enabling Elasticsearch retrieval"
        )
    try:
        minimum = float(raw_min_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("ELASTICSEARCH_MIN_SCORE must be numeric") from exc
    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("ELASTICSEARCH_MIN_SCORE must be nonnegative")
    return backend, minimum
