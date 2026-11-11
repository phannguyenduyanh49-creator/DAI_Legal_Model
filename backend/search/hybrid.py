"""Read-only, domain-scoped rank fusion. RRF is not a relevance confidence."""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class HybridHit:
    id: str
    score: float
    payload: dict[str, Any]
    retrieval_provenance: list[dict[str, Any]]


class HybridRetrievalUnavailable(RuntimeError):
    """Both branches failed; keep their safe diagnostics for the Core trace."""

    def __init__(self, diagnostics: dict[str, dict[str, Any]]):
        super().__init__("Both hybrid retrieval branches failed")
        self.diagnostics = diagnostics


def fuse_rankings(rankings, domain_id, *, limit, vector_min_score, rrf_k=60):
    """Preserve chunks and source payloads; merge only matching source+content.

    Untagged legacy Qdrant records are accepted only from collections already
    selected by the authoritative domain registry. ES must always be tagged.
    """
    merged = {}
    for backend, hits in rankings:
        seen = set()
        for rank, hit in enumerate(hits, 1):
            payload = hit.payload or {}
            tagged_domain = payload.get("domain_id")
            if tagged_domain and tagged_domain != domain_id:
                continue
            if not tagged_domain and (backend == "elasticsearch" or payload.get("source_type") == "domain_upload"):
                continue
            score = float(hit.score or 0)
            if not math.isfinite(score) or (backend == "qdrant" and score < vector_min_score):
                continue
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            if payload.get("source_type") == "domain_upload":
                document_id = str(payload.get("document_id") or "")
                source_id = str(payload.get("source_id") or "")
                evidence_id = str(payload.get("evidence_id") or "")
                if (not document_id or source_id != document_id or
                        not evidence_id.startswith(f"domain:{domain_id}:{document_id}:")):
                    continue
            source = payload.get("source_id") or payload.get("document_id")
            record_fields = payload.get("record_fields")
            record_id = payload.get("record_id") or (
                record_fields.get("record_id") if isinstance(record_fields, dict) else None
            )
            identity = (
                domain_id, str(source), str(payload.get("document_id") or ""),
                str(record_id or ""), str(payload.get("chunk_index", "")),
                hashlib.sha256(text.encode()).hexdigest(),
            ) if source else (backend, str(hit.id))
            if identity in seen:
                continue
            seen.add(identity)
            if identity not in merged:
                merged[identity] = HybridHit(str(hit.id), 0.0, dict(payload), [])
            result = merged[identity]
            result.score += 1.0 / (rrf_k + rank)
            result.retrieval_provenance.append({"backend": backend, "point_id": str(hit.id), "rank": rank, "raw_score": score})
    return sorted(merged.values(), key=lambda hit: (-hit.score, hit.id))[:limit]


async def retrieve_hybrid(*, query, domain_id, vector_search, keyword_search,
                          limit, vector_min_score, timeout=30.0):
    """Bound candidate work per branch; report partial outages explicitly.

    Timing out a worker does not kill its thread; clients must retain their
    socket timeouts. No retries, model loads or writes happen here.
    """
    async def run(name, callback):
        started = time.perf_counter()
        try:
            hits = await asyncio.wait_for(asyncio.to_thread(callback), timeout)
            status, error = "completed", None
        except Exception as exc:
            hits, status, error = [], "failed", type(exc).__name__
        return name, hits, {"status": status, "error_type": error,
                            "candidate_count": len(hits),
                            "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
    branches = await asyncio.gather(run("qdrant", vector_search), run("elasticsearch", keyword_search))
    diagnostics = {name: diagnostic for name, _, diagnostic in branches}
    if all(item["status"] == "failed" for item in diagnostics.values()):
        raise HybridRetrievalUnavailable(diagnostics)
    hits = fuse_rankings([(name, hits) for name, hits, _ in branches], domain_id,
                         limit=limit, vector_min_score=vector_min_score)
    return hits, diagnostics
