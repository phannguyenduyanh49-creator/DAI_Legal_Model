from types import SimpleNamespace

from backend.rag.candidate_diagnostics import (
    reranker_candidate_summary,
    retrieval_candidate_summary,
)


def test_retrieval_summary_records_identity_and_rejection_without_content():
    hit = SimpleNamespace(
        id="point-1", score=0.62,
        payload={
            "domain_id": "legal_ai", "source_id": "source-1",
            "document_id": "document-1", "record_fields": {"record_id": "row-1"},
            "text": "PRIVATE SOURCE TEXT", "title": "PRIVATE TITLE",
        },
    )
    summary = retrieval_candidate_summary(
        hit, "reject_irrelevant",
        relevance={"query_coverage": 0.1, "title_precision": 0.2},
    )
    assert summary["candidate_id"] == "point-1"
    assert summary["record_id"] == "row-1"
    assert summary["decision"] == "reject_irrelevant"
    assert summary["retrieval_score"] == 0.62
    assert "PRIVATE" not in str(summary)


def test_reranker_summary_distinguishes_normalized_score_from_unavailable_raw_logit():
    candidate = {
        "item": {"point_id": "point-1", "record_id": "row-1", "source_id": "source-1", "text": "PRIVATE"},
        "rank": 2, "score": 0.42, "selected": False,
        "rejection_reason": "score_below_threshold",
    }
    summary = reranker_candidate_summary(candidate, 0.5)
    assert summary["normalized_score"] == 0.42
    assert summary["raw_score"] is None
    assert summary["threshold"] == 0.5
    assert summary["rejection_reason"] == "score_below_threshold"
    assert "PRIVATE" not in str(summary)


def test_legacy_dvc_candidate_uses_real_formality_id_as_source_provenance():
    hit = SimpleNamespace(
        id=100272, score=0.584811,
        payload={"formality_id": "019d2bf9-6c83-7162-9202-92098b1baffd"},
    )
    summary = retrieval_candidate_summary(hit, "reject_irrelevant")
    assert summary["source_id"] == "019d2bf9-6c83-7162-9202-92098b1baffd"
    assert summary["document_id"] == ""
