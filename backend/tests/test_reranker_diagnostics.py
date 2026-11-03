from backend.rag.reranker_diagnostics import (
    build_reranker_document_text,
    format_reranker_diagnostic,
    score_and_select_candidates,
)
from backend.evidence.builder import EvidenceBuilder
from backend.rag.adapter import rag_result_to_evidence
import pytest


def test_uploaded_record_reranker_input_uses_record_text_not_legal_title_prefix():
    rendered = build_reranker_document_text({
        "source_type": "domain_upload",
        "procedure_name": "Healthcare_Education_RAG_Datasets.xlsx",
        "record_name": "Mock Dermatology Clinic",
        "text": "chuyen_khoa: Da lieu\nquan_huyen: Binh Thanh",
    })

    assert rendered.startswith("Mock Dermatology Clinic\n")
    assert "Tên thủ tục:" not in rendered
    assert "Healthcare_Education_RAG_Datasets.xlsx" not in rendered


def test_legal_reranker_input_keeps_existing_procedure_format():
    assert build_reranker_document_text({
        "procedure_name": "Company registration",
        "text": "Legal source text",
    }) == "Tên thủ tục: Company registration\nLegal source text"


def test_candidate_decisions_expose_threshold_margin_and_top_k_reasons():
    contexts = [
        {"point_id": "p1", "record_id": "HC-1", "source_id": "doc"},
        {"point_id": "p2", "record_id": "HC-2", "source_id": "doc"},
        {"point_id": "p3", "record_id": "HC-3", "source_id": "doc"},
        {"point_id": "p4", "record_id": "HC-4", "source_id": "doc"},
    ]
    results = score_and_select_candidates(
        contexts,
        [0.90, 0.78, 0.49, 0.70],
        threshold=0.50,
        max_results=1,
        apply_margin=False,
        margin_threshold=0.05,
    )

    assert [(item["item"]["point_id"], item["rank"]) for item in results] == [
        ("p1", 1), ("p2", 2), ("p4", 3), ("p3", 4)
    ]
    assert results[0]["selected"] is True
    assert results[1]["rejection_reason"] == "top_k_limit"
    assert results[2]["rejection_reason"] == "top_k_limit"
    assert results[3]["rejection_reason"] == "score_below_threshold"

    line = format_reranker_diagnostic(
        results[3]["item"], rank=results[3]["rank"], score=results[3]["score"],
        threshold=0.50, selected=results[3]["selected"],
        rejection_reason=results[3]["rejection_reason"],
    )
    assert "candidate_id=p3" in line
    assert "record_id=HC-3" in line
    assert "record_identity=record_id:HC-3" in line
    assert "source_id=doc" in line
    assert "reranker_score=0.490000" in line
    assert "score_normalized=true" in line
    assert "threshold=0.500000" in line
    assert "decision=rejected" in line
    assert "rejection_reason=score_below_threshold" in line


def test_margin_rejection_is_distinct_from_threshold_rejection():
    results = score_and_select_candidates(
        [{"point_id": "top"}, {"point_id": "below_margin"}],
        [0.90, 0.70],
        threshold=0.50,
        max_results=3,
        apply_margin=True,
        margin_threshold=0.05,
    )

    assert results[0]["selected"] is True
    assert results[1]["selected"] is False
    assert results[1]["rejection_reason"] == "score_gap_exceeds_margin"


def test_incomplete_reranker_scores_are_not_silently_aligned_to_candidates():
    with pytest.raises(ValueError, match="match the candidate count"):
        score_and_select_candidates(
            [{"point_id": "one"}, {"point_id": "two"}],
            [0.8],
            threshold=0.50,
            max_results=2,
            apply_margin=False,
            margin_threshold=0.05,
        )


def test_only_selected_reranker_candidates_can_become_evidence():
    contexts = [
        {
            "point_id": "point-supported",
            "source_type": "domain_upload",
            "domain_id": "healthcare",
            "record_id": "HC-1",
            "source_id": "health-source",
            "document_id": "health-source",
            "evidence_id": "domain:healthcare:health-source:1",
            "title": "health.xlsx",
            "text": "record_id: HC-1\nchuyen_khoa: Da lieu",
        },
        {
            "point_id": "point-rejected",
            "source_type": "domain_upload",
            "domain_id": "healthcare",
            "record_id": "HC-2",
            "source_id": "health-source",
            "document_id": "health-source",
            "evidence_id": "domain:healthcare:health-source:2",
            "title": "health.xlsx",
            "text": "record_id: HC-2\nchuyen_khoa: Pediatrics",
        },
    ]
    decisions = score_and_select_candidates(
        contexts,
        [0.80, 0.49],
        threshold=0.50,
        max_results=3,
        apply_margin=False,
        margin_threshold=0.05,
    )
    selected = [
        {**decision["item"], "rerank_score": decision["score"]}
        for decision in decisions
        if decision["selected"]
    ]
    bundle = EvidenceBuilder(min_score=0.3).build(
        rag_evidences=[rag_result_to_evidence(item) for item in selected]
    )

    assert [item.evidence_id for item in bundle.evidences] == [
        "domain:healthcare:health-source:1"
    ]
    assert decisions[1]["rejection_reason"] == "score_below_threshold"


def test_row_identity_survives_reranking_when_source_has_no_record_id():
    original = {
        "point_id": "point-1",
        "source_type": "domain_upload",
        "domain_id": "shopping",
        "source_id": "workbook-1",
        "document_id": "workbook-1",
        "evidence_id": "domain:shopping:workbook-1:2",
        "chunk_index": 2,
        "sheet_name": "Products",
        "row_number": 3,
        "title": "products.xlsx",
        "text": "product: kettle",
    }
    decisions = score_and_select_candidates(
        [original], [0.82], threshold=0.50, max_results=1,
        apply_margin=False, margin_threshold=0.05,
    )
    selected = dict(decisions[0]["item"], rerank_score=decisions[0]["score"])
    evidence = rag_result_to_evidence(selected)
    bundle = EvidenceBuilder().build(rag_evidences=[evidence])

    assert bundle.total == 1
    assert bundle.evidences[0].source_id == "workbook-1"
    assert "record_id" not in bundle.evidences[0].metadata
    assert bundle.evidences[0].metadata["document_id"] == "workbook-1"
    assert bundle.evidences[0].metadata["point_id"] == "point-1"
    assert bundle.evidences[0].metadata["chunk_index"] == 2
    assert bundle.evidences[0].metadata["sheet_name"] == "Products"
    assert bundle.evidences[0].metadata["row_number"] == 3
