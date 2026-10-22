from backend.evidence.schemas import Evidence
from backend.evidence.builder import EvidenceBuilder
from backend.rag.pipeline_policy import (
    build_catalog_fallback_after_retrieval_error,
    choose_data_source,
)


def make_catalog_evidence(fields=None):
    return Evidence(
        evidence_id="catalog:procedure-1",
        source_type="catalog",
        title="Thủ tục mẫu",
        content="Nội dung thủ tục trong catalog.",
        score=1.0,
        fields=fields or {},
    )


def test_catalog_evidence_is_preserved_when_retrieval_fails():
    evidence = make_catalog_evidence({"profile_components": "Đơn đăng ký"})
    result = build_catalog_fallback_after_retrieval_error(
        question="Hồ sơ cần giấy tờ gì?",
        catalog_evidences=[evidence],
        evidence_builder=EvidenceBuilder(min_score=0.3),
        allow_external=False,
    )

    assert result["retrieval_status"] == "error"
    assert result["evidence_builder_status"] == "completed"
    assert result["evidence_bundle"].evidences == [evidence]
    assert result["enough_data"] is True
    assert result["external_fallback"] == "not_needed"


def test_catalog_fallback_marks_requested_missing_field_and_does_not_request_mcp_without_opt_in():
    result = build_catalog_fallback_after_retrieval_error(
        question="Hồ sơ cần giấy tờ gì?",
        catalog_evidences=[make_catalog_evidence()],
        evidence_builder=EvidenceBuilder(min_score=0.3),
        allow_external=False,
    )

    assert result["enough_data"] is False
    assert result["missing_fields"] == ["profile_components"]
    assert result["external_fallback"] == "not_allowed"
    assert choose_data_source(
        has_local_evidence=True,
        enough_data=False,
        allow_external=False,
    ) == "local"


def test_external_mcp_requires_explicit_opt_in_when_local_data_is_absent():
    assert choose_data_source(False, False, False) == "permission"
    assert choose_data_source(False, False, True) == "external"


def test_insufficient_local_data_uses_external_only_after_opt_in():
    assert choose_data_source(True, False, False) == "local"
    assert choose_data_source(True, False, True) == "external"
