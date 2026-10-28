from types import SimpleNamespace

from backend.answer.evidence_gate import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    make_missing_prompt_evidence_result,
)


def test_empty_retrieval_returns_explicit_failure_without_model_or_verifier_run():
    pipeline = {
        "stage_status": {
            "retrieval": "completed",
            "evidence_builder": "completed",
            "prompt_assembly": "not_started",
            "model_api": "not_started",
            "verifier": "not_started",
        },
        "confidence_status": "not_evaluated",
        "claim_grounding_status": "not_evaluated",
    }

    result = make_missing_prompt_evidence_result(
        pipeline,
        source_candidate_count=0,
        evidence_bundle_count=0,
        prompt_evidences=(),
    )

    assert result["answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert result["verifier"] == "not_verified"
    assert result["pipeline"]["failure_stage"] == "retrieval_pipeline"
    assert result["pipeline"]["failure_reason_code"] == "EVIDENCE_IDS_MISSING"
    assert result["pipeline"]["evidence_count"] == 0
    assert result["pipeline"]["evidence_ids"] == []
    assert result["pipeline"]["model_api"] == "not_started"
    assert result["pipeline"]["stage_status"]["model_api"] == "not_started"
    assert result["pipeline"]["stage_status"]["verifier"] == "not_started"
    assert result["pipeline"]["confidence_status"] == "not_evaluated"
    assert result["pipeline"]["claim_grounding_status"] == "not_evaluated"


def test_empty_prompt_after_bundle_build_fails_at_prompt_assembly():
    pipeline = {"stage_status": {}}
    result = make_missing_prompt_evidence_result(
        pipeline,
        source_candidate_count=1,
        evidence_bundle_count=1,
        prompt_evidences=(),
    )
    assert result["pipeline"]["failure_stage"] == "prompt_assembly"
    assert result["pipeline"]["stage_status"]["prompt_assembly"] == "failed"


def test_candidate_data_dropped_by_evidence_builder_is_reported_at_that_stage():
    result = make_missing_prompt_evidence_result(
        {},
        source_candidate_count=1,
        evidence_bundle_count=0,
        prompt_evidences=(),
    )
    assert result["pipeline"]["failure_stage"] == "evidence_builder"
    assert result["pipeline"]["stage_status"]["evidence_builder"] == "failed"


def test_valid_prompt_evidence_does_not_trigger_empty_evidence_failure():
    pipeline = {"stage_status": {}}
    evidence = SimpleNamespace(evidence_id="catalog:record-1", content="Actual source text")
    assert make_missing_prompt_evidence_result(
        pipeline,
        source_candidate_count=1,
        evidence_bundle_count=1,
        prompt_evidences=(evidence,),
    ) is None
    assert pipeline == {"stage_status": {}}


def test_missing_evidence_remains_fail_closed_for_general_and_tourism_domains():
    expected = {
        "general_assistant": "Hiện chưa có Evidence phù hợp để hỗ trợ câu trả lời tổng quát; hệ thống chưa tạo câu trả lời.",
        "tourism": "Hiện chưa có Evidence du lịch phù hợp; hệ thống chưa tạo câu trả lời.",
    }
    for domain_id, answer in expected.items():
        pipeline = {"stage_status": {}}
        result = make_missing_prompt_evidence_result(
            pipeline,
            source_candidate_count=0,
            evidence_bundle_count=0,
            prompt_evidences=(),
            domain_id=domain_id,
        )
        assert result["answer"] == answer
        assert result["pipeline"]["model_api"] == "not_started"
        assert result["pipeline"]["stage_status"]["model_api"] == "not_started"
        assert result["pipeline"]["verification_status"] == "not_verified"
        assert result["pipeline"]["verifier_scope"] == "not_evaluated"
        assert result["pipeline"]["stage_status"]["verifier"] == "not_started"
