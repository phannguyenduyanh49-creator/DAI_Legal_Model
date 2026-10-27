import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import core_gateway
from backend.answer.diagnostics import classify_upstream_exception
from backend.core_gateway import (
    LegacyPipelineResult,
    UNVERIFIED_ANSWER_FALLBACK,
    _citation_integrity_failure_reason,
    _business_outcome,
    _verification_fallback_message,
)
from backend.answer.reference_policy import reference_answer_failure_reason


def verified_result(**pipeline_overrides):
    evidence_id = "rag:test-1"
    pipeline = {
        "citation_verification": "passed",
        "claim_grounding_status": "passed",
        "confidence_status": "not_required",
        "verification_status": "passed",
        "evidence_ids": [evidence_id],
        "cited_evidence_ids": [evidence_id],
    }
    pipeline.update(pipeline_overrides)
    return LegacyPipelineResult(
        answer=f"Answer [Evidence ID: {evidence_id}]",
        pipeline=pipeline,
        verifier_status="passed",
    )


def unverified_reference_result(**pipeline_overrides):
    evidence_id = "rag:reference-1"
    pipeline = {
        "model_api": "completed",
        "prompt_assembly": "completed",
        "evidence_builder": "completed",
        "enough_data": "sufficient",
        "required_fields": ["profile_components"],
        "missing_fields": [],
        "citation_verification": "passed",
        "claim_grounding_status": "not_evaluated",
        "confidence_status": "not_verified",
        "verification_threshold": 0.85,
        "verification_status": "not_verified",
        "evidence_count": 1,
        "evidence_ids": [evidence_id],
        "cited_evidence_ids": [evidence_id],
        "claim_verification_results": [{
            "claim": "The evidence supports this procedural statement.",
            "evidence_ids": [evidence_id],
            "passed": True,
            "citation_integrity_status": "passed",
            "claim_grounding_status": "not_evaluated",
            "reasons": [],
        }],
        "failure_stage": "verifier",
        "failure_reason_code": "VERIFICATION_CONFIDENCE_NOT_VERIFIED",
    }
    pipeline.update(pipeline_overrides)
    return LegacyPipelineResult(
        answer=f"The procedure requires the listed document. [Evidence ID: {evidence_id}]",
        pipeline=pipeline,
        verifier_status="not_verified",
    )


def test_business_outcome_separates_clarification_withholding_and_verification_failure():
    assert _business_outcome(
        LegacyPipelineResult("Please approve", {"response_status": "waiting_permission"}, "not_required"),
        True,
    ) == "needs_clarification"
    assert _business_outcome(
        LegacyPipelineResult("Withheld", {"failure_stage": "evidence_builder"}, "failed"),
        False,
    ) == "withheld"
    assert _business_outcome(
        LegacyPipelineResult(
            "Withheld",
            {
                "model_api": "completed",
                "citation_verification": "passed",
                "verification_status": "not_verified",
            },
            "not_verified",
        ),
        False,
    ) == "withheld"
    assert _business_outcome(verified_result(), True) == "answered"


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (LegacyPipelineResult("answer", {}, verifier_status="failed"), "VERIFIER_NOT_PASSED"),
        (verified_result(citation_verification="failed"), "CITATION_VERIFICATION_NOT_PASSED"),
        (verified_result(evidence_ids=[]), "EVIDENCE_IDS_MISSING"),
        (verified_result(cited_evidence_ids=[]), "CITATION_TELEMETRY_MISMATCH"),
        (verified_result(cited_evidence_ids=["rag:other"]), "EVIDENCE_ID_NOT_ALLOWED"),
    ],
)
def test_gateway_returns_stable_citation_failure_reasons(result, expected):
    assert _citation_integrity_failure_reason(result) == expected


def test_gateway_reports_missing_prompt_evidence_before_generic_citation_failure():
    result = LegacyPipelineResult(
        answer="",
        pipeline={
            "evidence_count": 0,
            "evidence_ids": [],
            "cited_evidence_ids": [],
            "citation_verification": "failed",
            "verification_status": "failed",
            "failure_stage": "verifier",
            "failure_reason_code": "CITATION_MISSING",
        },
        verifier_status="failed",
    )
    assert _citation_integrity_failure_reason(result) == "EVIDENCE_IDS_MISSING"


def test_gateway_passes_only_matching_citation_telemetry():
    assert _citation_integrity_failure_reason(verified_result()) is None


def test_safe_failure_messages_distinguish_generation_citation_and_grounding():
    from backend.core_gateway import _verification_fallback_message

    model_error = _verification_fallback_message("MODEL_API_REQUEST_FAILED")
    citation_error = _verification_fallback_message("CITATION_FORMAT_INVALID")
    grounding_error = _verification_fallback_message("CLAIM_GROUNDING_NOT_EVALUATED")
    missing_evidence = _verification_fallback_message("EVIDENCE_IDS_MISSING")

    assert "mô hình" in model_error
    assert "citation" in citation_error
    assert "Evidence" in grounding_error
    assert "Evidence" in missing_evidence
    assert len({model_error, citation_error, grounding_error, missing_evidence}) == 4


def test_gateway_requires_confidence_when_threshold_is_configured():
    result = verified_result(
        verification_threshold=0.85,
        confidence_status="not_verified",
        claim_grounding_status="not_evaluated",
        verification_status="not_verified",
    )
    assert (
        _citation_integrity_failure_reason(result)
        == "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
    )


def test_gateway_preserves_confidence_failure_when_citation_passed_but_score_is_missing():
    result = verified_result(
        verification_threshold=0.85,
        confidence_status="not_verified",
        confidence_score=None,
        citation_verification="passed",
        claim_grounding_status="not_evaluated",
        verification_status="not_verified",
    )

    assert _citation_integrity_failure_reason(result) == (
        "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
    )


def test_gateway_does_not_require_confidence_when_no_threshold_is_configured():
    result = verified_result(
        verification_threshold=None,
        confidence_status="not_required",
        claim_grounding_status="not_evaluated",
        verification_status="not_verified",
    )

    # Confidence is not the failure in a citation-only scope. The Gateway
    # still withholds an answer because this verifier has no grounding scorer.
    assert _citation_integrity_failure_reason(result) == (
        "CLAIM_GROUNDING_NOT_EVALUATED"
    )


def test_gateway_does_not_call_citation_integrity_semantic_grounding():
    result = verified_result(
        claim_grounding_status="not_evaluated",
        verification_status="not_verified",
    )
    assert (
        _citation_integrity_failure_reason(result)
        == "CLAIM_GROUNDING_NOT_EVALUATED"
    )


def test_gateway_rejects_grounding_failure_even_when_confidence_and_citations_pass():
    result = verified_result(
        verification_threshold=0.85,
        confidence_status="passed",
        confidence_score=0.91,
        claim_grounding_status="failed",
        verification_status="failed",
    )
    assert _citation_integrity_failure_reason(result) == "CLAIM_GROUNDING_NOT_PASSED"


def test_gateway_rejects_citation_telemetry_that_does_not_match_answer():
    result = verified_result()
    result.answer = "Answer without citation"
    assert _citation_integrity_failure_reason(result) == "CITATION_TELEMETRY_MISMATCH"


def test_reference_answer_policy_allows_only_complete_citation_linked_unverified_answer():
    result = unverified_reference_result()
    assert reference_answer_failure_reason(
        result.answer, result.pipeline, result.verifier_status
    ) is None
    assert _citation_integrity_failure_reason(result) == (
        "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
    )
    assert _business_outcome(result, False, True) == "answered_unverified"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"evidence_ids": ["rag:reference-1"], "cited_evidence_ids": ["rag:other"],
          "claim_verification_results": [{"passed": True, "citation_integrity_status": "passed",
                                            "evidence_ids": ["rag:other"], "reasons": []}]},
         "EVIDENCE_ID_NOT_ALLOWED"),
        ({"citation_verification": "failed"}, "CITATION_VERIFICATION_NOT_PASSED"),
        ({"evidence_count": 0, "evidence_ids": [], "cited_evidence_ids": [],
          "claim_verification_results": []}, "EVIDENCE_IDS_MISSING"),
        ({"enough_data": "insufficient", "missing_fields": ["fees"]},
         "REFERENCE_ANSWER_ESSENTIAL_FACTS_NOT_SUFFICIENT"),
        ({"claim_verification_results": []}, "CLAIM_CITATION_RESULTS_MISSING"),
        ({"claim_grounding_status": "insufficient_evidence"}, "CLAIM_GROUNDING_NOT_ELIGIBLE"),
        ({"claim_grounding_status": "inconclusive"}, "CLAIM_GROUNDING_NOT_ELIGIBLE"),
        ({"claim_grounding_status": "evaluator_unavailable"}, "CLAIM_GROUNDING_NOT_ELIGIBLE"),
        ({"confidence_status": "not_required", "claim_grounding_status": "passed"},
         "REFERENCE_ANSWER_NO_UNVERIFIED_EVALUATOR"),
    ],
)
def test_reference_answer_policy_rejects_integrity_or_sufficiency_gaps(overrides, reason):
    result = unverified_reference_result(**overrides)
    assert reference_answer_failure_reason(
        result.answer, result.pipeline, result.verifier_status
    ) == reason


def test_reference_answer_policy_preserves_explicit_evaluator_pass_and_failure():
    passed = unverified_reference_result(
        confidence_status="passed",
        confidence_score=0.93,
        claim_grounding_status="passed",
        verification_status="passed",
        failure_stage=None,
        failure_reason_code=None,
    )
    passed.verifier_status = "passed"
    assert _citation_integrity_failure_reason(passed) is None
    assert _business_outcome(passed, True) == "answered"

    failed = unverified_reference_result(
        claim_grounding_status="failed",
        verification_status="failed",
        failure_reason_code="CLAIM_GROUNDING_NOT_PASSED",
    )
    failed.verifier_status = "failed"
    assert reference_answer_failure_reason(
        failed.answer, failed.pipeline, failed.verifier_status
    ) is not None
    assert _business_outcome(failed, False) == "verification_failed"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ReadTimeout("timeout"), "UPSTREAM_TIMEOUT"),
        (
            httpx.HTTPStatusError(
                "bad status",
                request=httpx.Request("POST", "http://core"),
                response=httpx.Response(503),
            ),
            "UPSTREAM_HTTP_ERROR",
        ),
        (httpx.ConnectError("unreachable"), "UPSTREAM_CONNECTION_ERROR"),
        (ValueError("unexpected"), "UNKNOWN_PIPELINE_FAILURE"),
    ],
)
def test_upstream_exception_reason_codes_are_specific(error, expected):
    assert classify_upstream_exception(error) == expected


@pytest.mark.parametrize(
    ("exception", "reason", "upstream_status"),
    [
        (httpx.ReadTimeout("timeout"), "UPSTREAM_TIMEOUT", None),
        (httpx.ConnectError("unreachable"), "UPSTREAM_CONNECTION_ERROR", None),
        (
            httpx.HTTPStatusError(
                "bad status",
                request=httpx.Request("POST", "http://core"),
                response=httpx.Response(503),
            ),
            "UPSTREAM_HTTP_ERROR",
            "503",
        ),
    ],
)
def test_gateway_exposes_safe_failure_headers_for_upstream_errors(exception, reason, upstream_status):
    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            if isinstance(exception, httpx.HTTPStatusError):
                raise exception
            raise exception

    with (
        patch.object(core_gateway.httpx, "AsyncClient", FakeAsyncClient),
        TestClient(core_gateway.app) as client,
    ):
        response = client.post("/api/v2/legal-query", json={"query": "short test"})

    assert response.status_code == 502
    assert response.headers["x-failure-reason-code"] == reason
    assert response.headers["x-failure-stage"] == "legacy_api"
    assert response.headers.get("x-upstream-status-code") == upstream_status
    assert response.headers["x-request-id"].startswith("trace_legal_")


def test_gateway_withholds_answer_when_citation_integrity_fails():
    evidence_id = "rag:test-1"
    pipeline = {
        "citation_verification": "failed",
        "evidence_ids": [evidence_id],
        "cited_evidence_ids": [],
        "retrieval": "completed",
        "model_api": "error",
        "failure_stage": "model_api",
        "failure_reason_code": "UPSTREAM_TIMEOUT",
    }
    sse = "data: " + json.dumps({"delta": "Unverified legal answer"}) + "\n\n"
    sse += "data: " + json.dumps({
        "status": "done",
        "verifier": "failed",
        "verifier_scope": "citation_integrity_only",
        "pipeline": pipeline,
    }) + "\n\n"

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}
        text = sse

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    with (
        patch.object(core_gateway.httpx, "AsyncClient", FakeAsyncClient),
        TestClient(core_gateway.app) as client,
    ):
        response = client.post("/api/v2/legal-query", json={"query": "short test"})

    data = response.json()
    assert data["status"] == "FAILED"
    assert "lỗi dịch vụ mô hình" in data["answer"]
    assert data["pipeline_execution"]["failure_stage"] == "model_api"
    assert data["pipeline_execution"]["failure_reason_code"] == "UPSTREAM_TIMEOUT"
    assert data["pipeline_execution"]["gateway_citation_failure_reason_code"] == "CITATION_VERIFICATION_NOT_PASSED"
    assert data["pipeline_execution"]["stage_status"]["gateway_citation_validation"] == "failed"
    assert data["pipeline_execution"]["stage_durations_ms"]["gateway_citation_validation"] >= 0


def test_gateway_response_propagates_required_confidence_failure_after_citation_passes():
    evidence_id = "rag:test-1"
    pipeline = {
        "citation_verification": "passed",
        "confidence_status": "not_verified",
        "confidence_scorer_status": "unavailable",
        "confidence_reason": "NO_CALIBRATED_CONFIDENCE_SCORER_CONFIGURED",
        "confidence_score": None,
        "verification_threshold": 0.85,
        "claim_grounding_status": "not_evaluated",
        "grounding_scorer_status": "unavailable",
        "grounding_reason": "NO_GROUNDING_SCORER_CONFIGURED",
        "claim_verification_results": [],
        "verification_status": "not_verified",
        "evidence_ids": [evidence_id],
        "cited_evidence_ids": [evidence_id],
        "failure_stage": "verifier",
        "failure_reason_code": "VERIFICATION_CONFIDENCE_NOT_VERIFIED",
    }
    sse = "data: " + json.dumps({
        "delta": f"Answer [Evidence ID: {evidence_id}]"
    }) + "\n\n"
    sse += "data: " + json.dumps({
        "status": "done",
        "verifier": "not_verified",
        "verifier_scope": "citation_integrity_only",
        "pipeline": pipeline,
    }) + "\n\n"

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}
        text = sse

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    with (
        patch.object(core_gateway.httpx, "AsyncClient", FakeAsyncClient),
        TestClient(core_gateway.app) as client,
    ):
        response = client.post("/api/v2/legal-query", json={"query": "short test"})

    data = response.json()
    pipeline_execution = data["pipeline_execution"]
    assert data["status"] == "FAILED"
    assert data["answer"] == _verification_fallback_message(
        "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
    )
    assert "độ tin cậy" in data["answer"].lower()
    assert pipeline_execution["citation_verification"] == "passed"
    assert pipeline_execution["confidence_status"] == "not_verified"
    assert pipeline_execution["confidence_scorer_status"] == "unavailable"
    assert pipeline_execution["confidence_reason"] == (
        "NO_CALIBRATED_CONFIDENCE_SCORER_CONFIGURED"
    )
    assert pipeline_execution["claim_grounding_status"] == "not_evaluated"
    assert pipeline_execution["grounding_scorer_status"] == "unavailable"
    assert pipeline_execution["grounding_reason"] == "NO_GROUNDING_SCORER_CONFIGURED"
    assert pipeline_execution["failure_reason_code"] == (
        "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
    )
    assert pipeline_execution["gateway_citation_failure_reason_code"] == (
        "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
    )
