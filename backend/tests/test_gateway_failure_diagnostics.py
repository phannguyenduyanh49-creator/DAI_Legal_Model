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
)


def verified_result(**pipeline_overrides):
    evidence_id = "rag:test-1"
    pipeline = {
        "citation_verification": "passed",
        "evidence_ids": [evidence_id],
        "cited_evidence_ids": [evidence_id],
    }
    pipeline.update(pipeline_overrides)
    return LegacyPipelineResult(
        answer=f"Answer [Evidence ID: {evidence_id}]",
        pipeline=pipeline,
        verifier_status="passed",
    )


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


def test_gateway_passes_only_matching_citation_telemetry():
    assert _citation_integrity_failure_reason(verified_result()) is None


def test_gateway_rejects_citation_telemetry_that_does_not_match_answer():
    result = verified_result()
    result.answer = "Answer without citation"
    assert _citation_integrity_failure_reason(result) == "CITATION_TELEMETRY_MISMATCH"


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
    assert data["answer"] == UNVERIFIED_ANSWER_FALLBACK
    assert data["pipeline_execution"]["failure_stage"] == "model_api"
    assert data["pipeline_execution"]["failure_reason_code"] == "UPSTREAM_TIMEOUT"
    assert data["pipeline_execution"]["gateway_citation_failure_reason_code"] == "CITATION_VERIFICATION_NOT_PASSED"
    assert data["pipeline_execution"]["stage_status"]["gateway_citation_validation"] == "failed"
    assert data["pipeline_execution"]["stage_durations_ms"]["gateway_citation_validation"] >= 0
