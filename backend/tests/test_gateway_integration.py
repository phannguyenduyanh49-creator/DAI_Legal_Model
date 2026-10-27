import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from backend import core_gateway


def test_gateway_withholds_answer_when_confidence_and_grounding_are_unverified(caplog):
    evidence_id = "rag:procedure-123"
    answer = f"Nộp hồ sơ theo hướng dẫn. [Evidence ID: {evidence_id}]"
    pipeline = {
        "retrieval": "completed",
        "reranking": "completed",
        "evidence_builder": "completed",
        "enough_data": "insufficient",
        "required_fields": ["profile_components", "fees"],
        "missing_fields": ["fees"],
        "external_fallback": "not_allowed",
        "prompt_assembly": "completed",
        "model_api": "completed",
        "citation_verification": "passed",
        "verification_outcome": "not_verified",
        "claim_grounding_status": "not_evaluated",
        "confidence_status": "not_verified",
        "verification_status": "not_verified",
        "citation_attempts": 1,
        "citation_attempt_diagnostics": [{
            "attempt": 1,
            "claim_count": 2,
            "claims_with_citation": 1,
            "cited_evidence_count": 1,
            "allowed_evidence_count": 1,
            "parser_errors": [],
            "verifier_reason_counts": {"CLAIM_WITHOUT_EVIDENCE": 1},
        }],
        "evidence_count": 1,
        "evidence_ids": [evidence_id],
        "evidence_sources": [{
            "evidence_id": evidence_id,
            "source_type": "direct_catalog",
            "source_id": "record-123",
            "title": "Thủ tục đăng ký",
            "source_url": "https://example.gov.vn/procedure/123",
        }],
        "cited_evidence_ids": [evidence_id],
        "model_processing_time_ms": 45.5,
        "model_api_elapsed_ms": 50.0,
        "stage_status": {
            "retrieval": "completed",
            "evidence_builder": "completed",
            "model_api": "completed",
            "verifier": "not_verified",
        },
        "stage_durations_ms": {"retrieval": 18.25, "prompt_assembly": 1.75},
    }
    sse_body = (
        "data: " + json.dumps({"delta": answer}, ensure_ascii=False) + "\n\n"
        "data: " + json.dumps({
            "status": "done",
            "verifier": "not_verified",
            "verifier_scope": "citation_integrity_only",
            "pipeline": pipeline,
        }, ensure_ascii=False) + "\n\n"
    )
    captured = {}

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream; charset=utf-8"}
        text = sse_body

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, json, headers):
            captured.update(url=url, body=json, headers=headers)
            return FakeResponse()

    with (
        patch.object(core_gateway.httpx, "AsyncClient", FakeAsyncClient),
        TestClient(core_gateway.app) as client,
    ):
        response = client.post(
            "/api/v2/legal-query",
            json={"query": "Câu hỏi kiểm thử", "enable_hyde": False},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "FAILED"
    assert data["outcome"] == "withheld"
    assert data["pipeline_execution"]["answer_withheld"] is True
    assert data["pipeline_execution"]["claim_grounding_status"] == "not_evaluated"
    assert data["pipeline_execution"]["verification_outcome"] == "not_verified"
    assert data["pipeline_execution"]["confidence_status"] == "not_verified"
    assert answer not in data["answer"]
    assert data["pipeline_execution"]["evidence_builder"] == "completed"
    assert data["pipeline_execution"]["enough_data"] == "insufficient"
    assert data["pipeline_execution"]["missing_fields"] == ["fees"]
    assert data["pipeline_execution"]["evidence_ids"] == [evidence_id]
    assert data["pipeline_execution"]["evidence_sources"][0]["source_id"] == "record-123"
    assert data["pipeline_execution"]["evidence_sources"][0]["source_url"] == "https://example.gov.vn/procedure/123"
    assert data["pipeline_execution"]["citation_attempt_diagnostics"][0]["claim_count"] == 2
    assert data["pipeline_execution"]["verifier_scope"] == "citation_integrity_only"
    assert data["pipeline_execution"]["model_processing_time_ms"] == 45.5
    assert data["pipeline_execution"]["stage_status"]["retrieval"] == "completed"
    assert data["pipeline_execution"]["stage_durations_ms"]["retrieval"] == 18.25
    assert data["processing_time_ms"] >= 0
    assert float(response.headers["x-gateway-latency-ms"]) >= 0
    assert captured["body"]["prompt"] == "Câu hỏi kiểm thử"
    assert captured["headers"]["X-Request-ID"] == data["request_id"]
    assert captured["body"]["policy_domain"] == "legal_ai"
    assert captured["body"]["mcp_allowed_tools"] == [
        "search_dichvucong",
        "search_legal_documents",
        "search_procedure",
    ]
    assert "xác minh với văn bản" in captured["body"]["system_instruction"]
    assert "Reply tone: formal." in captured["body"]["system_instruction"]
    assert "Language style:" in captured["body"]["system_instruction"]
    assert data["request_id"].startswith("trace_legal_")
    assert "CÃ¢u há»i kiá»ƒm thá»­" not in caplog.text
    assert answer not in caplog.text


def test_gateway_returns_clearly_labeled_reference_answer_when_minimum_safeguards_pass():
    evidence_id = "rag:reference-1"
    answer = f"Nộp hồ sơ theo thủ tục. [Evidence ID: {evidence_id}]"
    pipeline = {
        "response_outcome": "answered_unverified",
        "retrieval": "completed",
        "reranking": "completed",
        "evidence_builder": "completed",
        "enough_data": "sufficient",
        "answer_data_sufficiency": "sufficient",
        "required_fields": ["profile_components"],
        "missing_fields": [],
        "prompt_assembly": "completed",
        "model_api": "completed",
        "citation_verification": "passed",
        "claim_grounding_status": "not_evaluated",
        "confidence_status": "not_verified",
        "confidence_scorer_status": "unavailable",
        "grounding_scorer_status": "unavailable",
        "verification_status": "not_verified",
        "verification_threshold": 0.85,
        "evidence_count": 1,
        "evidence_ids": [evidence_id],
        "cited_evidence_ids": [evidence_id],
        "claim_verification_results": [{
            "claim": "Nộp hồ sơ theo thủ tục.",
            "evidence_ids": [evidence_id],
            "passed": True,
            "citation_integrity_status": "passed",
            "claim_grounding_status": "not_evaluated",
            "reasons": [],
        }],
        "citation_attempts": 1,
        "citation_attempt_diagnostics": [{
            "attempt": 1,
            "claim_count": 1,
            "claims_with_citation": 1,
            "cited_evidence_count": 1,
            "allowed_evidence_count": 1,
            "unknown_evidence_count": 0,
            "parser_errors": [],
        }],
        "failure_stage": "verifier",
        "failure_reason_code": "VERIFICATION_CONFIDENCE_NOT_VERIFIED",
    }
    sse = (
        "data: " + json.dumps({"delta": answer}, ensure_ascii=False) + "\n\n"
        + "data: " + json.dumps({
            "status": "done",
            "outcome": "answered_unverified",
            "verifier": "not_verified",
            "verifier_scope": "citation_integrity_only",
            "pipeline": pipeline,
        }, ensure_ascii=False) + "\n\n"
    )

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream; charset=utf-8"}
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
        response = client.post(
            "/api/v2/legal-query",
            json={"query": "Thủ tục hồ sơ gồm gì?"},
        )

    data = response.json()
    execution = data["pipeline_execution"]
    assert response.status_code == 200
    assert data["status"] == "FAILED"  # legacy execution field remains compatible
    assert data["outcome"] == "answered_unverified"
    assert data["answer"] == answer
    assert execution["answer_withheld"] is False
    assert execution["citation_verification"] == "passed"
    assert execution["claim_grounding_status"] == "not_evaluated"
    assert execution["confidence_status"] == "not_verified"
    assert execution["response_outcome"] == execution["core_response_outcome"]
    assert execution["core_gateway_outcome_match"] is True


def test_gateway_preserves_empty_evidence_failure_from_sse_and_withholds_answer():
    pipeline = {
        "retrieval": "completed",
        "reranking": "completed",
        "evidence_builder": "failed",
        "evidence_count": 0,
        "evidence_ids": [],
        "cited_evidence_ids": [],
        "citation_verification": "not_evaluated",
        "verification_status": "not_verified",
        "confidence_status": "not_evaluated",
        "claim_grounding_status": "not_evaluated",
        "model_api": "not_started",
        "failure_stage": "evidence_builder",
        "failure_reason_code": "EVIDENCE_IDS_MISSING",
        "stage_status": {"model_api": "not_started", "verifier": "not_started"},
    }
    sse_body = (
        "event: chunk\n"
        "data: " + json.dumps({"delta": "Unverified answer must not pass"}) + "\n\n"
        "event: done\n"
        "data: " + json.dumps({
            "status": "done",
            "verifier": "not_verified",
            "verifier_scope": "not_evaluated",
            "pipeline": pipeline,
        }) + "\n\n"
    )

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream; charset=utf-8"}
        text = sse_body

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
        response = client.post(
            "/api/v2/legal-query",
            json={"query": "Question with no returned Evidence"},
        )

    data = response.json()
    execution = data["pipeline_execution"]
    assert response.status_code == 200
    assert data["status"] == "FAILED"
    assert data["outcome"] == "withheld"
    assert "Evidence phù hợp" in data["answer"]
    assert "Unverified answer" not in data["answer"]
    assert execution["evidence_count"] == 0
    assert execution["evidence_ids"] == []
    assert execution["failure_reason_code"] == "EVIDENCE_IDS_MISSING"
    assert execution["gateway_citation_failure_reason_code"] == "EVIDENCE_IDS_MISSING"
    assert execution["model_api"] == "not_started"


def test_gateway_forwards_and_returns_core_session_cookie_for_confirmation():
    session_id = "12345678-1234-5678-1234-567812345678"
    permission_sse = (
        "event: chunk\n"
        'data: {"delta":"Please approve external lookup"}\n\n'
        "event: done\n"
        'data: {"status":"waiting_permission",'
        '"verifier":"not_required",'
        '"pipeline":{"citation_verification":"not_required",'
        '"response_status":"waiting_permission"}}\n\n'
    )
    calls = []

    class FakeResponse:
        status_code = 200
        headers = httpx.Headers({
            "content-type": "text/event-stream; charset=utf-8",
            "set-cookie": f"session_id={session_id}; HttpOnly; Path=/; SameSite=lax",
        })
        text = permission_sse

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, json, headers):
            calls.append({"url": url, "body": json, "headers": headers})
            return FakeResponse()

    with (
        patch.object(core_gateway.httpx, "AsyncClient", FakeAsyncClient),
        TestClient(core_gateway.app) as client,
    ):
        first = client.post(
            "/api/v2/legal-query",
            json={"query": "Original question", "allow_external": False},
            headers={"Origin": "http://127.0.0.1:5175"},
        )
        assert first.status_code == 200
        assert first.json()["outcome"] == "needs_clarification"
        assert client.cookies.get("session_id") == session_id
        assert first.headers["access-control-allow-origin"] == "http://127.0.0.1:5175"
        assert first.headers["access-control-allow-credentials"] == "true"

        second = client.post(
            "/api/v2/legal-query",
            json={"query": "có", "allow_external": False},
            headers={"Origin": "http://127.0.0.1:5175"},
        )
        assert second.status_code == 200

    assert len(calls) == 2
    assert calls[0]["headers"].get("Cookie") is None
    assert calls[1]["headers"]["Cookie"] == f"session_id={session_id}"


def test_gateway_upload_forwards_document_to_registered_domain_only():
    captured = {}

    class FakeResponse:
        status_code = 200
        is_success = True

        def json(self):
            return {
                "status": "indexed",
                "domain_id": "tourism",
                "document_id": "doc-1",
                "title": "fixture.md",
                "chunk_count": 1,
            }

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, url, files):
            captured.update(url=url, files=files)
            return FakeResponse()

    with (
        patch.object(core_gateway.httpx, "AsyncClient", FakeAsyncClient),
        TestClient(core_gateway.app) as client,
    ):
        response = client.post(
            "/api/v2/domains/tourism/documents",
            files={"file": ("fixture.md", b"Pier shuttle leaves at 06:47.", "text/markdown")},
        )
        invalid = client.post(
            "/api/v2/domains/unknown_domain/documents",
            files={"file": ("fixture.md", b"data", "text/markdown")},
        )

    assert response.status_code == 200
    assert response.json()["domain_id"] == "tourism"
    assert captured["url"].endswith("/api/v1/domains/tourism/documents")
    assert captured["files"]["file"][0] == "fixture.md"
    assert invalid.status_code == 404


def test_gateway_rejects_unregistered_domain_hint_before_core_call():
    with TestClient(core_gateway.app) as client:
        response = client.post(
            "/api/v2/legal-query",
            json={"query": "Which route departs?", "domain_hint": "unknown_domain"},
        )

    assert response.status_code == 422
