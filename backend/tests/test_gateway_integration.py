import json
from unittest.mock import patch

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
        "claim_grounding_status": "not_evaluated",
        "confidence_status": "not_verified",
        "verification_status": "not_verified",
        "citation_attempts": 1,
        "evidence_count": 1,
        "evidence_ids": [evidence_id],
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
        headers = {"content-type": "text/event-stream"}
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
    assert data["pipeline_execution"]["answer_withheld"] is True
    assert data["pipeline_execution"]["claim_grounding_status"] == "not_evaluated"
    assert data["pipeline_execution"]["confidence_status"] == "not_verified"
    assert answer not in data["answer"]
    assert data["pipeline_execution"]["evidence_builder"] == "completed"
    assert data["pipeline_execution"]["enough_data"] == "insufficient"
    assert data["pipeline_execution"]["missing_fields"] == ["fees"]
    assert data["pipeline_execution"]["evidence_ids"] == [evidence_id]
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
    assert data["request_id"].startswith("trace_legal_")
    assert "CÃ¢u há»i kiá»ƒm thá»­" not in caplog.text
    assert answer not in caplog.text
