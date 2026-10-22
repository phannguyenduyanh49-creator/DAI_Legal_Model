import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import core_gateway


def test_gateway_proxies_core_sse_and_preserves_pipeline_telemetry(caplog):
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
            "verifier": "passed",
        },
        "stage_durations_ms": {"retrieval": 18.25, "prompt_assembly": 1.75},
    }
    sse_body = (
        "data: " + json.dumps({"delta": answer}, ensure_ascii=False) + "\n\n"
        "data: " + json.dumps({
            "status": "done",
            "verifier": "passed",
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
    assert data["status"] == "SUCCESS"
    assert data["answer"] == answer
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
    assert data["request_id"].startswith("trace_legal_")
    assert "CÃ¢u há»i kiá»ƒm thá»­" not in caplog.text
    assert answer not in caplog.text
