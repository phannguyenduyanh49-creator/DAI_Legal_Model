from pathlib import Path

from benchmark.run_benchmark import load_dataset, observe_response, summarize, write_json


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_dataset_has_fixed_group_sizes_and_explicit_mcp_permission():
    dataset, cases = load_dataset(ROOT / "benchmark" / "dataset" / "benchmark_questions.json")

    assert dataset["status"] == "candidate_unreviewed"
    assert len(cases) == 30
    assert sum(case["group"] == "local_rag" for case in cases) == 10
    assert sum(case["group"] == "mcp_fallback" for case in cases) == 10
    assert sum(case["group"] == "insufficient_evidence" for case in cases) == 10
    assert all(case["allow_external"] is True for case in cases if case["group"] == "mcp_fallback")
    assert all(case["allow_external"] is False for case in cases if case["group"] != "mcp_fallback")


def test_observation_preserves_gateway_verification_fields_without_calling_them_quality():
    response = {
        "request_id": "trace-test",
        "outcome": "answered_unverified",
        "status": "FAILED",
        "routing_and_planning": {"selected_domain": "legal_ai"},
        "pipeline_execution": {
            "evidence_ids": ["rag:source-1"],
            "evidence_count": 1,
            "citation_verification": "passed",
            "verification_status": "not_verified",
            "claim_grounding_status": "not_evaluated",
            "stage_durations_ms": {"retrieval": 12.0},
            "model_input_tokens": 1024,
            "model_output_tokens": 256,
            "model_finish_reason": "max_new_tokens",
            "model_prompt_preparation_ms": 8.5,
        },
    }

    observed = observe_response(response)

    assert observed["business_outcome"] == "answered_unverified"
    assert observed["citation_integrity"] == "passed"
    assert observed["verification_status"] == "not_verified"
    assert observed["grounding_status"] == "not_evaluated"
    assert observed["evidence_ids"] == ["rag:source-1"]
    assert observed["model_input_tokens"] == 1024
    assert observed["model_output_tokens"] == 256
    assert observed["model_finish_reason"] == "max_new_tokens"
    assert observed["model_prompt_preparation_ms"] == 8.5


def test_observation_keeps_mcp_results_and_top_level_error_message():
    observed = observe_response({
        "status": "FAILED",
        "error": "upstream unavailable",
        "pipeline_execution": {
            "mcp_status": "failed",
            "mcp_results": [{"url": "https://example.test"}],
        },
    })

    assert observed["error_message"] == "upstream unavailable"
    assert observed["mcp_status"] == "failed"
    assert observed["mcp_results"] == [{"url": "https://example.test"}]


def test_run_summary_separates_transport_success_from_evidence_observations():
    rows = [
        {"group": "local_rag", "latency_ms": 100.0, "http_status": 200,
         "transport_success": True, "transport_error": None,
         "observed": {"business_outcome": "answered_unverified", "evidence_ids": ["e1"],
                      "citation_integrity": "passed", "verifier_status": "not_verified"}},
        {"group": "local_rag", "latency_ms": 200.0, "http_status": 500,
         "transport_success": False, "transport_error": None,
         "observed": {"business_outcome": "withheld", "evidence_ids": [],
                      "citation_integrity": "failed", "verifier_status": "failed"}},
    ]

    summary = summarize(rows)

    assert summary["request_count"] == 2
    assert summary["transport_success_count"] == 1
    assert summary["http_error_count"] == 1
    assert summary["by_group"]["local_rag"]["evidence_present_count"] == 1
    assert summary["quality_metrics"].startswith("unavailable")


def test_run_summary_censors_timeouts_and_does_not_call_unverified_a_business_success():
    rows = [
        {
            "group": "local_rag", "latency_ms": 120050.0,
            "latency_censored_ms": 120000.0, "http_status": None,
            "http_success": False, "transport_success": False,
            "timed_out": True, "transport_error": "TimeoutError: timed out",
            "observed": {"business_outcome": None, "evidence_ids": []},
        },
        {
            "group": "local_rag", "latency_ms": 900.0,
            "latency_censored_ms": 900.0, "http_status": 200,
            "http_success": True, "transport_success": True,
            "timed_out": False, "transport_error": None,
            "observed": {
                "business_outcome": "answered_unverified",
                "evidence_ids": ["e1"], "citation_integrity": "passed",
                "verification_status": "not_verified",
            },
        },
    ]

    summary = summarize(rows)

    assert summary["timeout_count"] == 1
    assert summary["latency_ms"]["max"] == 120000.0
    assert summary["business_success_count"] == 0
    assert summary["answered_unverified_count"] == 1


def test_benchmark_json_artifacts_preserve_vietnamese_utf8(tmp_path):
    output = tmp_path / "raw.json"
    write_json(output, {"query": "Thủ tục đăng ký thành lập công ty cổ phần"})

    raw = output.read_text(encoding="utf-8")
    assert "Thủ tục đăng ký thành lập công ty cổ phần" in raw
    assert "\\u1ee7" not in raw
