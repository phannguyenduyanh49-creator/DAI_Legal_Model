
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "benchmark"
    / "dataset"
    / "benchmark_questions.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "benchmark"
    / "results"
)

DEFAULT_BASE_URL = "http://127.0.0.1:8002"

EXPECTED_GROUPS = {
    "local_rag": 10,
    "mcp_fallback": 10,
    "insufficient_evidence": 10,
}

API_ENDPOINT = "/api/v2/legal-query"


# ============================================================
# General utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(data, ensure_ascii=False) + "\n"
        )


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return round(values[0], 3)

    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    result = (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )

    return round(result, 3)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None

    return round(statistics.mean(values), 3)


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None

    return round(statistics.median(values), 3)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# Dataset loading and validation
# ============================================================

def load_dataset(path: Path) -> tuple[dict, list[dict]]:
    """
    Return:
        dataset: original JSON object, including metadata.
        cases: list of benchmark cases.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError(
            "Dataset root must be a JSON object."
        )

    if dataset.get("dataset_id") != "dai-legal-e2e-30-v1":
        raise ValueError(
            "Unexpected dataset_id."
        )

    if dataset.get("status") != "candidate_unreviewed":
        raise ValueError(
            "Unexpected dataset status."
        )

    cases = dataset.get("cases")

    if not isinstance(cases, list):
        raise ValueError(
            "Dataset field 'cases' must be a list."
        )

    if len(cases) != 30:
        raise ValueError(
            f"Expected exactly 30 cases, found {len(cases)}."
        )

    group_counts = Counter()
    seen_ids = set()

    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be a JSON object.")

        case_id = case.get("case_id")

        if not case_id:
            raise ValueError("A benchmark case has no case_id.")

        if case_id in seen_ids:
            raise ValueError(
                f"Duplicate case_id: {case_id}"
            )

        seen_ids.add(case_id)

        if not case.get("query"):
            raise ValueError(
                f"Missing query in case {case_id}."
            )

        group = case.get("group")

        if group not in EXPECTED_GROUPS:
            raise ValueError(
                f"Unknown group '{group}' in case {case_id}."
            )

        group_counts[group] += 1

        if not isinstance(case.get("allow_external"), bool):
            raise ValueError(
                f"allow_external must be boolean in {case_id}."
            )

        expected_permission = group == "mcp_fallback"

        if case["allow_external"] != expected_permission:
            raise ValueError(
                f"Incorrect allow_external permission in {case_id}."
            )

    for group, expected_count in EXPECTED_GROUPS.items():
        actual_count = group_counts[group]

        if actual_count != expected_count:
            raise ValueError(
                f"Group {group}: expected {expected_count}, "
                f"found {actual_count}."
            )

    return dataset, cases


# ============================================================
# Gateway response observation
# ============================================================

def observe_response(response: dict) -> dict:
    """
    Extract observations from the Gateway response.

    These fields describe what the system reported.
    They do not automatically establish answer quality.
    """

    if not isinstance(response, dict):
        response = {}

    pipeline = response.get("pipeline_execution") or {}

    if not isinstance(pipeline, dict):
        pipeline = {}

    routing = response.get("routing_and_planning") or {}

    if not isinstance(routing, dict):
        routing = {}

    evidence_ids = pipeline.get("evidence_ids") or []

    if not isinstance(evidence_ids, list):
        evidence_ids = []

    citations = pipeline.get("citations") or []

    if not isinstance(citations, list):
        citations = []

    stage_durations = pipeline.get("stage_durations_ms") or {}

    if not isinstance(stage_durations, dict):
        stage_durations = {}

    citation_integrity = pipeline.get(
        "citation_verification",
        pipeline.get("citation_integrity"),
    )

    verification_status = pipeline.get(
        "verification_status",
        pipeline.get("verifier_status"),
    )

    grounding_status = pipeline.get(
        "claim_grounding_status",
        pipeline.get("grounding_status"),
    )

    error_message = response.get("detail")
    if isinstance(error_message, dict):
        error_message = error_message.get("message") or json.dumps(
            error_message, ensure_ascii=False
        )
    if not isinstance(error_message, str):
        error_message = pipeline.get("error_message")
    if not isinstance(error_message, str):
        error_message = response.get("error") or response.get("message")
    if not isinstance(error_message, str):
        error_message = None

    return {
        "request_id": response.get("request_id"),

        "business_outcome": response.get(
            "outcome",
            response.get("status"),
        ),

        "response_status": response.get("status"),

        "selected_domain": routing.get("selected_domain"),

        "evidence_ids": evidence_ids,

        "evidence_count": pipeline.get(
            "evidence_count",
            len(evidence_ids),
        ),

        "citations": citations,

        "citation_integrity": citation_integrity,

        "verification_status": verification_status,

        "verifier_status": verification_status,

        "grounding_status": grounding_status,

        "failure_stage": pipeline.get("failure_stage"),

        "failure_reason_code": pipeline.get("failure_reason_code"),

        "exception_type": pipeline.get("exception_type"),

        "error_message": error_message,

        "stage_durations_ms": stage_durations,

        "model_input_tokens": pipeline.get("model_input_tokens"),
        "model_output_tokens": pipeline.get("model_output_tokens"),
        "model_finish_reason": pipeline.get("model_finish_reason"),
        "model_prompt_preparation_ms": pipeline.get("model_prompt_preparation_ms"),

        "mcp_invoked": pipeline.get(
            "mcp_invoked",
            pipeline.get("external_search_invoked"),
        ),

        "mcp_status": pipeline.get(
            "mcp_status",
            pipeline.get("external_search_status"),
        ),

        "mcp_raw_result_count": pipeline.get("mcp_raw_result_count"),

        "mcp_mapped_evidence_count": pipeline.get(
            "mcp_mapped_evidence_count"
        ),

        # Preserve MCP result metadata when the Gateway exposes it. The
        # benchmark records it as observed telemetry; it is not a quality
        # judgment and may legitimately be absent.
        "mcp_results": pipeline.get(
            "mcp_results",
            pipeline.get("external_search_results"),
        ),
    }


def is_timeout_error(error: str | None) -> bool:
    """Classify client-side timeout failures without conflating them with HTTP errors."""
    return isinstance(error, str) and "timeout" in error.lower()


# ============================================================
# HTTP requests
# ============================================================

def json_request(
    url: str,
    payload: dict | None = None,
    timeout: int = 60,
) -> tuple[int | None, dict | None, str | None]:
    """
    Return:
        HTTP status, parsed JSON, transport error.
    """

    data = None
    headers = {
        "Accept": "application/json",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:

            status = response.status
            raw_body = response.read().decode(
                "utf-8",
                errors="replace",
            )

            try:
                parsed = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed = {
                    "raw_response": raw_body
                }

            return status, parsed, None

    except urllib.error.HTTPError as error:
        raw_body = error.read().decode(
            "utf-8",
            errors="replace",
        )

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            parsed = {
                "raw_response": raw_body
            }

        return error.code, parsed, None

    except Exception as error:
        return None, None, (
            f"{type(error).__name__}: {error}"
        )


# ============================================================
# Service preflight
# ============================================================

def check_services(
    base_url: str,
    timeout: int = 10,
) -> dict:

    checks = {}

    endpoints = {
        "gateway": f"{base_url.rstrip('/')}/health",
        "core": "http://127.0.0.1:8000/",
        "core_readiness": (
            "http://127.0.0.1:8000/api/v2/search/readiness"
        ),
        "model": "http://127.0.0.1:8001/",
    }

    for name, url in endpoints.items():
        status, body, error = json_request(
            url,
            timeout=timeout,
        )

        checks[name] = {
            "url": url,
            "http_status": status,
            "reachable": (
                status is not None
                and 200 <= status < 300
            ),
            "error": error,
            "response": body,
        }

    failed = [
        name
        for name, result in checks.items()
        if not result["reachable"]
    ]

    return {
        "checked_at": utc_now(),
        "checks": checks,
        "all_reachable": len(failed) == 0,
        "failed_services": failed,
    }


# ============================================================
# Benchmark request
# ============================================================

def request_case(
    base_url: str,
    case: dict,
    repeat: int,
    timeout: int,
) -> dict:

    payload = {
        "query": case["query"],
        "domain_hint": case.get("domain_hint"),
        "allow_external": case["allow_external"],
        "enable_hyde": True,
        "temperature": 0.2,
        "response_style": "default",
    }

    url = base_url.rstrip("/") + API_ENDPOINT

    started_at = time.perf_counter()

    http_status, response, transport_error = json_request(
        url,
        payload=payload,
        timeout=timeout,
    )

    finished_at = time.perf_counter()

    latency_ms = (finished_at - started_at) * 1000

    # HTTP and transport are intentionally separate. An HTTP 4xx/5xx response
    # completed transport successfully, but is not an HTTP success.
    transport_success = transport_error is None and http_status is not None
    http_success = (
        http_status is not None
        and 200 <= http_status < 300
    )
    timed_out = is_timeout_error(transport_error)

    observed = observe_response(
        response if isinstance(response, dict) else {}
    )

    return {
        "case_id": case["case_id"],
        "group": case["group"],
        "repeat": repeat,

        "run_number": repeat,

        "timestamp_utc": utc_now(),

        "latency_ms": round(latency_ms, 3),

        # Timeout observations are right-censored at the configured client
        # timeout for latency distribution statistics. The observed wall-clock
        # value remains in latency_ms for forensic analysis.
        "latency_censored_ms": round(
            float(timeout) * 1000 if timed_out else latency_ms,
            3,
        ),

        "http_status": http_status,

        "transport_error": transport_error,

        "transport_success": transport_success,

        "http_success": http_success,

        "timed_out": timed_out,

        "query": case["query"],

        "domain_hint": case.get("domain_hint"),

        "allow_external": case["allow_external"],

        "request_payload": payload,

        "observed": observed,

        "raw_response": response,
    }


# ============================================================
# Summary
# ============================================================

def summarize(rows: list[dict]) -> dict:
    def rate(count: int, total: int) -> float | None:
        return round(count / total * 100, 2) if total else None

    def latency_summary(selected_rows: list[dict]) -> dict:
        # Use censored values for timeouts: observed elapsed values are kept in
        # raw output, while aggregate latency cannot imply a completed answer.
        values = [
            float(row.get("latency_censored_ms", row["latency_ms"]))
            for row in selected_rows
            if row.get("latency_ms") is not None
        ]
        return {
            "interpretation": "timeouts right-censored at configured timeout",
            "mean": safe_mean(values), "median": safe_median(values),
            "p90": percentile(values, 0.90), "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "min": round(min(values), 3) if values else None,
            "max": round(max(values), 3) if values else None,
        }

    def metrics(selected_rows: list[dict]) -> dict:
        total = len(selected_rows)
        observed = [row.get("observed") or {} for row in selected_rows]
        outcomes = Counter(item.get("business_outcome") for item in observed)
        citation_statuses = Counter(item.get("citation_integrity") for item in observed)
        verifier_statuses = Counter(item.get("verification_status") for item in observed)
        grounding_statuses = Counter(item.get("grounding_status") for item in observed)
        evidence_present = sum(bool(item.get("evidence_ids")) for item in observed)
        def row_http_success(row: dict) -> bool:
            explicit = row.get("http_success")
            if isinstance(explicit, bool):
                return explicit
            status = row.get("http_status")
            return isinstance(status, int) and 200 <= status < 300

        http_success = sum(row_http_success(row) for row in selected_rows)
        transport_success = sum(row.get("transport_success") is True for row in selected_rows)
        timed_out = sum(row.get("timed_out") is True for row in selected_rows)
        # Only `answered` is a completed verified legal answer. The reference
        # response status `answered_unverified` remains reported separately.
        business_success = outcomes.get("answered", 0)
        unverified_reference = outcomes.get("answered_unverified", 0)
        citation_denominator = sum(
            item.get("citation_integrity") is not None for item in observed
        )
        verification_denominator = sum(
            item.get("verification_status") is not None for item in observed
        )
        mcp_invoked = sum(item.get("mcp_invoked") is True for item in observed)
        mcp_completed = sum(item.get("mcp_status") == "completed" for item in observed)
        return {
            "request_count": total,
            "http_success_count": http_success,
            "http_success_rate_pct": rate(http_success, total),
            "transport_success_count": transport_success,
            "transport_success_rate_pct": rate(transport_success, total),
            "timeout_count": timed_out,
            "timeout_rate_pct": rate(timed_out, total),
            "http_error_count": sum(
                row.get("http_status") is not None and not row_http_success(row)
                for row in selected_rows
            ),
            "transport_error_count": sum(
                row.get("transport_error") is not None for row in selected_rows
            ),
            "business_success_definition": "outcome == answered (verified legal answer)",
            "business_success_count": business_success,
            "business_success_rate_pct": rate(business_success, total),
            "answered_unverified_count": unverified_reference,
            "business_outcomes": dict(outcomes),
            "latency_ms": latency_summary(selected_rows),
            "evidence_present_count": evidence_present,
            "evidence_presence_rate_pct": rate(evidence_present, total),
            "citation_integrity_statuses": dict(citation_statuses),
            "citation_integrity_pass_rate_pct": rate(
                citation_statuses.get("passed", 0), citation_denominator
            ),
            "verification_statuses": dict(verifier_statuses),
            "verification_pass_rate_pct": rate(
                verifier_statuses.get("passed", 0), verification_denominator
            ),
            "grounding_statuses": dict(grounding_statuses),
            "mcp_invoked_count": mcp_invoked,
            "mcp_invocation_rate_pct": rate(mcp_invoked, total),
            "mcp_completed_count": mcp_completed,
            "mcp_completed_rate_pct": rate(mcp_completed, total),
        }

    result = metrics(rows)
    result["by_group"] = {
        group: metrics([row for row in rows if row.get("group") == group])
        for group in sorted({row.get("group", "unknown") for row in rows})
    }

    # A successful HTTP response does not prove answer quality.
    # Ground-truth relevance labels have not been independently reviewed.
    quality_metrics = (
        "unavailable: no independently reviewed relevance "
        "labels or answer-quality ground truth."
    )

    result["generated_at"] = utc_now()
    result["quality_metrics"] = quality_metrics
    return result


# ============================================================
# CSV export
# ============================================================

def write_csv(path: Path, rows: list[dict]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "case_id",
        "group",
        "repeat",
        "run_number",
        "timestamp_utc",
        "latency_ms",
        "latency_censored_ms",
        "http_status",
        "http_success",
        "transport_success",
        "timed_out",
        "transport_error",
        "allow_external",
        "business_outcome",
        "request_id",
        "selected_domain",
        "evidence_count",
        "evidence_ids",
        "citation_integrity",
        "verification_status",
        "grounding_status",
        "failure_stage",
        "failure_reason_code",
        "exception_type",
        "error_message",
        "mcp_invoked",
        "mcp_status",
        "stage_durations_ms",
        "model_input_tokens",
        "model_output_tokens",
        "model_finish_reason",
        "model_prompt_preparation_ms",
        "mcp_results",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:
            observed = row.get("observed") or {}

            writer.writerow({
                "case_id": row.get("case_id"),
                "group": row.get("group"),
                "repeat": row.get("repeat"),
                "run_number": row.get("run_number"),
                "timestamp_utc": row.get("timestamp_utc"),
                "latency_ms": row.get("latency_ms"),
                "latency_censored_ms": row.get("latency_censored_ms"),
                "http_status": row.get("http_status"),
                "http_success": row.get("http_success"),
                "transport_success": row.get("transport_success"),
                "timed_out": row.get("timed_out"),
                "transport_error": row.get("transport_error"),
                "allow_external": row.get("allow_external"),
                "business_outcome": observed.get("business_outcome"),
                "request_id": observed.get("request_id"),
                "selected_domain": observed.get("selected_domain"),
                "evidence_count": observed.get("evidence_count"),
                "evidence_ids": json.dumps(
                    observed.get("evidence_ids", []),
                    ensure_ascii=False,
                ),
                "citation_integrity": observed.get("citation_integrity"),
                "verification_status": observed.get("verification_status"),
                "grounding_status": observed.get("grounding_status"),
                "failure_stage": observed.get("failure_stage"),
                "failure_reason_code": observed.get("failure_reason_code"),
                "exception_type": observed.get("exception_type"),
                "error_message": observed.get("error_message"),
                "mcp_invoked": observed.get("mcp_invoked"),
                "mcp_status": observed.get("mcp_status"),
                "stage_durations_ms": json.dumps(
                    observed.get("stage_durations_ms", {}),
                    ensure_ascii=False,
                ),
                "model_input_tokens": observed.get("model_input_tokens"),
                "model_output_tokens": observed.get("model_output_tokens"),
                "model_finish_reason": observed.get("model_finish_reason"),
                "model_prompt_preparation_ms": observed.get("model_prompt_preparation_ms"),
                "mcp_results": json.dumps(
                    observed.get("mcp_results"),
                    ensure_ascii=False,
                ),
            })


def write_benchmark_report(path: Path, metadata: dict, summary: dict) -> None:
    """Write a factual run report; quality metrics stay unavailable without gold labels."""
    lines = [
        "# DAI Legal Model E2E Benchmark Report",
        "",
        "## Experimental setup",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Dataset: `{metadata['dataset_id']}` ({metadata['dataset_status']})",
        f"- Measured requests: {summary['request_count']} (planned: {metadata['request_count_planned']})",
        f"- Client timeout: {metadata['timeout_seconds']} seconds",
        "",
        "## Reliability",
        f"- HTTP success: {summary['http_success_rate_pct']}% ({summary['http_success_count']})",
        f"- Transport success: {summary['transport_success_rate_pct']}% ({summary['transport_success_count']})",
        f"- Timeout: {summary['timeout_rate_pct']}% ({summary['timeout_count']})",
        f"- Completed verified legal answers: {summary['business_success_rate_pct']}% ({summary['business_success_count']})",
        f"- `answered_unverified`: {summary['answered_unverified_count']}",
        "",
        "## Latency",
        "Timeouts are right-censored at the configured client timeout in aggregate latency metrics.",
        f"- Mean / median / P95 (ms): {summary['latency_ms']['mean']} / {summary['latency_ms']['median']} / {summary['latency_ms']['p95']}",
        "",
        "## Evidence, citations and verifier",
        f"- Evidence present: {summary['evidence_presence_rate_pct']}% ({summary['evidence_present_count']})",
        f"- Citation integrity pass rate: {summary['citation_integrity_pass_rate_pct']}%",
        f"- Verification pass rate: {summary['verification_pass_rate_pct']}%",
        f"- Citation statuses: `{json.dumps(summary['citation_integrity_statuses'], ensure_ascii=False)}`",
        f"- Verification statuses: `{json.dumps(summary['verification_statuses'], ensure_ascii=False)}`",
        f"- Grounding statuses: `{json.dumps(summary['grounding_statuses'], ensure_ascii=False)}`",
        "",
        "## MCP evaluation",
        f"- Invoked: {summary['mcp_invocation_rate_pct']}% ({summary['mcp_invoked_count']})",
        f"- Completed: {summary['mcp_completed_rate_pct']}% ({summary['mcp_completed_count']})",
        "",
        "## Results by group",
        "| Group | Requests | HTTP success | Timeout | Verified answers | Evidence present |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, result in summary['by_group'].items():
        lines.append(
            f"| {group} | {result['request_count']} | {result['http_success_rate_pct']}% | "
            f"{result['timeout_rate_pct']}% | {result['business_success_rate_pct']}% | "
            f"{result['evidence_presence_rate_pct']}% |"
        )
    lines.extend([
        "",
        "## Limitations",
        f"- {summary['quality_metrics']}",
        "- A 2xx response, `needs_clarification`, `verification_failed`, and `answered_unverified` are not treated as completed verified legal answers.",
        "- Raw responses and request-level diagnostics are retained in `raw_responses.jsonl`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_charts(run_dir: Path, rows: list[dict], summary: dict) -> list[str]:
    """Create requested PNGs when matplotlib is available; never fabricate a chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; PNG artifacts not created"]

    warnings: list[str] = []
    latencies = [
        float(row.get("latency_censored_ms", row["latency_ms"]))
        for row in rows if row.get("latency_ms") is not None
    ]
    if latencies:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(latencies, bins=min(20, max(1, len(latencies))))
        ax.set(title="End-to-end latency (timeouts censored)", xlabel="Latency (ms)", ylabel="Requests")
        fig.tight_layout()
        fig.savefig(run_dir / "latency_distribution.png", dpi=150)
        plt.close(fig)

    stages: dict[str, list[float]] = {}
    for row in rows:
        for name, value in (row.get("observed") or {}).get("stage_durations_ms", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stages.setdefault(name, []).append(float(value))
    if stages:
        names = sorted(stages)
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 4.5))
        ax.bar(names, [statistics.mean(stages[name]) for name in names])
        ax.tick_params(axis="x", rotation=45)
        ax.set(title="Mean stage latency", ylabel="Milliseconds")
        fig.tight_layout()
        fig.savefig(run_dir / "stage_latency.png", dpi=150)
        plt.close(fig)

    reliability = {
        "HTTP success": summary["http_success_rate_pct"],
        "Transport success": summary["transport_success_rate_pct"],
        "Verified answer": summary["business_success_rate_pct"],
        "Citation pass": summary["citation_integrity_pass_rate_pct"],
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(list(reliability), [value or 0 for value in reliability.values()])
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    ax.set(title="Reliability metrics", ylabel="Percent")
    fig.tight_layout()
    fig.savefig(run_dir / "reliability.png", dpi=150)
    plt.close(fig)
    return warnings


# ============================================================
# Main benchmark execution
# ============================================================

def run(args: argparse.Namespace) -> Path:

    dataset_path = Path(args.dataset)

    dataset, cases = load_dataset(dataset_path)

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")

        cases = cases[:args.limit]

    print("\n[1/5] Dataset validation")

    print(f"Dataset ID : {dataset['dataset_id']}")
    print(f"Status     : {dataset['status']}")
    print(f"Cases      : {len(cases)}")

    print("\n[2/5] Service preflight")

    preflight = check_services(
        args.base_url,
        timeout=min(args.timeout, 15),
    )

    for name, result in preflight["checks"].items():
        status = result["http_status"]
        state = "OK" if result["reachable"] else "FAILED"

        print(f"{name:18} {state:8} HTTP={status}")

    if not preflight["all_reachable"]:
        raise RuntimeError(
            "Preflight failed. Start and verify all services "
            "before running the benchmark."
        )

    print("\n[3/5] Creating run directory")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = (
        Path(args.output_dir)
        / f"current_system_{timestamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = {
        "run_id": run_dir.name,
        "started_at": utc_now(),
        "dataset_id": dataset.get("dataset_id"),
        "dataset_status": dataset.get("status"),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "base_url": args.base_url,
        "endpoint": API_ENDPOINT,
        "repetitions": args.runs,
        "warmup_count": args.warmup,
        "case_count": len(cases),
        "request_count_planned": len(cases) * args.runs,
        "timeout_seconds": args.timeout,
        "preflight": preflight,
        "python_version": sys.version,
    }

    write_json(
        run_dir / "run_metadata.json",
        metadata,
    )

    print(f"Run directory: {run_dir}")

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("\n[4/5] Warm-up")

    warmup_rows = []

    for index in range(args.warmup):
        case = cases[index % len(cases)]

        print(
            f"Warm-up {index + 1}/{args.warmup}: "
            f"{case['case_id']}"
        )

        row = request_case(
            args.base_url,
            case,
            repeat=0,
            timeout=args.timeout,
        )

        warmup_rows.append(row)

        append_jsonl(
            run_dir / "warmups.jsonl",
            row,
        )

    # --------------------------------------------------------
    # Measured requests
    # --------------------------------------------------------

    print("\n[5/5] Measured benchmark")

    rows = []

    total = len(cases) * args.runs
    completed = 0

    for repeat in range(1, args.runs + 1):

        for case in cases:

            completed += 1

            print(
                f"[{completed}/{total}] "
                f"{case['case_id']} "
                f"({case['group']})",
                end=" ",
                flush=True,
            )

            row = request_case(
                args.base_url,
                case,
                repeat=repeat,
                timeout=args.timeout,
            )

            rows.append(row)

            append_jsonl(
                run_dir / "raw_responses.jsonl",
                row,
            )

            print(
                f"HTTP={row['http_status']} "
                f"latency={row['latency_ms']}ms"
            )

    # --------------------------------------------------------
    # Final outputs
    # --------------------------------------------------------

    summary = summarize(rows)

    summary["run_id"] = run_dir.name
    summary["dataset_id"] = dataset.get("dataset_id")
    summary["planned_request_count"] = total
    summary["warmup_request_count"] = len(warmup_rows)

    write_json(
        run_dir / "summary.json",
        summary,
    )

    write_csv(
        run_dir / "request_metrics.csv",
        rows,
    )

    write_benchmark_report(
        run_dir / "benchmark_report.md",
        metadata,
        summary,
    )
    chart_warnings = write_charts(run_dir, rows, summary)

    metadata["finished_at"] = utc_now()
    metadata["request_count_actual"] = len(rows)
    metadata["chart_warnings"] = chart_warnings

    write_json(
        run_dir / "run_metadata.json",
        metadata,
    )

    print("\n========================================")
    print("BENCHMARK COMPLETED")
    print("========================================")

    print(f"Run ID              : {run_dir.name}")
    print(f"Requests            : {summary['request_count']}")
    print(f"HTTP successes      : {summary['http_success_count']}")
    print(f"Transport successes : {summary['transport_success_count']}")
    print(f"Timeouts            : {summary['timeout_count']}")
    print(f"Verified answers    : {summary['business_success_count']}")
    print(f"HTTP errors         : {summary['http_error_count']}")
    print(f"Transport errors    : {summary['transport_error_count']}")

    print("\nLatency (ms)")

    for key, value in summary["latency_ms"].items():
        print(f"{key:10}: {value}")

    print("\nOutput files:")
    print(f"  {run_dir / 'run_metadata.json'}")
    print(f"  {run_dir / 'warmups.jsonl'}")
    print(f"  {run_dir / 'raw_responses.jsonl'}")
    print(f"  {run_dir / 'request_metrics.csv'}")
    print(f"  {run_dir / 'summary.json'}")
    print(f"  {run_dir / 'benchmark_report.md'}")

    return run_dir


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "End-to-end benchmark for the current "
            "DAI Legal Model system."
        )
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured repetitions per case.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up requests.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of cases for a smoke test.",
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")

    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative.")

    if args.timeout < 1:
        raise SystemExit("--timeout must be positive.")

    try:
        run(args)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        raise SystemExit(130)

    except Exception as error:
        print(
            f"\nBENCHMARK FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
