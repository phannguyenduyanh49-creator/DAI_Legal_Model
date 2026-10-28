"""Fail-closed preflight for requests with no Evidence in the model prompt."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


INSUFFICIENT_EVIDENCE_ANSWER = (
    "Hiện chưa có Evidence phù hợp để kiểm chứng câu trả lời; "
    "hệ thống chưa tạo câu trả lời pháp lý."
)
INSUFFICIENT_GENERAL_EVIDENCE_ANSWER = (
    "Hiện chưa có Evidence phù hợp để hỗ trợ câu trả lời tổng quát; "
    "hệ thống chưa tạo câu trả lời."
)
INSUFFICIENT_TOURISM_EVIDENCE_ANSWER = (
    "Hiện chưa có Evidence du lịch phù hợp; hệ thống chưa tạo câu trả lời."
)


def make_missing_prompt_evidence_result(
    pipeline_state: dict[str, Any],
    *,
    source_candidate_count: int,
    evidence_bundle_count: int,
    prompt_evidences: Sequence[Any],
    domain_id: str = "legal_ai",
) -> dict[str, Any] | None:
    """Prepare a safe Core SSE result when no usable Evidence reaches prompt.

    Returns ``None`` when at least one non-empty, identified Evidence is in
    the prompt. No Evidence ID or answer claim is invented by this gate.
    """
    has_prompt_evidence = any(
        isinstance(getattr(item, "evidence_id", None), str)
        and bool(item.evidence_id.strip())
        and isinstance(getattr(item, "content", None), str)
        and bool(item.content.strip())
        for item in prompt_evidences
    )
    if has_prompt_evidence:
        return None

    if evidence_bundle_count:
        failure_stage = "prompt_assembly"
        pipeline_state["prompt_assembly"] = "failed"
        pipeline_state.setdefault("stage_status", {})["prompt_assembly"] = "failed"
    elif source_candidate_count:
        failure_stage = "evidence_builder"
        pipeline_state["evidence_builder"] = "failed"
        pipeline_state.setdefault("stage_status", {})["evidence_builder"] = "failed"
        pipeline_state["prompt_assembly"] = "not_started"
        pipeline_state.setdefault("stage_status", {})["prompt_assembly"] = "not_started"
    else:
        failure_stage = "retrieval_pipeline"
        pipeline_state["prompt_assembly"] = "not_started"
        pipeline_state.setdefault("stage_status", {})["prompt_assembly"] = "not_started"

    pipeline_state.update(
        {
            "failure_stage": failure_stage,
            "failure_reason_code": "EVIDENCE_IDS_MISSING",
            "evidence_count": 0,
            "evidence_ids": [],
            "cited_evidence_ids": [],
            "citation_attempts": 0,
            "citation_verification": "not_evaluated",
            "verification_status": "not_verified",
            "verifier_scope": "not_evaluated",
            "model_api": "not_started",
            "answer_withheld": True,
        }
    )
    pipeline_state.setdefault("stage_status", {}).update(
        {"model_api": "not_started", "verifier": "not_started"}
    )
    answer = {
        "general_assistant": INSUFFICIENT_GENERAL_EVIDENCE_ANSWER,
        "tourism": INSUFFICIENT_TOURISM_EVIDENCE_ANSWER,
    }.get(domain_id, INSUFFICIENT_EVIDENCE_ANSWER)
    return {
        "answer": answer,
        "verifier": "not_verified",
        "verifier_scope": "not_evaluated",
        "pipeline": pipeline_state,
    }
