"""Conservative policy for showing citation-linked, explicitly unverified answers."""
from __future__ import annotations

import re
from typing import Any


def reference_answer_failure_reason(
    answer: str,
    pipeline: dict[str, Any],
    verifier_status: str,
) -> str | None:
    """Return why an unverified answer cannot be shown as a reference answer.

    This policy does not evaluate legal truth. It permits display only when the
    configured data sufficiency check passed and every answer claim has a valid
    citation to Evidence that was included in the model prompt.
    """
    if pipeline.get("response_status") == "waiting_permission":
        return "REFERENCE_ANSWER_PERMISSION_PENDING"
    if not isinstance(answer, str) or not answer.strip():
        return "REFERENCE_ANSWER_EMPTY"
    if pipeline.get("model_api") != "completed":
        return "REFERENCE_ANSWER_MODEL_INCOMPLETE"
    if pipeline.get("prompt_assembly") != "completed":
        return "REFERENCE_ANSWER_PROMPT_INCOMPLETE"
    if pipeline.get("evidence_builder") != "completed":
        return "REFERENCE_ANSWER_EVIDENCE_BUILDER_INCOMPLETE"
    answer_sufficiency = pipeline.get(
        "answer_data_sufficiency", pipeline.get("enough_data")
    )
    if answer_sufficiency != "sufficient":
        return "REFERENCE_ANSWER_ESSENTIAL_FACTS_NOT_SUFFICIENT"
    required_fields = pipeline.get(
        "answer_required_fields", pipeline.get("required_fields")
    )
    missing_fields = pipeline.get(
        "answer_missing_fields", pipeline.get("missing_fields")
    )
    if not isinstance(required_fields, list) or not isinstance(missing_fields, list):
        return "REFERENCE_ANSWER_DATA_SUFFICIENCY_UNAVAILABLE"
    if missing_fields:
        return "REFERENCE_ANSWER_ESSENTIAL_FACTS_MISSING"

    if pipeline.get("citation_verification") != "passed":
        return "CITATION_VERIFICATION_NOT_PASSED"
    if verifier_status != "not_verified":
        return "VERIFIER_STATUS_NOT_UNVERIFIED"
    if pipeline.get("verification_status") != "not_verified":
        return "VERIFICATION_STATUS_NOT_UNVERIFIED"
    if pipeline.get("confidence_status") not in {
        "not_verified", "not_evaluated", "not_required", "passed",
    }:
        return "VERIFICATION_CONFIDENCE_NOT_ELIGIBLE"
    if pipeline.get("claim_grounding_status") not in {"not_evaluated", "passed"}:
        return "CLAIM_GROUNDING_NOT_ELIGIBLE"
    if not (
        pipeline.get("confidence_status") in {"not_verified", "not_evaluated"}
        or pipeline.get("claim_grounding_status") == "not_evaluated"
    ):
        return "REFERENCE_ANSWER_NO_UNVERIFIED_EVALUATOR"
    if pipeline.get("failure_stage") not in (None, "verifier"):
        return "REFERENCE_ANSWER_PIPELINE_FAILURE"
    if pipeline.get("failure_reason_code") not in (
        None,
        "VERIFICATION_CONFIDENCE_NOT_VERIFIED",
        "CLAIM_GROUNDING_NOT_EVALUATED",
    ):
        return "REFERENCE_ANSWER_EXPLICIT_FAILURE"

    evidence_ids = pipeline.get("evidence_ids")
    cited_ids = pipeline.get("cited_evidence_ids")
    evidence_count = pipeline.get("evidence_count")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        return "EVIDENCE_IDS_MISSING"
    if (
        any(not isinstance(item, str) or not item.strip() for item in evidence_ids)
        or len(set(evidence_ids)) != len(evidence_ids)
        or evidence_count != len(evidence_ids)
    ):
        return "EVIDENCE_IDS_INVALID"
    allowed = set(evidence_ids)
    if not isinstance(cited_ids, list) or not cited_ids:
        return "CITATION_TELEMETRY_MISMATCH"
    if any(not isinstance(item, str) or not item.strip() for item in cited_ids):
        return "EVIDENCE_ID_INVALID"
    cited = set(cited_ids)
    if not cited.issubset(allowed):
        return "EVIDENCE_ID_NOT_ALLOWED"

    answer_citations = {
        match.strip()
        for match in re.findall(
            r"\[Evidence ID:\s*([^\]]+)\]", answer, flags=re.IGNORECASE
        )
        if match.strip()
    }
    if not answer_citations or answer_citations != cited:
        return "CITATION_TELEMETRY_MISMATCH"

    claim_results = pipeline.get("claim_verification_results")
    if not isinstance(claim_results, list) or not claim_results:
        return "CLAIM_CITATION_RESULTS_MISSING"
    linked_ids: set[str] = set()
    for claim in claim_results:
        if not isinstance(claim, dict) or claim.get("passed") is not True:
            return "CLAIM_CITATION_INTEGRITY_FAILED"
        if claim.get("citation_integrity_status") != "passed":
            return "CLAIM_CITATION_INTEGRITY_FAILED"
        if claim.get("claim_grounding_status") in {
            "unsupported", "contradicted", "partially_supported", "failed",
            "insufficient_evidence", "inconclusive", "evaluator_unavailable",
        }:
            return "CLAIM_GROUNDING_NOT_ELIGIBLE"
        claim_ids = claim.get("evidence_ids")
        if not isinstance(claim_ids, list) or not claim_ids:
            return "CLAIM_WITHOUT_EVIDENCE"
        if any(not isinstance(item, str) or item not in allowed for item in claim_ids):
            return "EVIDENCE_ID_NOT_ALLOWED"
        if claim.get("reasons"):
            return "CLAIM_CITATION_INTEGRITY_FAILED"
        linked_ids.update(claim_ids)
    if linked_ids != cited:
        return "CITATION_TELEMETRY_MISMATCH"
    return None


def classify_business_outcome(
    pipeline: dict[str, Any],
    *,
    answer_is_verified: bool,
    answer_is_unverified_reference: bool = False,
) -> str:
    """Classify product outcome separately from the HTTP transport status."""
    if pipeline.get("response_status") == "waiting_permission":
        return "needs_clarification"
    if answer_is_verified:
        return "answered"
    if answer_is_unverified_reference:
        return "answered_unverified"
    if pipeline.get("failure_stage") in {
        "retrieval_pipeline", "evidence_builder", "prompt_assembly", "model_api",
        "model_api_response",
    }:
        return "withheld"
    if (
        pipeline.get("citation_verification") in {"failed", "error"}
        or pipeline.get("verification_status") == "failed"
        or pipeline.get("claim_grounding_status") == "failed"
        or pipeline.get("confidence_status") in {"failed", "invalid_score"}
        or pipeline.get("verifier") == "failed"
        or pipeline.get("verifier_status") == "failed"
        or pipeline.get("gateway_citation_failure_reason_code") in {
            "CITATION_VERIFICATION_NOT_PASSED",
            "CITATION_TELEMETRY_MISMATCH",
            "EVIDENCE_IDS_INVALID",
            "EVIDENCE_ID_NOT_ALLOWED",
            "EVIDENCE_ID_INVALID",
            "CLAIM_GROUNDING_NOT_PASSED",
            "VERIFIER_NOT_PASSED",
            "VERIFICATION_NOT_PASSED",
        }
    ):
        return "verification_failed"
    return "withheld"
