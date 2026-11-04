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

    This policy does not evaluate factual truth. It permits an unverified
    partial reference only when Evidence identity passed, every factual claim
    has a valid citation, and the answer marks requested details not covered by
    the supplied Evidence.
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
    required_fields = pipeline.get(
        "answer_required_fields", pipeline.get("required_fields")
    )
    missing_fields = pipeline.get(
        "answer_missing_fields", pipeline.get("missing_fields")
    )
    if not isinstance(required_fields, list) or not isinstance(missing_fields, list):
        return "REFERENCE_ANSWER_DATA_SUFFICIENCY_UNAVAILABLE"
    partial_evidence = (
        answer_sufficiency == "insufficient"
        and bool(missing_fields)
        and isinstance(pipeline.get("evidence_count"), int)
        and not isinstance(pipeline.get("evidence_count"), bool)
        and pipeline.get("evidence_count", 0) > 0
    )
    if answer_sufficiency != "sufficient" and not partial_evidence:
        return "REFERENCE_ANSWER_ESSENTIAL_FACTS_NOT_SUFFICIENT"
    if missing_fields and not partial_evidence:
        return "REFERENCE_ANSWER_ESSENTIAL_FACTS_MISSING"
    evidence_ids = pipeline.get("evidence_ids")
    evidence_count = pipeline.get("evidence_count")
    if (
        pipeline.get("evidence_integrity_status") != "passed"
        or not isinstance(evidence_ids, list)
        or not evidence_ids
        or any(not isinstance(item, str) or not item.strip() for item in evidence_ids)
        or len(set(evidence_ids)) != len(evidence_ids)
        or evidence_count != len(evidence_ids)
    ):
        return "EVIDENCE_INTEGRITY_NOT_PASSED"

    # A generated answer with citation-link problems may be shown only as an
    # explicitly unverified reference when the exact supplied Evidence bundle
    # passed its independent integrity checks. No source link is trusted here.
    citation_status = pipeline.get("citation_verification")
    if partial_evidence:
        if citation_status != "passed":
            return "PARTIAL_ANSWER_CITATION_INTEGRITY_NOT_PASSED"
        if not re.search(
            r"(?mi)^#{1,6}\s*thông tin nguồn chưa nêu(?:\s*:.*)?\s*$",
            answer,
        ):
            return "PARTIAL_ANSWER_NOT_QUALIFIED"
    if citation_status != "passed":
        allowed_citation_failures = {
            "CITATION_FORMAT_INVALID",
            "CITATION_EVIDENCE_ID_UNKNOWN",
            "CITATION_MISSING",
            "CITATION_VERIFICATION_NOT_PASSED",
        }
        if pipeline.get("evidence_integrity_status") != "passed":
            return "EVIDENCE_INTEGRITY_NOT_PASSED"
        if pipeline.get("failure_reason_code") not in allowed_citation_failures:
            return "CITATION_FAILURE_NOT_REPAIRABLE"
        if pipeline.get("confidence_status") in {"failed", "invalid_score"}:
            return "VERIFICATION_CONFIDENCE_NOT_ELIGIBLE"
        if pipeline.get("claim_grounding_status") in {
            "failed", "insufficient_evidence", "unsupported", "contradicted",
        }:
            return "CLAIM_GROUNDING_NOT_ELIGIBLE"
        if pipeline.get("failure_stage") != "verifier":
            return "REFERENCE_ANSWER_PIPELINE_FAILURE"
        claim_results = pipeline.get("claim_verification_results")
        if not isinstance(claim_results, list) or not claim_results:
            return "CLAIM_CITATION_RESULTS_MISSING"
        allowed_reasons = (
            "Claim has no evidence:",
            "Evidence not found:",
        )
        for claim in claim_results:
            if not isinstance(claim, dict) or not isinstance(claim.get("claim"), str) or not claim["claim"].strip():
                return "CLAIM_CITATION_INTEGRITY_FAILED"
            if claim.get("claim_grounding_status") in {
                "unsupported", "contradicted", "partially_supported", "failed",
                "insufficient_evidence",
            }:
                return "CLAIM_GROUNDING_NOT_ELIGIBLE"
            reasons = claim.get("reasons")
            if not isinstance(reasons, list) or any(
                not isinstance(reason, str)
                or not reason.startswith(allowed_reasons)
                for reason in reasons
            ):
                return "CLAIM_CITATION_INTEGRITY_FAILED"
        return None

    if verifier_status != "not_verified":
        return "VERIFIER_STATUS_NOT_UNVERIFIED"
    if pipeline.get("verification_status") != "not_verified":
        return "VERIFICATION_STATUS_NOT_UNVERIFIED"
    if pipeline.get("confidence_status") not in {
        "not_verified", "not_evaluated", "not_required", "passed",
    }:
        return "VERIFICATION_CONFIDENCE_NOT_ELIGIBLE"
    if pipeline.get("claim_grounding_status") not in {
        "not_evaluated", "passed", "inconclusive", "evaluator_unavailable"
    }:
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
        "GROUNDING_EVALUATOR_UNAVAILABLE",
        "GROUNDING_EVALUATION_INCONCLUSIVE",
    ):
        return "REFERENCE_ANSWER_EXPLICIT_FAILURE"

    cited_ids = pipeline.get("cited_evidence_ids")
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
            "insufficient_evidence",
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
