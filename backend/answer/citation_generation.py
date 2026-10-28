from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from backend.answer.citations import structure_answer_with_explicit_evidence
from backend.answer.citation_contract import CITATION_RULES, format_allowed_evidence_ids
from backend.answer.diagnostics import verifier_reason_codes
from backend.answer.schemas import StructuredAnswer
from backend.evidence.schemas import EvidenceBundle
from backend.verifier.verifier import VerificationResult, Verifier


MAX_CITATION_ATTEMPTS = 2


@dataclass
class CitationGenerationResult:
    answer: str | None
    structured_answer: StructuredAnswer
    verification: VerificationResult
    attempts: int
    last_model_response: dict[str, Any]
    model_processing_time_ms: float | None
    reference_answer: str | None = None
    generation_error: str | None = None
    failure_reason_code: str | None = None
    attempt_diagnostics: list[dict[str, Any]] = field(default_factory=list)


GenerateCallback = Callable[[str, int], Awaitable[dict[str, Any]]]
AttemptCallback = Callable[[int, str, StructuredAnswer, VerificationResult], None]


def _retry_prompt(prompt: str, allowed_evidence_ids: tuple[str, ...]) -> str:
    allowed_ids = format_allowed_evidence_ids(allowed_evidence_ids)
    return (
        prompt
        + "\n\n=== CITATION REPAIR ===\n"
        + "Regenerate the answer under the same citation requirements. "
        + CITATION_RULES
        + " Use only these IDs: "
        + allowed_ids
        + ". Never invent a source, law, or ID. If evidence is insufficient, say so "
        + "without making unsupported legal claims."
    )


def _citation_failure_code(structured: StructuredAnswer, verification: VerificationResult) -> str:
    parse_errors = structured.metadata.get("citation_parse_errors", [])
    if parse_errors:
        return "CITATION_FORMAT_INVALID"
    if any("Evidence not found:" in error for error in verification.errors):
        return "CITATION_EVIDENCE_ID_UNKNOWN"
    if any("Claim has no evidence:" in error for error in verification.errors):
        return "CITATION_MISSING"
    if verification.checked_claims == 0:
        return "CITATION_NO_CLAIMS"
    return "CITATION_VERIFICATION_NOT_PASSED"


async def generate_with_citation_retry(
    prompt: str,
    evidence_bundle: EvidenceBundle,
    generate: GenerateCallback,
    verifier: Verifier,
    max_attempts: int = MAX_CITATION_ATTEMPTS,
    on_attempt: AttemptCallback | None = None,
    raw_answer_debug_enabled: bool = False,
) -> CitationGenerationResult:
    """Retry only a generated response whose citation integrity can be repaired.

    ``evidence_bundle`` must contain exactly the evidence included in the
    assembled prompt. The runtime caller constructs this bundle from
    ``PromptAssembler.included_evidences`` before invoking this function.
    """
    attempts_limit = max(1, min(int(max_attempts), MAX_CITATION_ATTEMPTS))
    allowed_evidence_ids = tuple(item.evidence_id for item in evidence_bundle.evidences)
    allowed_evidence_id_set = set(allowed_evidence_ids)
    current_prompt = prompt
    last_response: dict[str, Any] = {}
    model_processing_total = 0.0
    has_model_processing_time = False
    last_generation_error = None
    failure_reason_code = None
    diagnostics: list[dict[str, Any]] = []
    last_structured = structure_answer_with_explicit_evidence("")
    last_verification = verifier.verify(last_structured, evidence_bundle)

    for attempt in range(1, attempts_limit + 1):
        try:
            response = await generate(current_prompt, attempt)
        except Exception as exc:
            # A model transport/runtime error is reported immediately. A
            # second expensive generation cannot repair the failed request.
            last_generation_error = type(exc).__name__
            failure_reason_code = "MODEL_API_REQUEST_FAILED"
            diagnostics.append({
                "attempt": attempt,
                "raw_answer_chars": 0,
                "claim_count": 0,
                "claims_with_citation": 0,
                "cited_evidence_count": 0,
                "allowed_evidence_count": len(allowed_evidence_ids),
                "unknown_evidence_ids": [],
                "parser_errors": [],
                "verifier_reason_counts": {},
                "citation_integrity_passed": False,
                "claim_grounding_status": "not_evaluated",
                "confidence_status": "not_verified",
                "raw_answer_debug_enabled": raw_answer_debug_enabled,
                "failure_reason_code": failure_reason_code,
            })
            if on_attempt is not None:
                on_attempt(attempt, "", last_structured, last_verification)
            break

        last_response = response if isinstance(response, dict) else {}
        processing_time = last_response.get("processing_time_ms")
        if isinstance(processing_time, (int, float)) and not isinstance(processing_time, bool):
            model_processing_total += float(processing_time)
            has_model_processing_time = True

        raw_answer = last_response.get("answer")
        answer = (
            raw_answer
            if last_response.get("success") is True
            and isinstance(raw_answer, str)
            and raw_answer.strip()
            else ""
        )
        if not answer:
            failure_reason_code = (
                "MODEL_API_EMPTY_RESPONSE"
                if last_response.get("success") is True
                else "MODEL_API_GENERATION_FAILED"
            )
            last_generation_error = failure_reason_code
            diagnostics.append({
                "attempt": attempt,
                "raw_answer_chars": 0,
                "claim_count": 0,
                "claims_with_citation": 0,
                "cited_evidence_count": 0,
                "allowed_evidence_count": len(allowed_evidence_ids),
                "unknown_evidence_ids": [],
                "parser_errors": [],
                "verifier_reason_counts": {},
                "citation_integrity_passed": False,
                "claim_grounding_status": "not_evaluated",
                "confidence_status": "not_verified",
                "raw_answer_debug_enabled": raw_answer_debug_enabled,
                "failure_reason_code": failure_reason_code,
            })
            if on_attempt is not None:
                on_attempt(attempt, "", last_structured, last_verification)
            break

        last_generation_error = None
        last_structured = structure_answer_with_explicit_evidence(answer)
        last_verification = verifier.verify(last_structured, evidence_bundle)
        attempt_debug = {
            "attempt": attempt,
            "raw_answer_chars": len(answer),
            "claim_count": last_verification.checked_claims,
            "claims_with_citation": sum(bool(claim.evidence_ids) for claim in last_structured.claims),
            "cited_evidence_count": len({
                evidence_id
                for claim in last_structured.claims
                for evidence_id in claim.evidence_ids
            }),
            "allowed_evidence_count": len(allowed_evidence_ids),
            "unknown_evidence_ids": sorted({
                evidence_id
                for claim in last_structured.claims
                for evidence_id in claim.evidence_ids
                if evidence_id not in allowed_evidence_id_set
            }),
            "parser_errors": list(last_structured.metadata.get("citation_parse_errors", [])),
            "verifier_reason_counts": verifier_reason_codes(last_verification),
            "citation_integrity_passed": last_verification.citation_integrity_passed,
            "claim_grounding_status": last_verification.claim_grounding_status,
            "confidence_status": last_verification.confidence_status,
            "raw_answer_debug_enabled": raw_answer_debug_enabled,
        }
        diagnostics.append(attempt_debug)
        if on_attempt is not None:
            on_attempt(attempt, answer, last_structured, last_verification)

        if last_verification.passed:
            return CitationGenerationResult(
                answer=answer,
                structured_answer=last_structured,
                verification=last_verification,
                attempts=attempt,
                last_model_response=last_response,
                model_processing_time_ms=model_processing_total if has_model_processing_time else None,
                attempt_diagnostics=diagnostics,
            )

        # Missing/invalid citations may be repaired by regeneration. A valid
        # citation set failing confidence or semantic grounding cannot.
        if last_verification.citation_integrity_passed:
            failure_reason_code = (
                "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
                if last_verification.confidence_status == "not_verified"
                else "CLAIM_GROUNDING_NOT_EVALUATED"
                if last_verification.claim_grounding_status == "not_evaluated"
                else "VERIFIER_NOT_PASSED"
            )
            break

        failure_reason_code = _citation_failure_code(last_structured, last_verification)
        if attempt < attempts_limit and allowed_evidence_ids:
            current_prompt = _retry_prompt(prompt, allowed_evidence_ids)
        else:
            break

    return CitationGenerationResult(
        answer=None,
        reference_answer=(
            last_structured.answer
            if (
                last_verification.citation_integrity_passed
                and last_verification.confidence_status not in ("failed", "invalid_score")
                and last_verification.claim_grounding_status != "failed"
            )
            else None
        ),
        structured_answer=last_structured,
        verification=last_verification,
        attempts=len(diagnostics) if diagnostics else (1 if last_generation_error else 0),
        last_model_response=last_response,
        model_processing_time_ms=model_processing_total if has_model_processing_time else None,
        generation_error=last_generation_error,
        failure_reason_code=failure_reason_code,
        attempt_diagnostics=diagnostics,
    )
