from dataclasses import dataclass, field
import math
from typing import List

from backend.answer.schemas import StructuredAnswer
from backend.evidence.schemas import EvidenceBundle

CLAIM_GROUNDING_EVALUATED = False


@dataclass
class VerificationResult:
    # `passed` requires citation integrity, any configured confidence threshold,
    # and claim grounding. This implementation does not yet evaluate grounding.
    passed: bool
    citation_integrity_passed: bool = False
    claim_grounding_status: str = "not_evaluated"
    confidence_score: float | None = None
    confidence_threshold: float | None = None
    confidence_status: str = "not_required"
    confidence_reason: str | None = None
    grounding_reason: str = "NO_GROUNDING_SCORER_CONFIGURED"

    errors: List[str] = field(
        default_factory=list
    )

    checked_claims: int = 0

    linked_claims: int = 0

    semantic_support_evaluated: bool = False

    claim_results: List[dict] = field(default_factory=list)


class Verifier:

    def verify(
        self,
        answer: StructuredAnswer,
        evidence_bundle: EvidenceBundle,
        minimum_confidence: float | None = None,
        confidence_score: float | None = None,
    ) -> VerificationResult:

        errors = []

        evidence_map = {
            evidence.evidence_id: evidence
            for evidence in evidence_bundle.evidences
        }

        checked_claims = len(
            answer.claims
        )

        linked_claims = 0
        claim_results = []

        if checked_claims == 0:
            errors.append("Answer contains no claims to validate.")

        for claim in answer.claims:

            claim_errors = []

            if not claim.claim.strip():
                message = "Claim is empty."
                errors.append(message)
                claim_errors.append(message)
                claim_results.append({
                    "claim": claim.claim,
                    "evidence_ids": list(claim.evidence_ids),
                    "passed": False,
                    "reasons": claim_errors,
                })
                continue

            if not claim.evidence_ids:

                message = f"Claim has no evidence: {claim.claim}"
                errors.append(message)
                claim_errors.append(message)
                claim_results.append({
                    "claim": claim.claim,
                    "evidence_ids": list(claim.evidence_ids),
                    "passed": False,
                    "reasons": claim_errors,
                })

                continue

            claim_linked = False

            for evidence_id in claim.evidence_ids:

                evidence = evidence_map.get(
                    evidence_id
                )

                if evidence is None:

                    message = f"Evidence not found: {evidence_id}"
                    errors.append(message)
                    claim_errors.append(message)

                    continue

                if not evidence.content.strip():

                    message = f"Evidence has empty content: {evidence_id}"
                    errors.append(message)
                    claim_errors.append(message)

                    continue

                claim_linked = True

            if claim_linked:
                linked_claims += 1

            claim_results.append({
                "claim": claim.claim,
                "evidence_ids": list(claim.evidence_ids),
                "passed": claim_linked and not claim_errors,
                "reasons": claim_errors,
            })

        citation_integrity_passed = len(errors) == 0 and checked_claims == linked_claims
        confidence_reason = None
        if minimum_confidence is None:
            confidence_status = "not_required"
            confidence_passed = True
        elif (
            isinstance(minimum_confidence, bool)
            or not isinstance(minimum_confidence, (int, float))
            or not math.isfinite(float(minimum_confidence))
            or not 0.0 <= float(minimum_confidence) <= 1.0
        ):
            confidence_status = "invalid_threshold"
            confidence_passed = False
            confidence_reason = "THRESHOLD_MUST_BE_FINITE_AND_BETWEEN_0_AND_1"
            errors.append("Confidence threshold must be a finite value between 0 and 1.")
        elif confidence_score is None:
            confidence_status = "not_verified"
            confidence_passed = False
            confidence_reason = "NO_CALIBRATED_CONFIDENCE_SCORER_CONFIGURED"
        elif (
            isinstance(confidence_score, bool)
            or not isinstance(confidence_score, (int, float))
            or not math.isfinite(float(confidence_score))
            or not 0.0 <= float(confidence_score) <= 1.0
        ):
            confidence_status = "invalid_score"
            confidence_passed = False
            errors.append("Confidence score must be a finite value between 0 and 1.")
        else:
            confidence_status = (
                "passed"
                if float(confidence_score) >= minimum_confidence
                else "failed"
            )
            confidence_passed = confidence_status == "passed"
            confidence_reason = (
                None
                if confidence_passed
                else "SCORE_BELOW_CONFIGURED_THRESHOLD"
            )
        # The current implementation can establish citation integrity only.
        # A linked ID is not proof that the Evidence entails the claim.
        passed = (
            citation_integrity_passed
            and confidence_passed
            and CLAIM_GROUNDING_EVALUATED
        )

        for claim_result in claim_results:
            claim_result["citation_integrity_status"] = (
                "passed" if claim_result["passed"] else "failed"
            )
            claim_result["claim_grounding_status"] = "not_evaluated"
            claim_result["grounding_reason"] = "NO_GROUNDING_SCORER_CONFIGURED"

        return VerificationResult(
            passed=passed,
            citation_integrity_passed=citation_integrity_passed,
            claim_grounding_status="not_evaluated",
            confidence_score=confidence_score,
            confidence_threshold=minimum_confidence,
            confidence_status=confidence_status,
            confidence_reason=confidence_reason,
            errors=errors,
            checked_claims=checked_claims,
            linked_claims=linked_claims,
            semantic_support_evaluated=CLAIM_GROUNDING_EVALUATED,
            claim_results=claim_results,
        )
