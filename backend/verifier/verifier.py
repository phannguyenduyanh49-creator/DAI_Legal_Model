from collections import Counter
from dataclasses import dataclass, field
import math
from typing import List

from backend.answer.schemas import Claim, StructuredAnswer
from backend.evidence.schemas import Evidence, EvidenceBundle
from backend.verifier.scoring import (
    ConfidenceAssessment,
    ConfidenceScorer,
    GROUNDING_STATUSES,
    GroundingAssessment,
    GroundingScorer,
    ScorerUnavailableError,
)


@dataclass
class VerificationResult:
    # Overall pass requires valid citations, all configured checks evaluated,
    # confidence above threshold, and every claim semantically supported.
    passed: bool
    citation_integrity_passed: bool = False
    claim_grounding_status: str = "not_evaluated"
    confidence_score: float | None = None
    confidence_threshold: float | None = None
    confidence_status: str = "not_required"
    confidence_reason: str | None = None
    confidence_scorer_status: str = "not_configured"
    confidence_scorer_id: str | None = None
    grounding_reason: str = "NO_GROUNDING_SCORER_CONFIGURED"
    grounding_scorer_status: str = "not_configured"
    grounding_scorer_id: str | None = None
    errors: List[str] = field(default_factory=list)
    checked_claims: int = 0
    linked_claims: int = 0
    semantic_support_evaluated: bool = False
    claim_results: List[dict] = field(default_factory=list)
    # Product response outcomes (answered_unverified/withheld) are assigned by
    # pipeline policy. This field describes the verifier decision only. Kept at
    # the end to preserve the positional order of the existing result fields.
    outcome: str = "not_verified"


class Verifier:
    """Fail-closed citation, confidence and semantic grounding coordinator.

    Confidence scores are accepted only from an injected ConfidenceScorer.
    Grounding scorers see the exact non-empty Evidence objects named by each
    claim's citations. Citation validity alone never implies semantic support.
    """

    def __init__(
        self,
        confidence_scorer: ConfidenceScorer | None = None,
        grounding_scorer: GroundingScorer | None = None,
        scorer_timeout_seconds: float = 10.0,
    ) -> None:
        if (
            isinstance(scorer_timeout_seconds, bool)
            or not isinstance(scorer_timeout_seconds, (int, float))
            or not math.isfinite(float(scorer_timeout_seconds))
            or scorer_timeout_seconds <= 0
        ):
            raise ValueError("scorer_timeout_seconds must be finite and greater than zero")
        self.confidence_scorer = confidence_scorer
        self.grounding_scorer = grounding_scorer
        self.scorer_timeout_seconds = scorer_timeout_seconds

    def verify(
        self,
        answer: StructuredAnswer,
        evidence_bundle: EvidenceBundle,
        minimum_confidence: float | None = None,
        *,
        expected_domain_id: str | None = None,
    ) -> VerificationResult:
        errors: list[str] = []
        citation_parse_errors = answer.metadata.get("citation_parse_errors", [])
        for parse_error in citation_parse_errors:
            errors.append(f"Citation parse error: {parse_error}")

        evidence_id_counts = Counter(
            evidence.evidence_id for evidence in evidence_bundle.evidences
        )
        duplicate_evidence_ids = {
            evidence_id for evidence_id, count in evidence_id_counts.items()
            if count > 1
        }
        evidence_map = {
            evidence.evidence_id: evidence
            for evidence in evidence_bundle.evidences
            if evidence_id_counts[evidence.evidence_id] == 1
        }
        checked_claims = len(answer.claims)
        linked_claims = 0
        claim_results: list[dict] = []
        resolved_by_claim: list[list[Evidence]] = []

        if checked_claims == 0:
            errors.append("Answer contains no claims to validate.")

        for claim in answer.claims:
            claim_errors: list[str] = []
            cited_evidence: list[Evidence] = []

            if not claim.claim.strip():
                claim_errors.append("Claim is empty.")
            elif not claim.evidence_ids:
                claim_errors.append(f"Claim has no evidence: {claim.claim}")

            for evidence_id in claim.evidence_ids:
                if evidence_id in duplicate_evidence_ids:
                    claim_errors.append(f"Evidence ID is ambiguous: {evidence_id}")
                    continue
                evidence = evidence_map.get(evidence_id)
                if evidence is None:
                    claim_errors.append(f"Evidence not found: {evidence_id}")
                elif not isinstance(evidence.content, str) or not evidence.content.strip():
                    claim_errors.append(f"Evidence has empty content: {evidence_id}")
                else:
                    evidence_domain_id = None
                    if isinstance(evidence.metadata, dict):
                        evidence_domain_id = evidence.metadata.get("domain_id")
                    if evidence_domain_id is None and isinstance(evidence.fields, dict):
                        evidence_domain_id = evidence.fields.get("domain_id")
                    if (
                        expected_domain_id
                        and evidence_domain_id is not None
                        and evidence_domain_id != expected_domain_id
                    ):
                        claim_errors.append(
                            f"Evidence domain mismatch: {evidence_id}"
                        )
                        continue
                    cited_evidence.append(evidence)

            for message in claim_errors:
                errors.append(message)
            claim_linked = bool(cited_evidence) and not claim_errors
            if claim_linked:
                linked_claims += 1
            resolved_by_claim.append(cited_evidence if claim_linked else [])
            claim_results.append({
                "claim": claim.claim,
                "evidence_ids": list(claim.evidence_ids),
                # Backward-compatible alias: `passed` continues to mean
                # citation linkage. `claim_verified` is the full claim result.
                "passed": claim_linked,
                "citation_integrity_passed": claim_linked,
                "claim_verified": False,
                "citation_integrity_status": "passed" if claim_linked else "failed",
                "claim_grounding_status": "not_evaluated",
                "grounding_reason": "NO_GROUNDING_SCORER_CONFIGURED",
                "reasons": claim_errors,
            })

        citation_integrity_passed = (
            not errors and checked_claims > 0 and checked_claims == linked_claims
        )

        if citation_parse_errors:
            citation_integrity_passed = False

        confidence_score = None
        confidence_status = "not_required"
        confidence_reason = None
        confidence_scorer_status = "not_required"
        confidence_scorer_id = None
        confidence_passed = True

        threshold_valid = (
            minimum_confidence is None
            or (
                not isinstance(minimum_confidence, bool)
                and isinstance(minimum_confidence, (int, float))
                and math.isfinite(float(minimum_confidence))
                and 0.0 <= float(minimum_confidence) <= 1.0
            )
        )
        if not threshold_valid:
            confidence_status = "invalid_threshold"
            confidence_scorer_status = "not_run"
            confidence_passed = False
            confidence_reason = "THRESHOLD_MUST_BE_FINITE_AND_BETWEEN_0_AND_1"
            errors.append("Confidence threshold must be a finite value between 0 and 1.")
        elif minimum_confidence is not None:
            confidence_passed = False
            if self.confidence_scorer is None:
                confidence_status = "not_verified"
                confidence_scorer_status = "unavailable"
                confidence_reason = "NO_CALIBRATED_CONFIDENCE_SCORER_CONFIGURED"
            elif not citation_integrity_passed:
                confidence_status = "not_verified"
                confidence_scorer_status = "not_run"
                confidence_reason = "CITATION_INTEGRITY_NOT_PASSED"
            else:
                try:
                    assessment = self.confidence_scorer.score(
                        answer,
                        evidence_bundle,
                        timeout_seconds=self.scorer_timeout_seconds,
                    )
                    if not isinstance(assessment, ConfidenceAssessment):
                        raise ValueError("malformed confidence assessment")
                    confidence_scorer_id = assessment.scorer_id
                    score = assessment.score
                    if (
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                        or not 0.0 <= float(score) <= 1.0
                        or not isinstance(assessment.scorer_id, str)
                        or not assessment.scorer_id.strip()
                    ):
                        confidence_status = "invalid_score"
                        confidence_scorer_status = "invalid_output"
                        confidence_reason = "CONFIDENCE_SCORER_RETURNED_INVALID_OUTPUT"
                        errors.append("Confidence scorer returned invalid output.")
                    else:
                        confidence_score = float(score)
                        confidence_scorer_status = "completed"
                        confidence_status = (
                            "passed"
                            if confidence_score >= float(minimum_confidence)
                            else "failed"
                        )
                        confidence_passed = confidence_status == "passed"
                        if not confidence_passed:
                            confidence_reason = "SCORE_BELOW_CONFIGURED_THRESHOLD"
                except TimeoutError:
                    confidence_status = "not_verified"
                    confidence_scorer_status = "timeout"
                    confidence_reason = "CONFIDENCE_SCORER_TIMEOUT"
                except ScorerUnavailableError as exc:
                    confidence_status = "not_verified"
                    confidence_scorer_status = "unavailable"
                    confidence_reason = (
                        exc.reason_code or "CONFIDENCE_SCORER_UNAVAILABLE"
                    )
                except ValueError:
                    confidence_status = "invalid_score"
                    confidence_scorer_status = "invalid_output"
                    confidence_reason = "CONFIDENCE_SCORER_RETURNED_INVALID_OUTPUT"
                    errors.append("Confidence scorer returned invalid output.")
                except Exception:
                    confidence_status = "not_verified"
                    confidence_scorer_status = "error"
                    confidence_reason = "CONFIDENCE_SCORER_ERROR"

        grounding_scorer_status = "unavailable"
        grounding_scorer_id = None
        grounding_reason = "NO_GROUNDING_SCORER_CONFIGURED"
        grounding_status = "not_evaluated"
        semantic_support_evaluated = False

        if self.grounding_scorer is not None and citation_integrity_passed:
            assessments: list[str] = []
            scorer_failed = False
            grounding_scorer_status = "completed"
            grounding_reason = "ALL_CLAIMS_EVALUATED"
            for claim, cited_evidence, claim_result in zip(
                answer.claims, resolved_by_claim, claim_results
            ):
                try:
                    assessment = self.grounding_scorer.evaluate(
                        claim,
                        tuple(cited_evidence),
                        timeout_seconds=self.scorer_timeout_seconds,
                    )
                    if (
                        not isinstance(assessment, GroundingAssessment)
                        or assessment.status not in GROUNDING_STATUSES
                        or not isinstance(assessment.reason_code, str)
                        or not assessment.reason_code.strip()
                        or not isinstance(assessment.scorer_id, str)
                        or not assessment.scorer_id.strip()
                    ):
                        raise ValueError("malformed grounding assessment")
                    grounding_scorer_id = assessment.scorer_id
                    assessments.append(assessment.status)
                    claim_result["claim_grounding_status"] = assessment.status
                    claim_result["grounding_reason"] = assessment.reason_code
                    claim_result["grounding_scorer_id"] = assessment.scorer_id
                    claim_result["claim_verified"] = (
                        claim_result["citation_integrity_status"] == "passed"
                        and assessment.status == "supported"
                    )
                    if assessment.status == "evaluator_unavailable":
                        scorer_failed = True
                        grounding_scorer_status = "unavailable"
                        grounding_reason = assessment.reason_code
                        semantic_support_evaluated = False
                        break
                    if assessment.status == "inconclusive":
                        grounding_reason = "GROUNDING_EVALUATION_INCONCLUSIVE"
                except TimeoutError:
                    scorer_failed = True
                    grounding_scorer_status = "timeout"
                    grounding_reason = "GROUNDING_SCORER_TIMEOUT"
                    claim_result["claim_grounding_status"] = "evaluator_unavailable"
                    claim_result["grounding_reason"] = grounding_reason
                    break
                except ScorerUnavailableError as exc:
                    scorer_failed = True
                    grounding_scorer_status = "unavailable"
                    grounding_reason = (
                        exc.reason_code or "GROUNDING_SCORER_UNAVAILABLE"
                    )
                    claim_result["claim_grounding_status"] = "evaluator_unavailable"
                    claim_result["grounding_reason"] = grounding_reason
                    break
                except ValueError:
                    scorer_failed = True
                    grounding_scorer_status = "invalid_output"
                    grounding_reason = "GROUNDING_SCORER_INVALID_OUTPUT"
                    claim_result["claim_grounding_status"] = "evaluator_unavailable"
                    claim_result["grounding_reason"] = grounding_reason
                    break
                except Exception:
                    scorer_failed = True
                    grounding_scorer_status = "error"
                    grounding_reason = "GROUNDING_SCORER_ERROR_OR_INVALID_OUTPUT"
                    claim_result["claim_grounding_status"] = "evaluator_unavailable"
                    claim_result["grounding_reason"] = grounding_reason
                    break

            if scorer_failed:
                for claim_result in claim_results:
                    if claim_result["claim_grounding_status"] == "not_evaluated":
                        claim_result["claim_grounding_status"] = "evaluator_unavailable"
                        claim_result["grounding_reason"] = grounding_reason

            if not scorer_failed and len(assessments) == checked_claims:
                semantic_support_evaluated = True
                if all(status == "supported" for status in assessments):
                    grounding_status = "passed"
                elif "inconclusive" in assessments:
                    grounding_status = "inconclusive"
                elif "insufficient_evidence" in assessments:
                    grounding_status = "insufficient_evidence"
                else:
                    grounding_status = "failed"
                if grounding_status not in {"passed", "inconclusive"}:
                    grounding_reason = "ONE_OR_MORE_CLAIMS_NOT_FULLY_SUPPORTED"
                for claim_result in claim_results:
                    claim_result["claim_verified"] = (
                        claim_result["citation_integrity_status"] == "passed"
                        and claim_result["claim_grounding_status"] == "supported"
                    )
        elif self.grounding_scorer is not None:
            grounding_scorer_status = "not_run"
            grounding_reason = "CITATION_INTEGRITY_NOT_PASSED"

        passed = (
            citation_integrity_passed
            and confidence_passed
            and grounding_status == "passed"
            and semantic_support_evaluated
        )
        outcome = (
            "verified" if passed
            else "failed" if (
                not citation_integrity_passed
                or confidence_status in {"failed", "invalid_threshold"}
                or grounding_status in {"failed", "insufficient_evidence"}
            )
            else "inconclusive" if grounding_status == "inconclusive"
            else "evaluator_unavailable" if (
                confidence_scorer_status in {
                    "unavailable", "timeout", "error", "invalid_output"
                }
                or confidence_status == "invalid_score"
                or grounding_scorer_status in {
                    "not_configured", "unavailable", "timeout", "error",
                    "invalid_output",
                }
            )
            else "not_verified"
        )

        return VerificationResult(
            passed=passed,
            outcome=outcome,
            citation_integrity_passed=citation_integrity_passed,
            claim_grounding_status=grounding_status,
            confidence_score=confidence_score,
            confidence_threshold=minimum_confidence,
            confidence_status=confidence_status,
            confidence_reason=confidence_reason,
            confidence_scorer_status=confidence_scorer_status,
            confidence_scorer_id=confidence_scorer_id,
            grounding_reason=grounding_reason,
            grounding_scorer_status=grounding_scorer_status,
            grounding_scorer_id=grounding_scorer_id,
            errors=errors,
            checked_claims=checked_claims,
            linked_claims=linked_claims,
            semantic_support_evaluated=semantic_support_evaluated,
            claim_results=claim_results,
        )
