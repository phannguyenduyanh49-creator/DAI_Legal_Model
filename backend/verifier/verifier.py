from dataclasses import dataclass, field
from typing import List

from backend.answer.schemas import StructuredAnswer
from backend.evidence.schemas import EvidenceBundle


@dataclass
class VerificationResult:
    # `passed` means citation integrity passed; semantic entailment is not run.
    passed: bool

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

        passed = len(errors) == 0 and checked_claims == linked_claims

        return VerificationResult(
            passed=passed,
            errors=errors,
            checked_claims=checked_claims,
            linked_claims=linked_claims,
            semantic_support_evaluated=False,
            claim_results=claim_results,
        )
