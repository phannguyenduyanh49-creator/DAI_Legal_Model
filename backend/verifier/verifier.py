from dataclasses import dataclass, field
from typing import List

from backend.answer.schemas import StructuredAnswer
from backend.evidence.schemas import EvidenceBundle


@dataclass
class VerificationResult:
    passed: bool

    errors: List[str] = field(
        default_factory=list
    )

    checked_claims: int = 0

    supported_claims: int = 0


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

        supported_claims = 0

        for claim in answer.claims:

            if not claim.claim.strip():
                errors.append(
                    "Claim is empty."
                )
                continue

            if not claim.evidence_ids:

                errors.append(
                    f"Claim has no evidence: "
                    f"{claim.claim}"
                )

                continue

            claim_supported = False

            for evidence_id in claim.evidence_ids:

                evidence = evidence_map.get(
                    evidence_id
                )

                if evidence is None:

                    errors.append(
                        f"Evidence not found: "
                        f"{evidence_id}"
                    )

                    continue

                if not evidence.content.strip():

                    errors.append(
                        f"Evidence has empty content: "
                        f"{evidence_id}"
                    )

                    continue

                claim_supported = True

            if claim_supported:
                supported_claims += 1

            else:
                errors.append(
                    f"Claim is not supported: "
                    f"{claim.claim}"
                )

        passed = (
            len(errors) == 0
            and checked_claims
            == supported_claims
        )

        return VerificationResult(
            passed=passed,
            errors=errors,
            checked_claims=checked_claims,
            supported_claims=supported_claims,
        )