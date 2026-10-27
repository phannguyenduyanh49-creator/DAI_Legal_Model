"""Typed interfaces for calibrated answer confidence and claim grounding.

No production scorer is configured in this repository. Implementations must
be independently evaluated on labeled legal data before deployment.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence

from backend.answer.schemas import Claim, StructuredAnswer
from backend.evidence.schemas import Evidence, EvidenceBundle


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Calibrated answer-level estimate emitted by a trusted scorer."""

    score: float
    scorer_id: str


@dataclass(frozen=True)
class GroundingAssessment:
    """Assessment of one claim against only its cited Evidence content."""

    status: str
    reason_code: str
    scorer_id: str


class ConfidenceScorer(Protocol):
    def score(
        self,
        answer: StructuredAnswer,
        evidence_bundle: EvidenceBundle,
        *,
        timeout_seconds: float,
    ) -> ConfidenceAssessment: ...


class GroundingScorer(Protocol):
    def evaluate(
        self,
        claim: Claim,
        cited_evidence: Sequence[Evidence],
        *,
        timeout_seconds: float,
    ) -> GroundingAssessment: ...


class ScorerUnavailableError(RuntimeError):
    """Raised by a scorer adapter when its model/service is unavailable."""

    def __init__(self, message: str = "scorer unavailable", reason_code: str | None = None):
        super().__init__(message)
        self.reason_code = reason_code


GROUNDING_STATUSES = frozenset({
    "supported",
    "unsupported",
    "contradicted",
    "partially_supported",
    "insufficient_evidence",
    "inconclusive",
    "evaluator_unavailable",
})
