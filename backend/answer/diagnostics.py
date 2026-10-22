import json
from collections.abc import Callable
from collections import Counter

import httpx

from backend.answer.schemas import StructuredAnswer
from backend.verifier.verifier import VerificationResult


def classify_upstream_exception(exc: BaseException) -> str:
    """Map known HTTP client failures to stable, non-sensitive reason codes."""
    if isinstance(exc, httpx.TimeoutException):
        return "UPSTREAM_TIMEOUT"
    if isinstance(exc, httpx.HTTPStatusError):
        return "UPSTREAM_HTTP_ERROR"
    if isinstance(exc, httpx.ConnectError):
        return "UPSTREAM_CONNECTION_ERROR"
    return "UNKNOWN_PIPELINE_FAILURE"


def verifier_reason_codes(result: VerificationResult) -> dict[str, int]:
    """Aggregate verifier failures without logging claim or evidence content."""
    counts: Counter[str] = Counter()
    for claim_result in result.claim_results:
        for reason in claim_result.get("reasons", []):
            if reason == "Claim is empty.":
                code = "CLAIM_EMPTY"
            elif reason.startswith("Claim has no evidence:"):
                code = "CLAIM_WITHOUT_EVIDENCE"
            elif reason.startswith("Evidence not found:"):
                code = "EVIDENCE_ID_NOT_FOUND"
            elif reason.startswith("Evidence has empty content:"):
                code = "EVIDENCE_CONTENT_EMPTY"
            else:
                code = "VERIFIER_REJECTED"
            counts[code] += 1
    if result.checked_claims == 0:
        counts["NO_CLAIMS_TO_VERIFY"] += 1
    return dict(sorted(counts.items()))


def make_citation_attempt_logger(
    request_id: str,
    safe_text: Callable[[str], str],
):
    """Build a callback compatible with the citation retry attempt contract."""

    def log_citation_attempt(
        citation_attempt: int,
        answer: str,
        structured_answer: StructuredAnswer,
        verification_result: VerificationResult,
    ) -> None:
        print(
            "[CITATION PARSE]",
            "request_id=", request_id,
            "citation_attempt=", citation_attempt,
            "answer_chars=", len(answer),
            "claim_count=", len(structured_answer.claims),
            "claims_with_evidence=", sum(
                bool(claim.evidence_ids) for claim in structured_answer.claims
            ),
            "cited_evidence_count=", len({
                evidence_id
                for claim in structured_answer.claims
                for evidence_id in claim.evidence_ids
            }),
            "verifier_reason_counts=", json.dumps(
                verifier_reason_codes(verification_result),
                sort_keys=True,
            ),
        )

    return log_citation_attempt
