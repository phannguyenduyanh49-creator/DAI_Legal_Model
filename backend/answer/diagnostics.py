import json
import re
from collections.abc import Callable
from collections import Counter
from typing import Any

import httpx

from backend.answer.schemas import StructuredAnswer
from backend.verifier.verifier import VerificationResult

_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:bearer\s+)[A-Za-z0-9._~+/=-]+|"
    r"(?:api[_-]?key|access[_-]?token|token|secret|password)\s*[:=]\s*[^\s,;]+|"
    r"\bsk-[A-Za-z0-9_-]{12,}\b|"
    r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?84|0)(?:[ .()-]*\d){8,10}(?!\w)")


def safe_raw_answer_for_debug(value: str, limit: int = 2000) -> tuple[str, bool, bool]:
    """Return bounded answer text with explicit redaction/truncation indicators."""
    sanitized = value
    redacted = False
    for pattern, replacement in (
        (_SECRET_VALUE_RE, "[REDACTED_SECRET]"),
        (_EMAIL_RE, "[REDACTED_EMAIL]"),
        (_PHONE_RE, "[REDACTED_PHONE]"),
    ):
        sanitized, count = pattern.subn(replacement, sanitized)
        redacted = redacted or count > 0
    truncated = len(sanitized) > limit
    if truncated:
        sanitized = sanitized[:limit] + "...[TRUNCATED]"
    return sanitized, redacted, truncated


def citation_raw_debug_enabled(environment) -> bool:
    """Raw model text is logged only when both explicit development gates are set."""
    environment_name = str(environment.get("DAI_ENVIRONMENT", "")).strip().lower()
    debug_value = str(
        environment.get("DAI_DEBUG_CITATION_RAW_ANSWER", "")
    ).strip().lower()
    return environment_name in {"development", "local"} and debug_value in {
        "1", "true", "yes"
    }


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
    safe_text: Callable[[str], Any],
    include_raw_answer: bool = False,
    allowed_evidence_ids: tuple[str, ...] | list[str] = (),
):
    """Build a callback; raw text is opt-in and passed through safe redaction."""

    def log_citation_attempt(
        citation_attempt: int,
        answer: str,
        structured_answer: StructuredAnswer,
        verification_result: VerificationResult,
    ) -> None:
        allowed_ids = set(allowed_evidence_ids)
        cited_ids = {
            evidence_id
            for claim in structured_answer.claims
            for evidence_id in claim.evidence_ids
        }
        unknown_ids = sorted(cited_ids - allowed_ids)
        if include_raw_answer:
            safe_result = safe_text(answer)
            if isinstance(safe_result, tuple):
                safe_answer = safe_result[0]
                was_redacted = bool(safe_result[1]) if len(safe_result) > 1 else False
                was_truncated = bool(safe_result[2]) if len(safe_result) > 2 else False
            else:
                safe_answer = safe_result
                was_redacted = safe_answer != answer
                was_truncated = False
            print(
                "[CITATION RAW DEBUG]",
                "request_id=", request_id,
                "citation_attempt=", citation_attempt,
                "answer_chars=", len(answer),
                "redacted=", was_redacted,
                "truncated=", was_truncated,
                "answer=", safe_answer,
            )
        print(
            "[CITATION PARSE]",
            "request_id=", request_id,
            "citation_attempt=", citation_attempt,
            "answer_chars=", len(answer),
            "raw_answer_debug_enabled=", include_raw_answer,
            "claim_count=", len(structured_answer.claims),
            "claims_with_evidence=", sum(
                bool(claim.evidence_ids) for claim in structured_answer.claims
            ),
            "cited_evidence_count=", len({
                evidence_id
                for claim in structured_answer.claims
                for evidence_id in claim.evidence_ids
            }),
            "allowed_evidence_count=", len(allowed_ids),
            "unknown_evidence_ids=", json.dumps(unknown_ids, ensure_ascii=False),
            "parser_errors=", json.dumps(
                structured_answer.metadata.get("citation_parse_errors", []),
                sort_keys=True,
            ),
            "verifier_error_count=", len(verification_result.errors),
            "verifier_reason_counts=", json.dumps(
                verifier_reason_codes(verification_result),
                sort_keys=True,
            ),
            "claim_grounding_status=", verification_result.claim_grounding_status,
            "confidence_status=", verification_result.confidence_status,
        )

    return log_citation_attempt
