from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from backend.answer.citations import structure_answer_with_explicit_evidence
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
    generation_error: str | None = None


GenerateCallback = Callable[[str, int], Awaitable[dict[str, Any]]]
AttemptCallback = Callable[[int, str, StructuredAnswer, VerificationResult], None]


def _retry_prompt(prompt: str, evidence_bundle: EvidenceBundle) -> str:
    evidence_ids = [item.evidence_id for item in evidence_bundle.evidences]
    allowed_ids = ", ".join(evidence_ids) if evidence_ids else "none"
    return (
        prompt
        + "\n\n=== CITATION REPAIR ===\n"
        + "Rewrite concisely. End every legal factual sentence with "
        + "[Evidence ID: <exact ID from the allowed list>]. Use only these IDs: "
        + allowed_ids
        + ". Never invent a source, law, or ID. If evidence is insufficient, say so "
        + "without making unsupported legal claims."
    )

async def generate_with_citation_retry(
    prompt: str,
    evidence_bundle: EvidenceBundle,
    generate: GenerateCallback,
    verifier: Verifier,
    max_attempts: int = MAX_CITATION_ATTEMPTS,
    on_attempt: AttemptCallback | None = None,
) -> CitationGenerationResult:
    """Generate at most twice; return an answer only after citation integrity passes."""
    attempts_limit = max(1, min(int(max_attempts), MAX_CITATION_ATTEMPTS))
    current_prompt = prompt
    last_response: dict[str, Any] = {}
    model_processing_total = 0.0
    has_model_processing_time = False
    last_generation_error = None
    last_structured = structure_answer_with_explicit_evidence("")
    last_verification = verifier.verify(last_structured, evidence_bundle)

    for attempt in range(1, attempts_limit + 1):
        try:
            response = await generate(current_prompt, attempt)
        except Exception as exc:
            last_generation_error = type(exc).__name__
            if attempt < attempts_limit:
                continue
            break

        last_generation_error = None
        last_response = response if isinstance(response, dict) else {}
        processing_time = last_response.get("processing_time_ms")
        if isinstance(processing_time, (int, float)) and not isinstance(
            processing_time, bool
        ):
            model_processing_total += float(processing_time)
            has_model_processing_time = True
        raw_answer = last_response.get("answer")
        answer = (
            raw_answer.strip()
            if last_response.get("success") is True
            and isinstance(raw_answer, str)
            else ""
        )
        last_structured = structure_answer_with_explicit_evidence(answer)
        last_verification = verifier.verify(last_structured, evidence_bundle)
        if on_attempt is not None:
            on_attempt(attempt, answer, last_structured, last_verification)

        if answer and last_verification.passed:
            return CitationGenerationResult(
                answer=answer,
                structured_answer=last_structured,
                verification=last_verification,
                attempts=attempt,
                last_model_response=last_response,
                model_processing_time_ms=(
                    model_processing_total if has_model_processing_time else None
                ),
            )

        if attempt < attempts_limit and evidence_bundle.evidences:
            current_prompt = _retry_prompt(prompt, evidence_bundle)

    return CitationGenerationResult(
        answer=None,
        structured_answer=last_structured,
        verification=last_verification,
        attempts=attempts_limit,
        last_model_response=last_response,
        model_processing_time_ms=(
            model_processing_total if has_model_processing_time else None
        ),
        generation_error=last_generation_error,
    )
