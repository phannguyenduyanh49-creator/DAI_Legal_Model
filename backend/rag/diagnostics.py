"""Safe classification of retrieval failures for pipeline telemetry."""

import httpx
from pydantic import ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException


def retrieval_exception_details(exc: Exception) -> dict[str, str]:
    """Expose the wrapped Qdrant failure category without logging its message."""
    if isinstance(exc, ResponseHandlingException):
        source = getattr(exc, "source", None)
        source_type = type(source).__name__ if source is not None else "unknown"
        if isinstance(source, ValidationError):
            reason_code = "QDRANT_RESPONSE_SCHEMA_ERROR"
        elif isinstance(source, httpx.TimeoutException):
            reason_code = "QDRANT_TIMEOUT"
        elif isinstance(source, httpx.TransportError):
            reason_code = "QDRANT_TRANSPORT_ERROR"
        else:
            reason_code = "QDRANT_RESPONSE_HANDLING_ERROR"
        return {
            "component": "qdrant",
            "reason_code": reason_code,
            "exception_type": type(exc).__name__,
            "cause_exception_type": source_type,
        }

    return {
        "component": "retrieval_pipeline",
        "reason_code": "UNKNOWN_PIPELINE_FAILURE",
        "exception_type": type(exc).__name__,
        "cause_exception_type": type(exc).__name__,
    }


def attach_retrieval_failure_telemetry(
    result: dict,
    exc: Exception,
) -> dict:
    """Mark a fallback result as a retrieval failure, never ordinary no-data."""
    details = retrieval_exception_details(exc)
    result.update(
        retrieval_status="error",
        failure_stage="retrieval_pipeline",
        failure_reason_code=details["reason_code"],
        exception_type=details["exception_type"],
        cause_exception_type=details["cause_exception_type"],
        failure_component=details["component"],
    )
    return result


def classify_retrieval_result(
    retrieved_count: int,
    accepted_count: int,
    rejected_by_relevance_count: int,
) -> str:
    """Separate an empty vector result from candidates rejected as irrelevant."""
    if retrieved_count <= 0:
        return "no_match"
    if accepted_count <= 0 and rejected_by_relevance_count > 0:
        return "rejected_by_relevance_filter"
    return "completed"
