import httpx
from backend.rag.diagnostics import classify_retrieval_result


def test_retrieval_result_status_distinguishes_empty_and_relevance_rejection():
    assert classify_retrieval_result(0, 0, 0) == "no_match"
    assert classify_retrieval_result(4, 0, 4) == "rejected_by_relevance_filter"
    assert classify_retrieval_result(4, 1, 3) == "completed"
import pytest
from pydantic import BaseModel, ValidationError
from qdrant_client.http.api.search_api import SyncSearchApi
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.api_client import ApiClient
from qdrant_client.http import models as qdrant_models

from backend.rag.diagnostics import (
    attach_retrieval_failure_telemetry,
    retrieval_exception_details,
)


@pytest.mark.parametrize(
    ("cause", "reason"),
    [
        (httpx.ConnectError("connection refused"), "QDRANT_TRANSPORT_ERROR"),
        (httpx.ReadTimeout("read timed out"), "QDRANT_TIMEOUT"),
    ],
)
def test_qdrant_response_handling_transport_cause_is_preserved(cause, reason):
    details = retrieval_exception_details(ResponseHandlingException(cause))

    assert details == {
        "component": "qdrant",
        "reason_code": reason,
        "exception_type": "ResponseHandlingException",
        "cause_exception_type": type(cause).__name__,
    }


def test_qdrant_success_response_schema_error_is_distinguished():
    class ExpectedResponse(BaseModel):
        result: str

    with pytest.raises(ValidationError) as error:
        ExpectedResponse.model_validate({})

    details = retrieval_exception_details(ResponseHandlingException(error.value))

    assert details["component"] == "qdrant"
    assert details["reason_code"] == "QDRANT_RESPONSE_SCHEMA_ERROR"
    assert details["cause_exception_type"] == "ValidationError"


def test_unknown_qdrant_response_failure_remains_explicitly_unknown():
    details = retrieval_exception_details(ResponseHandlingException(RuntimeError()))

    assert details["component"] == "qdrant"
    assert details["reason_code"] == "QDRANT_RESPONSE_HANDLING_ERROR"
    assert details["cause_exception_type"] == "RuntimeError"


def test_non_qdrant_retrieval_failure_is_not_mislabeled_as_qdrant():
    details = retrieval_exception_details(ValueError("bad local payload"))

    assert details["component"] == "retrieval_pipeline"
    assert details["reason_code"] == "UNKNOWN_PIPELINE_FAILURE"


def test_qdrant_client_wraps_malformed_success_schema_and_reports_source_type():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"status": "ok", "result": {}},
            request=request,
        )
    )
    client = ApiClient("http://qdrant", transport=transport)
    try:
        with pytest.raises(ResponseHandlingException) as error:
            SyncSearchApi(client).query_points(
                collection_name="legal_docs_bge_m3",
                query_request=qdrant_models.QueryRequest(query=[0.1, 0.2]),
            )
    finally:
        client.close()

    details = retrieval_exception_details(error.value)
    assert details["reason_code"] == "QDRANT_RESPONSE_SCHEMA_ERROR"
    assert details["cause_exception_type"] == "ValidationError"


def test_qdrant_client_wraps_transport_failure_and_reports_source_type():
    def fail_transport(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = ApiClient("http://qdrant", transport=httpx.MockTransport(fail_transport))
    try:
        with pytest.raises(ResponseHandlingException) as error:
            SyncSearchApi(client).query_points(
                collection_name="legal_docs_bge_m3",
                query_request=qdrant_models.QueryRequest(query=[0.1, 0.2]),
            )
    finally:
        client.close()

    details = retrieval_exception_details(error.value)
    assert details["reason_code"] == "QDRANT_TRANSPORT_ERROR"
    assert details["cause_exception_type"] == "ConnectError"


def test_retrieval_fallback_keeps_qdrant_error_distinct_from_insufficient_data():
    failure = ResponseHandlingException(httpx.ConnectError("refused"))
    fallback = {
        "retrieval_status": "error",
        "enough_data": False,
        "stage_status": {"retrieval": "failed"},
    }

    result = attach_retrieval_failure_telemetry(fallback, failure)

    assert result["enough_data"] is False
    assert result["retrieval_status"] == "error"
    assert result["failure_stage"] == "retrieval_pipeline"
    assert result["failure_reason_code"] == "QDRANT_TRANSPORT_ERROR"
    assert result["failure_component"] == "qdrant"
