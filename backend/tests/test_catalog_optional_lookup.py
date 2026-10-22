import pytest
from sqlalchemy.exc import OperationalError

from backend.catalog.schemas import CatalogQuery, CatalogResult
from backend.catalog.service import CatalogService


def make_service(lookup):
    service = CatalogService()
    service.lookup = lookup
    return service


def test_catalog_lookup_returns_completed_when_catalog_responds():
    result = CatalogResult(
        matched=False,
        match_type="none",
        confidence=0.0,
        reason="No exact catalog record found",
    )
    service = make_service(lambda _query: result)

    actual, status = service.lookup_optional(
        CatalogQuery(domain="legal", intent="procedure_lookup", procedure_name="test")
    )

    assert actual is result
    assert status == "completed"


def test_catalog_lookup_marks_database_unavailable_without_failing_request():
    def fail_lookup(_query):
        raise OperationalError("SELECT 1", {}, RuntimeError("connection refused"))

    service = make_service(fail_lookup)

    actual, status = service.lookup_optional(
        CatalogQuery(domain="legal", intent="procedure_lookup", procedure_name="test")
    )

    assert actual is None
    assert status == "unavailable"


def test_catalog_lookup_none_is_unknown_and_safe_for_optional_callers():
    service = make_service(lambda _query: None)

    actual, status = service.lookup_optional(
        CatalogQuery(domain="legal", intent="procedure_lookup", procedure_name="test")
    )

    assert actual is None
    assert status == "unknown"


def test_catalog_lookup_does_not_hide_unexpected_errors():
    service = make_service(lambda _query: (_ for _ in ()).throw(RuntimeError("bug")))

    with pytest.raises(RuntimeError, match="bug"):
        service.lookup_optional(
            CatalogQuery(domain="legal", intent="procedure_lookup", procedure_name="test")
        )
