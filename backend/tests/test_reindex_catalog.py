from types import SimpleNamespace

from backend.search.reindex_catalog import catalog_domains, catalog_record_for_domain


def test_catalog_domains_come_from_enabled_registry_catalog_routes():
    assert set(catalog_domains()) == {"legal_ai", "administrative_law"}


def test_catalog_reindex_preserves_catalog_source_and_record_identity():
    procedure = SimpleNamespace(
        formality_id="FORM-001",
        code="1.002421",
        procedure_name="Test administrative procedure",
        target_type="organization",
        category_name="public services",
        department_promulgate="Ministry",
        description="Description from Catalog.",
        execution_steps="Step 1.",
        execution_methods="Online",
        profile_components="Application",
        requirements_conditions="Eligibility",
        fees="No fee",
        legal_basis="Decree 1",
        results="Permit",
        target_objects="Applicant",
        executing_agencies="Agency",
        coordinating_agencies="",
        source="dichvucong-gov-vn",
        source_url="https://dichvucong.gov.vn/procedure/FORM-001",
        status="active",
    )

    record = catalog_record_for_domain(procedure, "administrative_law")
    payload = record.to_payload()

    assert record.domain_id == "administrative_law"
    assert record.source_id == "FORM-001"
    assert record.document_id == "FORM-001"
    assert payload["record_id"] == "FORM-001"
    assert payload["source_url"] == "https://dichvucong.gov.vn/procedure/FORM-001"
    assert payload["evidence_id"] == "catalog:FORM-001"
    assert "Test administrative procedure" in record.content
