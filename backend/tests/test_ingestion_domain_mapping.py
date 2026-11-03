from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from openpyxl import Workbook

from backend.domain_documents import DomainDocumentError, DomainDocumentService
from backend.domain_registry import DomainRegistryService


class FakeQdrant:
    def __init__(self):
        self.collections = set()
        self.upserts = []
        self.deletes = []

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections.add(collection_name)

    def upsert(self, collection_name, points, wait):
        self.upserts.append((collection_name, points, wait))
        return SimpleNamespace(status="completed")

    def delete(self, collection_name, points_selector, wait):
        self.deletes.append((collection_name, points_selector, wait))
        return SimpleNamespace(status="completed")


class FakeEmbedder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **_kwargs):
        return [[1.0, 0.0, 0.0] for _ in texts]


def mixed_workbook() -> bytes:
    workbook = Workbook()
    health = workbook.active
    health.title = "Y_te_100"
    health.append(["record_id", "ten_co_so", "chuyen_khoa"])
    health.append(["HC-F-001", "Clinic mock", "Da liễu"])

    education = workbook.create_sheet("Giao_duc_100")
    education.append(["record_id", "ten_khoa_hoc", "trinh_do_dau_vao"])
    education.append(["EDU-001", "AI nhập môn", "Người mới bắt đầu"])

    readme = workbook.create_sheet("README_Y_te")
    readme.append(["section", "content"])
    readme.append(["notice", "Instruction text must not be indexed"])

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def test_mixed_workbook_routes_each_sheet_to_its_domain_and_excludes_readme():
    qdrant = FakeQdrant()
    service = DomainDocumentService(qdrant, FakeEmbedder(), DomainRegistryService())

    result = service.upload("healthcare", "Healthcare_Education.xlsx", mixed_workbook())

    assert result["status"] == "indexed"
    assert result["domain_id"] == "healthcare"
    assert result["chunk_count"] == 1
    assert result["total_chunk_count"] == 2
    assert {item["domain_id"] for item in result["indexed_domains"]} == {
        "healthcare", "education"
    }
    uploads = {collection: points for collection, points, _wait in qdrant.upserts}
    assert set(uploads) == {
        "domain_documents_healthcare", "domain_documents_education"
    }

    healthcare = uploads["domain_documents_healthcare"][0].payload
    education = uploads["domain_documents_education"][0].payload
    assert healthcare["domain_id"] == "healthcare"
    assert healthcare["record_id"] == "HC-F-001"
    assert healthcare["sheet_name"] == "Y_te_100"
    assert healthcare["row_number"] == 2
    assert healthcare["record_name"] == "Clinic mock"
    assert education["domain_id"] == "education"
    assert education["record_id"] == "EDU-001"
    assert education["sheet_name"] == "Giao_duc_100"
    assert education["row_number"] == 2
    assert education["record_name"] == "AI nhập môn"
    assert all("README" not in point.payload.get("sheet_name", "") for points in uploads.values() for point in points)
    assert all(point.payload["source_id"] == point.payload["document_id"] for points in uploads.values() for point in points)


def test_ambiguous_sheet_in_mixed_workbook_is_rejected_before_indexing():
    workbook = Workbook()
    healthcare = workbook.active
    healthcare.title = "Y_te_100"
    healthcare.append(["record_id", "ten_co_so", "chuyen_khoa"])
    healthcare.append(["HC-F-001", "Clinic", "Da liễu"])
    unknown = workbook.create_sheet("Extra data")
    unknown.append(["key", "value"])
    unknown.append(["x", "y"])
    source = BytesIO()
    workbook.save(source)
    qdrant = FakeQdrant()

    with pytest.raises(DomainDocumentError, match="Ambiguous Excel worksheet"):
        DomainDocumentService(qdrant, FakeEmbedder(), DomainRegistryService()).upload(
            "healthcare", "mixed.xlsx", source.getvalue()
        )

    assert qdrant.upserts == []
    assert qdrant.collections == set()


def test_domain_specific_headers_map_generic_sheet_names_and_reject_selection_mismatch():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["record_id", "ten_khoa_hoc", "trinh_do_dau_vao"])
    sheet.append(["EDU-002", "Deep Learning", "Python"])
    source = BytesIO()
    workbook.save(source)
    qdrant = FakeQdrant()

    with pytest.raises(DomainDocumentError, match="does not match"):
        DomainDocumentService(qdrant, FakeEmbedder(), DomainRegistryService()).upload(
            "healthcare", "education.xlsx", source.getvalue()
        )

    assert qdrant.upserts == []
    assert qdrant.collections == set()


def test_single_domain_generic_workbook_remains_backward_compatible():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Services"
    sheet.append(["Name", "Description"])
    sheet.append(["Example", "Generic domain record"])
    source = BytesIO()
    workbook.save(source)
    qdrant = FakeQdrant()

    result = DomainDocumentService(qdrant, FakeEmbedder(), DomainRegistryService()).upload(
        "healthcare", "services.xlsx", source.getvalue()
    )

    assert result["domain_id"] == "healthcare"
    assert len(qdrant.upserts) == 1
    assert qdrant.upserts[0][0] == "domain_documents_healthcare"


def test_mixed_xlsx_reupload_is_idempotent_per_domain_and_preserves_ids():
    qdrant = FakeQdrant()
    service = DomainDocumentService(qdrant, FakeEmbedder(), DomainRegistryService())
    source = mixed_workbook()

    first = service.upload("healthcare", "Healthcare_Education.xlsx", source)
    second = service.upload("healthcare", "Healthcare_Education.xlsx", source)

    assert first["indexed_domains"] == second["indexed_domains"]
    first_points = {
        collection: [(str(point.id), point.payload["evidence_id"]) for point in points]
        for collection, points, _wait in qdrant.upserts[:2]
    }
    second_points = {
        collection: [(str(point.id), point.payload["evidence_id"]) for point in points]
        for collection, points, _wait in qdrant.upserts[2:]
    }
    assert first_points == second_points
    assert qdrant.deletes == []


def test_partial_multi_collection_upsert_is_reported_as_failure_not_success():
    class FailsSecondUpsert(FakeQdrant):
        def upsert(self, collection_name, points, wait):
            result = super().upsert(collection_name, points, wait)
            if len(self.upserts) == 2:
                return SimpleNamespace(status="acknowledged")
            return result

    qdrant = FailsSecondUpsert()
    with pytest.raises(RuntimeError, match="did not confirm completed indexing"):
        DomainDocumentService(qdrant, FakeEmbedder(), DomainRegistryService()).upload(
            "healthcare", "Healthcare_Education.xlsx", mixed_workbook()
        )

    assert len(qdrant.upserts) == 2


def test_upload_domain_isolation_does_not_put_education_records_in_healthcare_payloads():
    qdrant = FakeQdrant()
    DomainDocumentService(qdrant, FakeEmbedder(), DomainRegistryService()).upload(
        "healthcare", "Healthcare_Education.xlsx", mixed_workbook()
    )

    for collection, points, _wait in qdrant.upserts:
        expected_domain = collection.removeprefix("domain_documents_")
        assert all(point.payload["domain_id"] == expected_domain for point in points)
