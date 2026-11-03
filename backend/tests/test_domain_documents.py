from io import BytesIO
from types import SimpleNamespace

import pytest
from openpyxl import Workbook

from backend.domain_documents import (
    DomainDocumentError,
    DomainDocumentService,
    chunk_document,
    decode_document,
    domain_document_collection,
    domain_upload_payload_matches,
    excel_to_text,
    excel_row_records,
    retrieval_collections_for_domain,
)
from backend.domain_registry import DomainRegistryService
from backend.policy.middleware import PolicyEnforcementMiddleware
from backend.rag.adapter import rag_result_to_evidence


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


def test_domain_upload_indexes_real_source_metadata_into_only_its_collection():
    registry = DomainRegistryService()
    qdrant = FakeQdrant()
    service = DomainDocumentService(qdrant, FakeEmbedder(), registry)

    result = service.upload(
        "tourism",
        "North Pier shuttle.md",
        "Route ZK-481 leaves North Pier at 06:47.".encode(),
    )

    assert result["status"] == "indexed"
    assert result["domain_id"] == "tourism"
    assert result["chunk_count"] == 1
    collection, points, wait = qdrant.upserts[0]
    assert collection == "domain_documents_tourism"
    assert wait is True
    payload = points[0].payload
    assert payload["domain_id"] == "tourism"
    assert payload["source_type"] == "domain_upload"
    assert payload["source_id"] == result["document_id"]
    assert payload["evidence_id"] == f"domain:tourism:{result['document_id']}:1"
    assert payload["title"] == "North Pier shuttle.md"
    assert "source_url" not in payload
    assert "ZK-481" in payload["text"]


@pytest.mark.parametrize(
    "domain_id",
    [
        "legal_ai",
        "administrative_law",
        "tourism",
        "healthcare",
        "education",
        "shopping",
        "food",
    ],
)
def test_each_published_upload_domain_indexes_to_its_own_collection(domain_id):
    qdrant = FakeQdrant()
    result = DomainDocumentService(
        qdrant,
        FakeEmbedder(),
        DomainRegistryService(),
    ).upload(domain_id, "source.md", b"Source-backed sample content.")

    collection, points, wait = qdrant.upserts[0]
    assert result["status"] == "indexed"
    assert collection == f"domain_documents_{domain_id}"
    assert result["domain_id"] == domain_id
    assert wait is True
    assert points[0].payload["domain_id"] == domain_id


def test_upload_is_not_reported_as_indexed_when_qdrant_does_not_confirm_completion():
    class UnconfirmedQdrant(FakeQdrant):
        def upsert(self, collection_name, points, wait):
            super().upsert(collection_name, points, wait)
            return SimpleNamespace(status="acknowledged")

    qdrant = UnconfirmedQdrant()
    with pytest.raises(RuntimeError, match="did not confirm completed"):
        DomainDocumentService(qdrant, FakeEmbedder()).upload(
            "tourism", "routes.md", b"A route with source data."
        )


def test_general_assistant_is_not_an_upload_target():
    qdrant = FakeQdrant()
    with pytest.raises(DomainDocumentError, match="not enabled"):
        DomainDocumentService(qdrant, FakeEmbedder()).upload(
            "general_assistant", "note.md", b"General assistant note."
        )
    assert qdrant.upserts == []


def test_xlsx_is_normalized_across_sheets_and_indexed_in_the_selected_domain():
    workbook = Workbook()
    schools = workbook.active
    schools.title = "Schools"
    schools.append(["school", "tuition"])
    schools.append(["Example Campus", 12500000])
    admissions = workbook.create_sheet("Admissions")
    admissions.append(["term", "deadline"])
    admissions.append(["Fall", "2026-08-01"])
    source = BytesIO()
    workbook.save(source)

    normalized = excel_to_text(source.getvalue())
    assert "[Sheet: Schools | Row: 2]" in normalized
    assert "school: Example Campus" in normalized
    assert "tuition: 12500000" in normalized
    assert "[Sheet: Admissions | Row: 2]" in normalized
    assert "deadline: 2026-08-01" in normalized

    qdrant = FakeQdrant()
    result = DomainDocumentService(
        qdrant, FakeEmbedder(), DomainRegistryService()
    ).upload("education", "schools.xlsx", source.getvalue())
    collection, points, _ = qdrant.upserts[0]
    assert result["status"] == "indexed"
    assert collection == "domain_documents_education"
    assert points[0].payload["domain_id"] == "education"
    assert points[0].payload["title"] == "schools.xlsx"
    assert len(points) == 2
    assert points[0].payload["sheet_name"] == "Schools"
    assert points[0].payload["row_number"] == 2
    assert points[0].payload["record_fields"] == {
        "school": "Example Campus", "tuition": "12500000"
    }
    assert points[1].payload["sheet_name"] == "Admissions"
    assert points[1].payload["record_fields"] == {
        "term": "Fall", "deadline": "2026-08-01"
    }


def test_shopping_xlsx_is_parsed_and_indexed_with_document_metadata():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Products"
    sheet.append(["product", "price", "store"])
    sheet.append(["Example kettle", 490000, "Saigon shop"])
    source = BytesIO()
    workbook.save(source)
    qdrant = FakeQdrant()

    result = DomainDocumentService(
        qdrant, FakeEmbedder(), DomainRegistryService()
    ).upload("shopping", "Mua_Sam_Sai_Gon_100_San_Pham.xlsx", source.getvalue())

    collection, points, wait = qdrant.upserts[0]
    assert result["status"] == "indexed"
    assert result["domain_id"] == "shopping"
    assert result["title"] == "Mua_Sam_Sai_Gon_100_San_Pham.xlsx"
    assert result["chunk_count"] == 1
    assert collection == "domain_documents_shopping"
    assert wait is True
    assert points[0].payload["domain_id"] == "shopping"
    assert points[0].payload["document_id"] == result["document_id"]
    assert points[0].payload["source_id"] == result["document_id"]
    assert points[0].payload["title"] == result["title"]
    assert "Example kettle" in points[0].payload["text"]
    assert points[0].payload["record_fields"] == {
        "product": "Example kettle", "price": "490000", "store": "Saigon shop"
    }
    assert points[0].payload["record_name"] == "Example kettle"
    assert points[0].payload["sheet_name"] == "Products"
    assert points[0].payload["row_number"] == 2


def test_xlsx_tourism_records_keep_each_destination_and_its_fields_together():
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Destinations"
    sheet.append([
        "Tên địa điểm", "Tỉnh/Thành phố", "Khu vực",
        "Thời gian tham quan gợi ý", "Giá vé người lớn (VND)",
        "Tình trạng dữ liệu",
    ])
    sheet.append(["Hội An", "Quảng Nam", "Miền Trung", "2–4 giờ", None, "đã kiểm tra"])
    sheet.append(["Bà Nà Hills", "Đà Nẵng", "Miền Trung", "Nửa ngày đến 1 ngày", 900000, "đã kiểm tra"])
    sheet.append(["Hồ Gươm", "Hà Nội", "Miền Bắc", "2 giờ", None, "chưa xác minh tọa độ"])
    source = BytesIO()
    workbook.save(source)

    normalized = excel_to_text(source.getvalue())
    records = excel_row_records(normalized)
    assert len(records) == 3
    assert records[0]["fields"]["Tên địa điểm"] == "Hội An"
    assert records[0]["fields"]["Thời gian tham quan gợi ý"] == "2–4 giờ"

    qdrant = FakeQdrant()
    result = DomainDocumentService(qdrant, FakeEmbedder()).upload(
        "tourism", "destinations.xlsx", source.getvalue()
    )
    collection, points, _ = qdrant.upserts[0]
    assert collection == "domain_documents_tourism"
    assert result["chunk_count"] == 3
    assert [point.payload["record_name"] for point in points] == ["Hội An", "Bà Nà Hills", "Hồ Gươm"]
    assert points[0].payload["record_fields"]["thoi_gian_tham_quan_goi_y"] == "2–4 giờ"
    assert points[0].payload["tourism_region"] == "Miền Trung"
    assert points[0].payload["sheet_name"] == "Destinations"
    assert points[0].payload["row_number"] == 2
    assert "Giá vé người lớn (VND)" not in points[0].payload["text"]
    assert points[2].payload["record_fields"]["tinh_thanh_pho"] == "Hà Nội"
    assert points[2].payload["record_fields"]["tinh_thanh_pho"] != ""


def test_uploading_identical_workbook_again_upserts_same_point_ids():
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Tên địa điểm", "Khu vực"])
    sheet.append(["Example", "Miền Trung"])
    source = BytesIO()
    workbook.save(source)
    qdrant = FakeQdrant()
    service = DomainDocumentService(qdrant, FakeEmbedder())

    first = service.upload("tourism", "same.xlsx", source.getvalue())
    second = service.upload("tourism", "same.xlsx", source.getvalue())

    first_points = qdrant.upserts[0][1]
    second_points = qdrant.upserts[1][1]
    assert first["document_id"] == second["document_id"]
    assert [point.id for point in first_points] == [point.id for point in second_points]
    assert [point.payload["evidence_id"] for point in first_points] == [
        point.payload["evidence_id"] for point in second_points
    ]
    assert len(qdrant.deletes) == 2
    delete_filter = qdrant.deletes[-1][1].filter
    assert any(condition.key == "title" for condition in delete_filter.must)
    assert any(condition.key == "document_id" for condition in delete_filter.must_not)


def test_legal_xlsx_upload_keeps_existing_generic_chunking_behavior():
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["procedure", "detail"])
    sheet.append(["Legal record", "Source text"])
    source = BytesIO()
    workbook.save(source)
    qdrant = FakeQdrant()

    DomainDocumentService(qdrant, FakeEmbedder()).upload(
        "legal_ai", "legal.xlsx", source.getvalue()
    )

    _collection, points, _ = qdrant.upserts[0]
    assert len(points) == 1
    assert "[Sheet: Sheet | Row: 2]" in points[0].payload["text"]
    assert "record_fields" not in points[0].payload


def test_xlsx_rejects_invalid_zip_and_domain_extension_disallowance():
    with pytest.raises(DomainDocumentError, match="Excel"):
        excel_to_text(b"not-an-xlsx")
    with pytest.raises(DomainDocumentError, match="Chỉ hỗ trợ"):
        decode_document("source.xlsx", b"data", allowed_extensions={".txt", ".md"})


def test_domain_upload_does_not_map_missing_domain_to_legal_ai():
    with pytest.raises(DomainDocumentError, match="Unknown domain"):
        DomainDocumentService(FakeQdrant(), FakeEmbedder()).upload(
            "unregistered", "note.txt", b"private value"
        )

    middleware = PolicyEnforcementMiddleware(DomainRegistryService())
    with pytest.raises(KeyError):
        middleware.apply("unregistered")


def test_domain_upload_rejects_disabled_domain_before_embedding_or_storage():
    registry = DomainRegistryService()
    tourism = registry.get_domain_config("tourism")
    registry._registry["tourism"] = tourism.model_copy(update={"enabled": False})
    qdrant = FakeQdrant()
    with pytest.raises(DomainDocumentError, match="disabled"):
        DomainDocumentService(qdrant, FakeEmbedder(), registry).upload(
            "tourism", "routes.txt", b"Route XY-77 departs at 07:00."
        )
    assert qdrant.collections == set()
    assert qdrant.upserts == []


def test_incomplete_embedding_output_fails_without_creating_or_upserting_collection():
    class IncompleteEmbedder(FakeEmbedder):
        def encode(self, texts, **_kwargs):
            return []

    qdrant = FakeQdrant()
    service = DomainDocumentService(qdrant, IncompleteEmbedder(), DomainRegistryService())

    with pytest.raises(RuntimeError, match="incomplete vector set"):
        service.upload("tourism", "routes.txt", b"Route XY-77 departs at 07:00.")

    assert "domain_documents_tourism" not in qdrant.collections
    assert qdrant.upserts == []


def test_domain_collection_names_are_validated_and_isolated():
    assert domain_document_collection("tourism") == "domain_documents_tourism"
    assert domain_document_collection("general_assistant") != domain_document_collection("tourism")
    for invalid in ("../tourism", "tourism/../../legal_ai", ""):
        with pytest.raises(DomainDocumentError):
            domain_document_collection(invalid)


def test_retrieval_collections_never_fall_back_to_legal_for_other_domains():
    registry = DomainRegistryService()
    existing = {"domain_documents_tourism", "domain_documents_shopping"}
    exists = existing.__contains__
    assert retrieval_collections_for_domain("tourism", exists, registry) == [
        "domain_documents_tourism"
    ]
    assert retrieval_collections_for_domain("shopping", exists, registry) == [
        "domain_documents_shopping"
    ]
    assert retrieval_collections_for_domain("general_assistant", exists, registry) == []
    existing.add("legal_docs_bge_m3")
    existing.add("domain_documents_legal_ai")
    assert retrieval_collections_for_domain("legal_ai", exists, registry) == [
        "legal_docs_bge_m3", "domain_documents_legal_ai"
    ]


def test_retrieved_domain_upload_payload_must_match_the_requested_domain():
    assert domain_upload_payload_matches(
        {"source_type": "domain_upload", "domain_id": "healthcare"},
        "healthcare",
    )
    assert not domain_upload_payload_matches(
        {"source_type": "domain_upload", "domain_id": "legal_ai"},
        "healthcare",
    )
    assert not domain_upload_payload_matches(
        {"source_type": "domain_upload"},
        "healthcare",
    )
    assert domain_upload_payload_matches({"source_type": "catalog"}, "legal_ai")


def test_missing_base_collection_does_not_block_domain_upload_retrieval():
    registry = DomainRegistryService()
    existing = {"domain_documents_legal_ai"}
    collections = retrieval_collections_for_domain(
        "legal_ai", existing.__contains__, registry
    )

    assert collections == ["domain_documents_legal_ai"]
    assert "legal_docs_bge_m3" not in collections


def test_upload_rejects_unsupported_or_invalid_documents():
    for filename, content in (
        ("law.pdf", b"not parsed"),
        ("note.txt", b"\xff"),
        ("empty.md", b"   "),
    ):
        with pytest.raises(DomainDocumentError):
            decode_document(filename, content)
    with pytest.raises(DomainDocumentError, match="2 MiB"):
        decode_document("large.txt", b"x" * (2 * 1024 * 1024 + 1))


def test_chunking_keeps_text_and_domain_evidence_source_identity():
    text = ("First segment. " * 80) + "UNIQUE-DOMAIN-FACT-447" + (" Later text." * 80)
    chunks = chunk_document(text, chunk_size=64, chunk_overlap=8)
    assert len(chunks) > 1
    assert any("UNIQUE-DOMAIN-FACT-447" in chunk for _, _, chunk in chunks)

    evidence = rag_result_to_evidence({
        "source_type": "domain_upload",
        "domain_id": "tourism",
        "evidence_id": "domain:tourism:doc-1:2",
        "source_id": "doc-1",
        "title": "route.md",
        "location": "Characters 100-180",
        "text": "UNIQUE-DOMAIN-FACT-447",
        "score": 0.9,
    })
    assert evidence.evidence_id == "domain:tourism:doc-1:2"
    assert evidence.source_type == "domain_upload"
    assert evidence.source_id == "doc-1"
    assert evidence.title == "route.md"
    assert evidence.location == "Characters 100-180"
    assert evidence.metadata["domain_id"] == "tourism"
    assert evidence.source_url is None


def test_rag_adapter_preserves_row_fields_and_source_verification_metadata():
    evidence = rag_result_to_evidence({
        "source_type": "domain_upload",
        "domain_id": "shopping",
        "evidence_id": "domain:shopping:doc-2:4",
        "document_id": "doc-2",
        "source_id": "doc-2",
        "point_id": "point-4",
        "chunk_index": 4,
        "record_id": "SKU-123",
        "sheet_name": "Products",
        "row_number": 5,
        "title": "products.xlsx",
        "location": "Sheet Products · row 5",
        "text": "product: kettle\nprice: 490000",
        "record_fields": {"product": "kettle", "price": "490000"},
        "metadata": {"verification_status": "unverified", "source_name": "products.xlsx"},
        "validity_from": "2025-01-01",
        "completeness": {"price": True},
    })

    assert evidence.evidence_id == "domain:shopping:doc-2:4"
    assert evidence.source_id == "doc-2"
    assert evidence.metadata["document_id"] == "doc-2"
    assert evidence.metadata["record_id"] == "SKU-123"
    assert evidence.metadata["chunk_index"] == 4
    assert evidence.metadata["point_id"] == "point-4"
    assert evidence.metadata["sheet_name"] == "Products"
    assert evidence.metadata["row_number"] == 5
    assert evidence.fields == {"product": "kettle", "price": "490000"}
    assert evidence.metadata["verification_status"] == "unverified"
    assert evidence.metadata["domain_id"] == "shopping"
    assert evidence.validity_from == "2025-01-01"
    assert evidence.completeness == {"price": True}


def test_domain_upload_adapter_does_not_infer_source_or_evidence_identity():
    evidence = rag_result_to_evidence({
        "source_type": "domain_upload",
        "domain_id": "shopping",
        "document_id": "doc-2",
        "title": "products.xlsx",
        "text": "product: kettle",
    })

    assert evidence.evidence_id == ""
    assert evidence.source_id is None
import pytest

