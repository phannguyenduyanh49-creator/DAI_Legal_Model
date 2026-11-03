from backend.domain_documents import DomainDocumentService
from backend.domain_registry import DomainRegistryService
from backend.search.repository import StoredDocument


class FakeSearchEngine:
    index_name = "dai_documents_test"


class FakeSearchStore:
    def __init__(self):
        self.elasticsearch = FakeSearchEngine()
        self.rows = []

    def index_documents(self, rows):
        self.rows.extend(rows)
        return len(rows)


def test_domain_upload_can_persist_and_index_without_qdrant_or_embedding():
    store = FakeSearchStore()
    service = DomainDocumentService(
        qdrant=None,
        embedder=None,
        registry=DomainRegistryService(),
        search_store=store,
    )

    result = service.upload("tourism", "parks.txt", "Vườn quốc gia Cát Tiên có rừng và tuyến tham quan.".encode())

    assert result["status"] == "indexed"
    assert result["domain_id"] == "tourism"
    assert result["indexed_domains"][0]["collection"] == "dai_documents_test"
    assert len(store.rows) == result["chunk_count"]
    assert all(payload["domain_id"] == "tourism" for _, payload in store.rows)
    assert all(payload["evidence_id"].startswith("domain:tourism:") for _, payload in store.rows)


def test_search_store_upload_uses_stable_ids_for_idempotent_reupload():
    store = FakeSearchStore()
    service = DomainDocumentService(
        qdrant=None,
        embedder=None,
        registry=DomainRegistryService(),
        search_store=store,
    )
    content = "Nguồn thông tin du lịch có văn bản xác định."

    first = service.upload("tourism", "guide.txt", content.encode())
    first_ids = [key for key, _payload in store.rows]
    service.upload("tourism", "guide.txt", content.encode())
    second_ids = [key for key, _payload in store.rows[len(first_ids):]]

    assert first["document_id"]
    assert second_ids == first_ids


def test_stored_document_requires_domain_source_and_content():
    doc = StoredDocument.from_payload("key-1", {
        "domain_id": "healthcare",
        "source_id": "source-1",
        "document_id": "document-1",
        "text": "A source-backed record",
        "record_id": "HC-K-1",
    })
    assert doc.domain_id == "healthcare"
    assert doc.to_payload()["record_id"] == "HC-K-1"


def test_stored_document_indexes_chunk_text_and_preserves_full_source_text():
    doc = StoredDocument.from_payload("key-3", {
        "domain_id": "legal_ai",
        "source_id": "source-3",
        "document_id": "document-3",
        "text": "One chunk",
        "full_text": "The complete source document",
    })

    assert doc.content == "One chunk"
    assert doc.to_payload()["text"] == "One chunk"
    assert doc.to_payload()["full_text"] == "The complete source document"

    try:
        StoredDocument.from_payload("key-2", {"domain_id": "healthcare", "text": "No source"})
    except ValueError as exc:
        assert "source ID" in str(exc)
    else:
        raise AssertionError("Expected a missing source ID to be rejected")
