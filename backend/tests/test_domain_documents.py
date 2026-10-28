from backend.domain_documents import (
    DomainDocumentError,
    DomainDocumentService,
    chunk_document,
    decode_document,
    domain_document_collection,
    retrieval_collections_for_domain,
)
from backend.domain_registry import DomainRegistryService
from backend.policy.middleware import PolicyEnforcementMiddleware
from backend.rag.adapter import rag_result_to_evidence


class FakeQdrant:
    def __init__(self):
        self.collections = set()
        self.upserts = []

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections.add(collection_name)

    def upsert(self, collection_name, points, wait):
        self.upserts.append((collection_name, points, wait))


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
    existing = {"domain_documents_tourism"}
    exists = existing.__contains__
    assert retrieval_collections_for_domain("tourism", exists, registry) == [
        "domain_documents_tourism"
    ]
    assert retrieval_collections_for_domain("general_assistant", exists, registry) == []
    existing.add("legal_docs_bge_m3")
    existing.add("domain_documents_legal_ai")
    assert retrieval_collections_for_domain("legal_ai", exists, registry) == [
        "legal_docs_bge_m3", "domain_documents_legal_ai"
    ]


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
import pytest

