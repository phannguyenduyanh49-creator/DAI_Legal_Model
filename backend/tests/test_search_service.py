from backend.search.service import DomainSearchService
from backend.search.repository import StoredDocument
from backend.search.service import DocumentIndexingPending


class FakeRepository:
    def __init__(self, docs):
        self.docs = docs

    def health(self):
        return True

    def get_by_keys(self, keys):
        return {key: self.docs[key] for key in keys if key in self.docs}


class FakeElasticsearch:
    index_name = "dai_documents_test"

    def __init__(self, hits):
        self.hits = hits
        self.query = None

    def health(self):
        return True

    def search(self, query, domain_id, **kwargs):
        self.query = (query, domain_id, kwargs)
        return self.hits

    def ensure_index(self):
        return None

    def bulk_index(self, docs):
        self.indexed_docs = docs
        return [True] * len(docs)


class FakeOutboxRepository(FakeRepository):
    def __init__(self):
        super().__init__({})
        self.pending = []
        self.marked = []

    def store_and_queue(self, documents):
        self.pending = [{
            "event_id": index + 1,
            "document_key": document.document_key,
            "domain_id": document.domain_id,
            "source_id": document.source_id,
            "document_id": document.document_id,
            "chunk_index": document.chunk_index,
            "title": document.title,
            "content": document.content,
            "source_type": document.source_type,
            "record_id": document.record_id,
            "metadata": document.metadata,
        } for index, document in enumerate(documents)]

    def pending_outbox(self, limit):
        return self.pending[:limit]

    def mark_indexed(self, event_id):
        self.marked.append((event_id, "indexed"))

    def mark_failed(self, event_id, message):
        self.marked.append((event_id, message))


def stored(key, domain):
    return StoredDocument(
        document_key=key,
        domain_id=domain,
        source_id=f"source-{key}",
        document_id=f"document-{key}",
        chunk_index=1,
        title="Record",
        content="Evidence text",
        source_type="domain_upload",
        record_id=None,
        metadata={},
    )


def test_search_hydrates_from_postgres_and_fails_closed_on_cross_domain_hit():
    elasticsearch = FakeElasticsearch([
        {"_id": "health", "_score": 2.3},
        {"_id": "tourism", "_score": 2.2},
    ])
    service = DomainSearchService(
        repository=FakeRepository({
            "health": stored("health", "healthcare"),
            "tourism": stored("tourism", "tourism"),
        }),
        elasticsearch=elasticsearch,
    )

    hits = service.search("query", "healthcare", limit=7, min_score=1.1)

    assert [hit.id for hit in hits] == ["health"]
    assert hits[0].payload["domain_id"] == "healthcare"
    assert elasticsearch.query == ("query", "healthcare", {
        "limit": 7,
        "min_score": 1.1,
        "offset": 0,
        "metadata_filters": None,
    })


def test_search_refuses_to_query_when_elasticsearch_is_unavailable():
    elasticsearch = FakeElasticsearch([])
    elasticsearch.health = lambda: False
    service = DomainSearchService(repository=FakeRepository({}), elasticsearch=elasticsearch)

    try:
        service.search("query", "healthcare")
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("Expected unavailable search to fail closed")


def test_ingest_returns_only_after_elasticsearch_confirms_indexing():
    repository = FakeOutboxRepository()
    elasticsearch = FakeElasticsearch([])
    service = DomainSearchService(repository=repository, elasticsearch=elasticsearch)
    payload = {
        "domain_id": "tourism",
        "source_id": "source-a",
        "document_id": "doc-a",
        "chunk_index": 1,
        "title": "Guide",
        "text": "A tourism evidence record",
        "source_type": "domain_upload",
    }

    count = service.index_documents([("stable-key", payload)])

    assert count == 1
    assert repository.marked == [(1, "indexed")]
    assert elasticsearch.indexed_docs[0]["domain_id"] == "tourism"


def test_outbox_omits_verbose_record_fields_from_flattened_es_metadata():
    repository = FakeOutboxRepository()
    elasticsearch = FakeElasticsearch([])
    service = DomainSearchService(repository=repository, elasticsearch=elasticsearch)
    payload = {
        "domain_id": "legal_ai",
        "source_id": "catalog",
        "document_id": "procedure-1",
        "text": "Short searchable evidence",
        "record_fields": {"description": "x" * 60_000},
        "full_text": "x" * 60_000,
        "source_url": "https://example.gov.vn/procedure-1",
    }

    assert service.index_documents([("stable-key", payload)]) == 1

    indexed_metadata = elasticsearch.indexed_docs[0]["metadata"]
    assert "record_fields" not in indexed_metadata
    assert "full_text" not in indexed_metadata
    assert indexed_metadata["source_url"] == payload["source_url"]


def test_ingest_reports_pending_when_elasticsearch_item_fails():
    repository = FakeOutboxRepository()
    elasticsearch = FakeElasticsearch([])
    elasticsearch.bulk_index = lambda docs: [False]
    service = DomainSearchService(repository=repository, elasticsearch=elasticsearch)
    payload = {
        "domain_id": "tourism",
        "source_id": "source-a",
        "document_id": "doc-a",
        "text": "A tourism evidence record",
    }

    try:
        service.index_documents([("stable-key", payload)])
    except DocumentIndexingPending:
        assert repository.marked == [(1, "elasticsearch_item_error")]
    else:
        raise AssertionError("An unindexed document must not be reported as success")
