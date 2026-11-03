import json

import httpx

from backend.search.elasticsearch import ElasticsearchClient, build_index_definition


def test_search_uses_domain_and_metadata_filters_and_configured_min_score():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"hits": {"hits": []}})

    client = ElasticsearchClient(
        "http://search.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.search(
        "weekend dermatology",
        "healthcare",
        limit=12,
        min_score=1.2,
        offset=4,
        metadata_filters={"record_type": "HC-F"},
    )

    assert seen["url"].endswith("/dai_documents_v1/_search")
    assert seen["body"]["from"] == 4
    assert seen["body"]["min_score"] == 1.2
    assert {"term": {"domain_id": "healthcare"}} in seen["body"]["query"]["bool"]["filter"]
    assert {"term": {"metadata.record_type": "HC-F"}} in seen["body"]["query"]["bool"]["filter"]


def test_mapping_preserves_domain_and_metadata_as_filterable_fields():
    mapping = build_index_definition()["mappings"]["properties"]
    assert mapping["domain_id"]["type"] == "keyword"
    assert mapping["metadata"]["type"] == "flattened"
    assert mapping["content"]["type"] == "text"


def test_invalid_metadata_filter_name_is_rejected_before_request():
    client = ElasticsearchClient(
        "http://search.test",
        client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"hits": {"hits": []}})
        )),
    )
    try:
        client.search("query", "tourism", metadata_filters={"_index": "other"})
    except ValueError as exc:
        assert "filter key" in str(exc)
    else:
        raise AssertionError("Expected an invalid metadata key to fail")
