"""Create the versioned Elasticsearch mapping without indexing documents."""

from backend.search.elasticsearch import ElasticsearchClient


if __name__ == "__main__":
    client = ElasticsearchClient()
    if not client.cluster_health():
        raise SystemExit("Elasticsearch is unavailable; index was not created")
    client.ensure_index()
    print(f"Elasticsearch index is ready: {client.index_name}")
