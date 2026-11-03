"""Small Elasticsearch REST client; PostgreSQL remains the source of truth."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


INDEX_NAME = os.getenv("ELASTICSEARCH_INDEX", "dai_documents_v1")


def build_index_definition() -> dict[str, Any]:
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "vi_folded": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "document_key": {"type": "keyword"},
                "domain_id": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "record_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "source_type": {"type": "keyword"},
                "metadata": {"type": "flattened"},
                "title": {"type": "text", "analyzer": "vi_folded"},
                "content": {"type": "text", "analyzer": "vi_folded"},
            }
        },
    }


class ElasticsearchUnavailable(RuntimeError):
    pass


class ElasticsearchClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        index_name: str = INDEX_NAME,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.base_url = (base_url or os.getenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200")).rstrip("/")
        self.index_name = index_name
        key = api_key if api_key is not None else os.getenv("ELASTICSEARCH_API_KEY")
        user = username if username is not None else os.getenv("ELASTICSEARCH_USERNAME")
        secret = password if password is not None else os.getenv("ELASTICSEARCH_PASSWORD")
        headers = {"Authorization": f"ApiKey {key}"} if key else {}
        auth = (user, secret) if user and secret else None
        self.client = client or httpx.Client(timeout=10.0, headers=headers, auth=auth)

    def cluster_health(self) -> bool:
        try:
            response = self.client.get(f"{self.base_url}/_cluster/health")
            return response.is_success
        except httpx.HTTPError:
            return False

    def health(self) -> bool:
        if not self.cluster_health():
            return False
        try:
            return self.client.head(f"{self.base_url}/{self.index_name}").is_success
        except httpx.HTTPError:
            return False

    def ensure_index(self) -> None:
        if not self.cluster_health():
            raise ElasticsearchUnavailable("Elasticsearch cluster is unavailable")
        response = self.client.put(
            f"{self.base_url}/{self.index_name}",
            json=build_index_definition(),
        )
        if response.is_success:
            return
        if response.status_code == 400:
            try:
                error_type = response.json().get("error", {}).get("type")
            except (ValueError, AttributeError):
                error_type = None
            if error_type == "resource_already_exists_exception":
                return
        raise ElasticsearchUnavailable(f"Could not create search index (HTTP {response.status_code})")

    def bulk_index(self, documents: list[dict[str, Any]]) -> list[bool]:
        if not documents:
            return []
        lines = []
        for document in documents:
            lines.append(json.dumps({"index": {"_index": self.index_name, "_id": document["document_key"]}}, ensure_ascii=False))
            lines.append(json.dumps({
                key: document.get(key)
                for key in (
                    "document_key", "domain_id", "source_id", "document_id", "record_id",
                    "chunk_index", "source_type", "title", "content", "metadata",
                )
            }, ensure_ascii=False))
        response = self.client.post(
            f"{self.base_url}/_bulk?refresh=wait_for",
            content="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        if not response.is_success:
            raise ElasticsearchUnavailable(f"Bulk indexing failed (HTTP {response.status_code})")
        result = response.json()
        return [
            not bool(item.get("index", {}).get("error"))
            for item in result.get("items", [])
        ]

    def search(
        self,
        query: str,
        domain_id: str,
        *,
        limit: int = 10,
        min_score: float | None = None,
        offset: int = 0,
        metadata_filters: dict[str, str | int | bool] | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [{"term": {"domain_id": domain_id}}]
        for key, value in (metadata_filters or {}).items():
            normalized_key = str(key).strip()
            if not normalized_key or normalized_key.startswith("_"):
                raise ValueError("Invalid metadata filter key")
            filters.append({"term": {f"metadata.{normalized_key}": value}})
        body: dict[str, Any] = {
            "size": max(1, min(int(limit), 100)),
            "from": max(0, min(int(offset), 10000)),
            "query": {
                "bool": {
                    "filter": filters,
                    "must": [{
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "record_id^2", "content"],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    }],
                }
            },
        }
        if min_score is not None:
            body["min_score"] = float(min_score)
        response = self.client.post(
            f"{self.base_url}/{self.index_name}/_search",
            json=body,
        )
        if not response.is_success:
            raise ElasticsearchUnavailable(f"Search failed (HTTP {response.status_code})")
        return response.json().get("hits", {}).get("hits", [])
