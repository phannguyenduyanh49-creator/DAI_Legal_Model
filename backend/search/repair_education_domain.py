"""One-off, guarded repair for the confirmed mixed Healthcare/Education upload.

Dry-run creates an exclusive local backup. Apply requires that exact backup and
revalidates every live record before writing. This never touches legal data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from qdrant_client import QdrantClient, models
from sqlalchemy import text

from backend.catalog.database import engine as catalog_engine
from backend.search.elasticsearch import ElasticsearchClient
from backend.search.repository import PostgresDocumentRepository
from backend.search.service import DomainSearchService


SOURCE_COLLECTION = "domain_documents_healthcare"
TARGET_COLLECTION = "domain_documents_education"
EDUCATION_SHEET = "Giao_duc_100"
README_SHEETS = {"README_Y_te", "README_Giao_duc"}
EXPECTED_EDUCATION = 100
EXPECTED_README = 9


def repaired_payload(point) -> dict:
    """Accept only the exact known upload identity; preserve source and text."""
    old = dict(point.payload or {})
    if old.get("sheet_name") != EDUCATION_SHEET or old.get("domain_id") != "healthcare":
        raise ValueError("Point is not a known misassigned Education record")
    document_id = str(old.get("document_id") or "")
    chunk_index = old.get("chunk_index")
    if (
        not document_id
        or old.get("source_id") != document_id
        or not isinstance(chunk_index, int)
        or chunk_index < 1
        or old.get("evidence_id") != f"domain:healthcare:{document_id}:{chunk_index}"
        or not str(old.get("text") or "").strip()
    ):
        raise ValueError("Education point has ambiguous source or Evidence identity")
    old["domain_id"] = "education"
    old["evidence_id"] = f"domain:education:{document_id}:{chunk_index}"
    return old


def scan_source(client: QdrantClient):
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=SOURCE_COLLECTION, limit=256, offset=offset,
            with_payload=True, with_vectors=True,
        )
        points.extend(batch)
        if offset is None:
            break
    education = [p for p in points if (p.payload or {}).get("sheet_name") == EDUCATION_SHEET]
    readme = [p for p in points if (p.payload or {}).get("sheet_name") in README_SHEETS]
    healthcare = [p for p in points if (p.payload or {}).get("sheet_name") == "Y_te_100"]
    if (len(points), len(education), len(readme), len(healthcare)) != (209, 100, 9, 100):
        raise ValueError("Source collection no longer matches the reviewed 209-point layout")
    for point in education:
        repaired_payload(point)
        if not isinstance(point.vector, list) or len(point.vector) != 1024:
            raise ValueError("Education point has an unexpected vector format/dimension")
    for point in readme:
        if (point.payload or {}).get("domain_id") != "healthcare":
            raise ValueError("README point domain changed")
    return education, readme


def search_snapshot(education, repo, es):
    keys = [f"{SOURCE_COLLECTION}:{point.id}" for point in education]
    documents = repo.get_by_keys(keys)
    if set(documents) != set(keys):
        raise ValueError("PostgreSQL does not contain exactly the 100 Education keys")
    response = es.client.post(
        f"{es.base_url}/{es.index_name}/_mget", json={"ids": keys},
    )
    response.raise_for_status()
    es_docs = {item["_id"]: item["_source"] for item in response.json()["docs"] if item.get("found")}
    if set(es_docs) != set(keys):
        raise ValueError("Elasticsearch does not contain exactly the 100 Education keys")
    for point in education:
        key = f"{SOURCE_COLLECTION}:{point.id}"
        old = point.payload
        pg = documents[key]
        indexed = es_docs[key]
        if (
            pg.domain_id != "education"
            or indexed.get("domain_id") != "education"
            or pg.document_id != old.get("document_id")
            or pg.source_id != old.get("source_id")
            or pg.content != old.get("text")
            or pg.metadata.get("evidence_id") != old.get("evidence_id")
            or (indexed.get("metadata") or {}).get("evidence_id") != old.get("evidence_id")
        ):
            raise ValueError(f"Cross-store identity differs for point {point.id}")
    return documents, es_docs


def point_snapshot(point) -> dict:
    return {"id": str(point.id), "vector": point.vector, "payload": dict(point.payload)}


def digest(data) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_backup(directory: Path, education, readme, documents, es_docs):
    directory.mkdir(parents=True, exist_ok=False)
    points_snapshot = sorted(
        [point_snapshot(p) for p in education + readme], key=lambda item: item["id"]
    )
    postgres_snapshot = {key: doc.to_payload() for key, doc in documents.items()}
    snapshots = {
        "source_points.json": points_snapshot,
        "postgres_documents.json": postgres_snapshot,
        "elasticsearch_documents.json": es_docs,
        "manifest.json": {
            "migration_id": "repair_education_domain_v1",
            "source_collection": SOURCE_COLLECTION,
            "target_collection": TARGET_COLLECTION,
            "education_point_ids": sorted(str(p.id) for p in education),
            "readme_point_ids": sorted(str(p.id) for p in readme),
            "point_digest": digest(points_snapshot),
            "postgres_digest": digest(postgres_snapshot),
            "elasticsearch_digest": digest(es_docs),
        },
    }
    for name, data in snapshots.items():
        with (directory / name).open("x", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False)


def load_backup(directory: Path):
    names = ("manifest", "source_points", "postgres_documents", "elasticsearch_documents")
    items = {
        name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in names
    }
    manifest = items["manifest"]
    if manifest.get("migration_id") != "repair_education_domain_v1":
        raise ValueError("Unknown migration backup")
    for name, field in (
        ("source_points", "point_digest"),
        ("postgres_documents", "postgres_digest"),
        ("elasticsearch_documents", "elasticsearch_digest"),
    ):
        if digest(items[name]) != manifest.get(field):
            raise ValueError(f"Backup integrity failed: {name}")
    return items


def verify_target(client, education):
    ids = [point.id for point in education]
    found = client.retrieve(TARGET_COLLECTION, ids=ids, with_payload=True, with_vectors=False)
    if len(found) != len(education) or client.get_collection(TARGET_COLLECTION).points_count != 100:
        raise ValueError("Education destination point count is not exactly 100")
    expected = {str(point.id): repaired_payload(point) for point in education}
    if any((point.payload or {}) != expected.get(str(point.id)) for point in found):
        raise ValueError("Education destination payloads do not match the reviewed source")


def apply(client, education, readme, documents, es):
    if client.collection_exists(TARGET_COLLECTION):
        if client.get_collection(TARGET_COLLECTION).points_count != 0:
            raise ValueError("Education destination is not empty; refusing to overwrite it")
    else:
        client.create_collection(
            TARGET_COLLECTION,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )
    result = client.upsert(
        TARGET_COLLECTION,
        points=[models.PointStruct(id=p.id, vector=p.vector, payload=repaired_payload(p)) for p in education],
        wait=True,
    )
    if str(getattr(result.status, "value", result.status)) != "completed":
        raise RuntimeError("Qdrant did not confirm Education indexing")
    verify_target(client, education)

    store = DomainSearchService(
        repository=PostgresDocumentRepository(engine=catalog_engine), elasticsearch=es,
    )
    rows = []
    for point in education:
        key = f"{SOURCE_COLLECTION}:{point.id}"
        rows.append((key, {**documents[key].to_payload(), "evidence_id": repaired_payload(point)["evidence_id"]}))
    if store.index_documents(rows) != 100:
        raise RuntimeError("PostgreSQL/Elasticsearch did not confirm all 100 records")
    updated, indexed = search_snapshot_repaired(education, store.repository, es)
    if len(updated) != 100 or len(indexed) != 100:
        raise RuntimeError("PostgreSQL/Elasticsearch reconciliation failed")

    old_ids = [p.id for p in education + readme]
    deleted = client.delete(
        SOURCE_COLLECTION,
        points_selector=models.PointIdsList(points=old_ids), wait=True,
    )
    if str(getattr(deleted.status, "value", deleted.status)) != "completed":
        raise RuntimeError("Qdrant did not confirm removal of misassigned/instruction points")
    remaining = client.retrieve(SOURCE_COLLECTION, ids=old_ids, with_payload=False)
    if remaining or client.get_collection(SOURCE_COLLECTION).points_count != 100:
        raise RuntimeError("Healthcare source did not reconcile to 100 points")
    verify_target(client, education)


def search_snapshot_repaired(education, repo, es):
    keys = [f"{SOURCE_COLLECTION}:{point.id}" for point in education]
    documents = repo.get_by_keys(keys)
    response = es.client.post(f"{es.base_url}/{es.index_name}/_mget", json={"ids": keys})
    response.raise_for_status()
    indexed = {item["_id"]: item["_source"] for item in response.json()["docs"] if item.get("found")}
    if len(documents) != 100 or len(indexed) != 100:
        raise ValueError("Education records missing after repair")
    for point in education:
        key = f"{SOURCE_COLLECTION}:{point.id}"
        expected = repaired_payload(point)["evidence_id"]
        if (
            documents[key].domain_id != "education"
            or documents[key].metadata.get("evidence_id") != expected
            or indexed[key]["domain_id"] != "education"
            or (indexed[key].get("metadata") or {}).get("evidence_id") != expected
        ):
            raise ValueError(f"Education identity still inconsistent for {point.id}")
    return documents, indexed


def search_documents(es, keys):
    response = es.client.post(
        f"{es.base_url}/{es.index_name}/_mget", json={"ids": keys},
    )
    response.raise_for_status()
    return {
        item["_id"]: item["_source"]
        for item in response.json()["docs"] if item.get("found")
    }


def delete_search_keys(repo, es, keys):
    """Delete only keys that have already been backed up and replaced."""
    lines = "".join(
        json.dumps({"delete": {"_index": es.index_name, "_id": key}}) + "\n"
        for key in keys
    )
    response = es.client.post(
        f"{es.base_url}/_bulk?refresh=wait_for", content=lines,
        headers={"Content-Type": "application/x-ndjson"},
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    if len(items) != len(keys) or any(item.get("delete", {}).get("result") != "deleted" for item in items):
        raise RuntimeError("Elasticsearch did not confirm every exact-key deletion")
    with repo.engine.begin() as connection:
        deleted = connection.execute(
            text("DELETE FROM domain_search_documents WHERE document_key = ANY(:keys)"),
            {"keys": keys},
        )
        if deleted.rowcount != len(keys):
            raise RuntimeError("PostgreSQL exact-key deletion count differs")


def finalize_keys(client, backup, es):
    """Move search document keys to their actual Education collection identity."""
    if not client.collection_exists(TARGET_COLLECTION):
        raise ValueError("Education collection is missing")
    manifest = backup["manifest"]
    ids = manifest["education_point_ids"]
    old_keys = [f"{SOURCE_COLLECTION}:{point_id}" for point_id in ids]
    new_keys = [f"{TARGET_COLLECTION}:{point_id}" for point_id in ids]
    repo = PostgresDocumentRepository(engine=catalog_engine)
    old_docs = repo.get_by_keys(old_keys)
    if set(old_docs) != set(old_keys) or repo.get_by_keys(new_keys):
        raise ValueError("Search keys differ from the reviewed pre-finalization state")
    old_es = search_documents(es, old_keys)
    if set(old_es) != set(old_keys) or search_documents(es, new_keys):
        raise ValueError("Elasticsearch keys differ from the reviewed pre-finalization state")
    for point_id in ids:
        old_key = f"{SOURCE_COLLECTION}:{point_id}"
        expected = f"domain:education:{old_docs[old_key].document_id}:{old_docs[old_key].chunk_index}"
        if (
            old_docs[old_key].domain_id != "education"
            or old_docs[old_key].metadata.get("evidence_id") != expected
            or (old_es[old_key].get("metadata") or {}).get("evidence_id") != expected
        ):
            raise ValueError("Education identity has not been repaired in both search stores")
    store = DomainSearchService(repository=repo, elasticsearch=es)
    rows = [
        (new_key, old_docs[old_key].to_payload())
        for old_key, new_key in zip(old_keys, new_keys)
    ]
    if store.index_documents(rows) != EXPECTED_EDUCATION:
        raise RuntimeError("Canonical Education indexing was not confirmed")
    new_docs = repo.get_by_keys(new_keys)
    new_es = search_documents(es, new_keys)
    if set(new_docs) != set(new_keys) or set(new_es) != set(new_keys):
        raise RuntimeError("Canonical Education keys are missing")
    for old_key, new_key in zip(old_keys, new_keys):
        if (
            new_docs[new_key].to_payload() != old_docs[old_key].to_payload()
            or new_es[new_key].get("domain_id") != "education"
        ):
            raise RuntimeError("Canonical Education payload differs from the source")
    delete_search_keys(repo, es, old_keys)
    if repo.get_by_keys(old_keys) or search_documents(es, old_keys):
        raise RuntimeError("Old collection search keys remain")
    if client.get_collection(SOURCE_COLLECTION).points_count != 100:
        raise RuntimeError("Healthcare Qdrant count changed during finalization")
    print("FINALIZED: 100 canonical Education search keys; old keys removed")


def rollback(client, backup, es):
    points = backup["source_points"]
    manifest = backup["manifest"]
    if len(points) != EXPECTED_EDUCATION + EXPECTED_README:
        raise ValueError("Backup point count is not the reviewed 109")
    education_ids = set(manifest["education_point_ids"])
    if not client.collection_exists(TARGET_COLLECTION):
        raise ValueError("Education destination does not exist; nothing to roll back")
    current = client.retrieve(
        TARGET_COLLECTION, ids=list(education_ids), with_payload=True, with_vectors=False,
    )
    if len(current) != EXPECTED_EDUCATION:
        raise ValueError("Education destination differs from the applied migration")
    old_by_id = {item["id"]: item for item in points}
    if any((p.payload or {}) != repaired_payload(SimpleNamespace(payload=old_by_id[str(p.id)]["payload"])) for p in current):
        raise ValueError("Education destination changed since migration")
    restored = client.upsert(
        SOURCE_COLLECTION,
        points=[models.PointStruct(id=item["id"], vector=item["vector"], payload=item["payload"]) for item in points],
        wait=True,
    )
    if str(getattr(restored.status, "value", restored.status)) != "completed":
        raise RuntimeError("Qdrant source restoration was not confirmed")
    store = DomainSearchService(
        repository=PostgresDocumentRepository(engine=catalog_engine), elasticsearch=es,
    )
    originals = backup["postgres_documents"]
    if store.index_documents(list(originals.items())) != EXPECTED_EDUCATION:
        raise RuntimeError("Search record restoration was not confirmed")
    canonical_keys = [f"{TARGET_COLLECTION}:{point_id}" for point_id in education_ids]
    if store.repository.get_by_keys(canonical_keys):
        if len(store.repository.get_by_keys(canonical_keys)) != EXPECTED_EDUCATION:
            raise RuntimeError("Canonical search keys changed since migration")
        delete_search_keys(store.repository, es, canonical_keys)
    removed = client.delete(
        TARGET_COLLECTION,
        points_selector=models.PointIdsList(points=list(education_ids)), wait=True,
    )
    if str(getattr(removed.status, "value", removed.status)) != "completed":
        raise RuntimeError("Education destination removal was not confirmed")
    if client.get_collection(SOURCE_COLLECTION).points_count != 209:
        raise RuntimeError("Healthcare point count did not restore to 209")
    print("ROLLED BACK: 109 source points restored, 100 search records restored, 100 destination points removed")


def already_applied(client, backup, es) -> bool:
    """Permit a second apply to succeed only when the complete final state matches."""
    if not client.collection_exists(TARGET_COLLECTION):
        return False
    if (
        client.get_collection(SOURCE_COLLECTION).points_count != 100
        or client.get_collection(TARGET_COLLECTION).points_count != 100
    ):
        return False
    manifest = backup["manifest"]
    ids = manifest["education_point_ids"]
    originals = {
        item["id"]: item for item in backup["source_points"]
        if item["id"] in ids
    }
    target = client.retrieve(TARGET_COLLECTION, ids=ids, with_payload=True, with_vectors=False)
    if len(target) != 100 or any(
        point.payload != repaired_payload(SimpleNamespace(payload=originals[str(point.id)]["payload"]))
        for point in target
    ):
        return False
    if client.retrieve(
        SOURCE_COLLECTION,
        ids=ids + manifest["readme_point_ids"], with_payload=False,
    ):
        return False
    repo = PostgresDocumentRepository(engine=catalog_engine)
    new_keys = [f"{TARGET_COLLECTION}:{point_id}" for point_id in ids]
    old_keys = [f"{SOURCE_COLLECTION}:{point_id}" for point_id in ids]
    documents = repo.get_by_keys(new_keys)
    indexed = search_documents(es, new_keys)
    if (
        set(documents) != set(new_keys)
        or set(indexed) != set(new_keys)
        or repo.get_by_keys(old_keys)
        or search_documents(es, old_keys)
    ):
        return False
    return all(
        documents[key].domain_id == "education"
        and documents[key].metadata.get("evidence_id") == (
            f"domain:education:{documents[key].document_id}:{documents[key].chunk_index}"
        )
        and (indexed[key].get("metadata") or {}).get("evidence_id") == documents[key].metadata.get("evidence_id")
        for key in new_keys
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if sum((args.apply, args.rollback, args.finalize)) > 1:
        parser.error("Choose only --apply, --finalize or --rollback")
    client = QdrantClient(url="http://127.0.0.1:6333")
    es = ElasticsearchClient()
    if args.rollback:
        rollback(client, load_backup(args.backup_dir), es)
        return
    if args.finalize:
        finalize_keys(client, load_backup(args.backup_dir), es)
        return
    if args.apply and already_applied(client, load_backup(args.backup_dir), es):
        print("ALREADY APPLIED: all 100 Education records and search keys match the reviewed backup")
        return
    repo = PostgresDocumentRepository(engine=catalog_engine)
    education, readme = scan_source(client)
    documents, es_docs = search_snapshot(education, repo, es)
    if args.apply:
        backup = load_backup(args.backup_dir)
        manifest = backup["manifest"]
        if (
            manifest.get("education_point_ids") != sorted(str(p.id) for p in education)
            or manifest.get("readme_point_ids") != sorted(str(p.id) for p in readme)
            or manifest.get("point_digest") != digest(sorted(
                [point_snapshot(p) for p in education + readme], key=lambda item: item["id"]
            ))
            or manifest.get("postgres_digest") != digest({
                key: doc.to_payload() for key, doc in documents.items()
            })
            or manifest.get("elasticsearch_digest") != digest(es_docs)
        ):
            raise ValueError("Live data differs from the reviewed backup")
        apply(client, education, readme, documents, es)
        print("APPLIED: 100 Education points relocated; 9 instruction points removed; 100 search records repaired")
    else:
        write_backup(args.backup_dir, education, readme, documents, es_docs)
        print(f"DRY RUN: 100 Education, 9 README, 100 search records; backup: {args.backup_dir}")


if __name__ == "__main__":
    main()
