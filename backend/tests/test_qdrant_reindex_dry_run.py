from types import SimpleNamespace

import backend.search.reindex_qdrant as reindex


class FakeQdrant:
    def __init__(self, points):
        self.points = points

    def collection_exists(self, collection):
        return True

    def scroll(self, **kwargs):
        return self.points, None


def point(point_id, *, sheet, domain, record_id):
    return SimpleNamespace(
        id=point_id,
        payload={
            "domain_id": domain,
            "document_id": "workbook-id",
            "source_id": "workbook-id",
            "chunk_index": 1,
            "source_type": "domain_upload",
            "title": "Healthcare_Education_RAG_Datasets_100_each.xlsx",
            "sheet_name": sheet,
            "record_fields": {"record_id": record_id},
            "text": f"record contents {record_id}",
        },
    )


def test_dry_run_remaps_education_worksheet_and_excludes_readme(monkeypatch):
    monkeypatch.setattr(
        reindex,
        "_expected_domains_by_collection",
        lambda: {"domain_documents_healthcare": {"healthcare"}},
    )
    client = FakeQdrant([
        point("hc-1", sheet="Y_te_100", domain="healthcare", record_id="HC-1"),
        point("edu-1", sheet="Giao_duc_100", domain="healthcare", record_id="EDU-1"),
        point("readme-1", sheet="README_Giao_duc", domain="healthcare", record_id=""),
    ])

    records, counts = reindex.collect_records(client)

    assert [record.domain_id for record in records] == ["healthcare", "education"]
    assert counts["remapped_domain:healthcare:education"] == 1
    assert counts["excluded_instruction_sheet"] == 1


def test_dry_run_does_not_guess_domain_for_shared_legacy_collection(monkeypatch):
    monkeypatch.setattr(
        reindex,
        "_expected_domains_by_collection",
        lambda: {"legal_docs_bge_m3": {"legal_ai", "administrative_law"}},
    )
    legacy = SimpleNamespace(
        id="legal-point",
        payload={"source": "catalog", "text": "A legal record"},
    )
    client = FakeQdrant([legacy])

    records, counts = reindex.collect_records(client)

    assert records == []
    assert counts["missing_domain_metadata"] == 1
