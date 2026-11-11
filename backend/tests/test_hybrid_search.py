import asyncio
from types import SimpleNamespace

import pytest

from backend.search.config import resolve_search_config
from backend.search.candidate import candidate_name
from backend.search.hybrid import HybridRetrievalUnavailable, fuse_rankings, retrieve_hybrid
from backend.rag.adapter import rag_result_to_evidence
from run_gateway_stack import build_child_environment


def hit(key, score=0.8, domain="tourism", text=None, source="source"):
    return SimpleNamespace(id=key, score=score, payload={
        "domain_id": domain, "source_id": source, "document_id": source,
        "text": text or key,
        "evidence_id": f"domain:{domain}:{source}:{key}", "source_type": "domain_upload",
    })


def test_config_and_launcher_preserve_explicit_rollback():
    assert resolve_search_config({"SEARCH_BACKEND": "hybrid"}) == ("hybrid", None)
    assert "SEARCH_BACKEND" not in build_child_environment({})
    assert build_child_environment({"SEARCH_BACKEND": "qdrant"})["SEARCH_BACKEND"] == "qdrant"


def test_postgres_catalog_title_is_used_as_candidate_name():
    assert candidate_name({"title": "Thủ tục đăng ký doanh nghiệp",
                           "source_type": "direct_catalog"}) == "Thủ tục đăng ký doanh nghiệp"
    assert candidate_name({"procedure_name": "Nguồn chính", "title": "Nguồn khác"}) == "Nguồn chính"
    assert candidate_name({"text": "Đừng tự tạo tên từ nội dung"}) == ""


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "<calibrated>"])
def test_invalid_score_is_not_silently_accepted(value):
    with pytest.raises(ValueError):
        resolve_search_config({"SEARCH_BACKEND": "hybrid", "ELASTICSEARCH_MIN_SCORE": value})


def test_rank_fusion_deduplicates_without_mixing_raw_scores_or_payloads():
    vector = hit("q1", text="same chunk")
    keyword = hit("es1", score=900, text="same chunk")
    result = fuse_rankings([
        ("qdrant", [vector, hit("q2")]),
        ("elasticsearch", [hit("es2", score=1000), keyword]),
    ], "tourism", limit=10, vector_min_score=.45)
    assert len(result) == 3
    assert result[0].id == "q1"
    assert result[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert result[0].payload == vector.payload
    assert len(result[0].retrieval_provenance) == 2
    assert vector.score == .8


def test_filtering_keeps_threshold_domains_and_distinct_chunks():
    result = fuse_rankings([
        ("qdrant", [hit("low", .44), hit("wrong", domain="healthcare"), hit("one"), hit("one")]),
        ("elasticsearch", [hit("two", 15), hit("missing", domain=None)]),
    ], "tourism", limit=10, vector_min_score=.45)
    assert {h.id for h in result} == {"one", "two"}
    assert all(len(h.retrieval_provenance) == 1 for h in result)


def test_parallel_retrieval_and_partial_failure_remain_explicit():
    def failed():
        raise ConnectionError("unavailable")
    async def run():
        results, status = await retrieve_hybrid(
            query="query", domain_id="tourism", vector_search=failed,
            keyword_search=lambda: [hit("es", 4)], limit=5, vector_min_score=.45,
        )
        assert [h.id for h in results] == ["es"]
        assert status["qdrant"]["status"] == "failed"
        assert status["elasticsearch"]["candidate_count"] == 1
        with pytest.raises(HybridRetrievalUnavailable, match="Both hybrid") as failure:
            await retrieve_hybrid(query="query", domain_id="tourism",
                vector_search=failed, keyword_search=failed, limit=5, vector_min_score=.45)
        assert failure.value.diagnostics["qdrant"]["status"] == "failed"
        assert failure.value.diagnostics["elasticsearch"]["status"] == "failed"
    asyncio.run(run())


def test_empty_candidates_do_not_fabricate_evidence():
    assert fuse_rankings([("qdrant", []), ("elasticsearch", [])],
                         "tourism", limit=10, vector_min_score=.45) == []


def test_identical_text_in_distinct_records_is_not_collapsed():
    a, b = hit("a", text="same"), hit("b", text="same")
    a.payload["record_id"], b.payload["record_id"] = "one", "two"
    results = fuse_rankings([("qdrant", [a]), ("elasticsearch", [b])],
                            "tourism", limit=10, vector_min_score=.45)
    assert len(results) == 2


def test_real_shape_nested_record_id_merges_with_postgres_hydrated_record():
    qdrant = hit("point-1", text="same record")
    qdrant.payload["record_fields"] = {"record_id": "HC-F-008"}
    elastic = hit("domain_documents_healthcare:point-1", score=13, text="same record")
    elastic.payload["record_id"] = "HC-F-008"
    # The actual Qdrant/ES point pair has one canonical Evidence ID.
    elastic.payload["evidence_id"] = qdrant.payload["evidence_id"]
    fused = fuse_rankings([("qdrant", [qdrant]), ("elasticsearch", [elastic])],
                          "tourism", limit=10, vector_min_score=.45)
    assert len(fused) == 1
    assert fused[0].payload["evidence_id"] == qdrant.payload["evidence_id"]
    assert {p["backend"] for p in fused[0].retrieval_provenance} == {"qdrant", "elasticsearch"}


def test_misassigned_education_evidence_is_excluded_from_education_results():
    record = hit("misassigned", domain="education")
    record.payload["evidence_id"] = "domain:healthcare:source:191"
    assert fuse_rankings([("elasticsearch", [record])], "education",
                         limit=10, vector_min_score=.45) == []


def test_fused_provenance_reaches_evidence_without_changing_canonical_id():
    point = hit("point-1")
    fused = fuse_rankings([("qdrant", [point])], "tourism", limit=5,
                          vector_min_score=.45)[0]
    evidence = rag_result_to_evidence({**fused.payload,
        "point_id": fused.id, "retrieval_provenance": fused.retrieval_provenance})
    assert evidence.evidence_id == point.payload["evidence_id"]
    assert evidence.metadata["retrieval_provenance"][0]["backend"] == "qdrant"


def test_timeout_does_not_fabricate_success(monkeypatch):
    # Deterministic scheduling: test the timeout boundary without depending on
    # Windows worker-pool cold-start latency.
    async def fake_thread(callback):
        if callback == "slow":
            await asyncio.sleep(60)
        return []
    monkeypatch.setattr("backend.search.hybrid.asyncio.to_thread", fake_thread)
    async def run():
        hits, status = await retrieve_hybrid(
            query="q", domain_id="tourism", vector_search="slow",
            keyword_search=lambda: [], limit=5, vector_min_score=.45, timeout=.01,
        )
        assert hits == []
        assert status["qdrant"]["error_type"] == "TimeoutError"
    asyncio.run(run())
