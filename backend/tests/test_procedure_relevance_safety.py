import math
from types import SimpleNamespace

from backend.rag.candidate_diagnostics import retrieval_candidate_summary
from backend.rag.relevance import finite_retrieval_score, procedure_relevance


QUESTION = (
    "Người lao động đơn phương chấm dứt hợp đồng lao động không xác định "
    "thời hạn phải báo trước bao nhiêu ngày?"
)


def test_relevant_labor_title_survives_even_when_query_has_extra_qualifiers():
    result = procedure_relevance(QUESTION, "Đơn phương chấm dứt hợp đồng lao động")
    assert result["relevant"] is True
    assert result["match_type"] == "exact"


def test_unrelated_labor_procedure_from_live_a03_is_rejected():
    title = (
        "Hỗ trợ người lao động trong một số trường hợp rủi ro khi làm việc "
        "ở nước ngoài theo hợp đồng phải về nước trước thời hạn"
    )
    result = procedure_relevance(QUESTION, title)
    assert result["relevant"] is False
    assert result["query_coverage"] < 0.60


def test_nonfinite_or_malformed_retrieval_scores_fail_closed_and_log_safely():
    for value in (None, "not-a-score", float("nan"), math.inf, -math.inf, True):
        assert finite_retrieval_score(value) is None
        summary = retrieval_candidate_summary(
            SimpleNamespace(id="point", score=value, payload={}), "reject_invalid_score"
        )
        assert summary["retrieval_score"] is None
        assert summary["decision"] == "reject_invalid_score"
    assert finite_retrieval_score(0.584811) == 0.584811
