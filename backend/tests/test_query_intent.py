from backend.rag.query_intent import (
    requires_multiple_evidence,
    retrieval_breadth,
    should_use_exact_match_shortcut,
)


def test_multi_evidence_intent_recognizes_vietnamese_and_english_comparisons():
    assert requires_multiple_evidence("Công ty cổ phần khác công ty TNHH ở điểm nào?")
    assert requires_multiple_evidence("So sánh hai loại hình doanh nghiệp")
    assert requires_multiple_evidence("Compare these product return policies")


def test_multi_evidence_intent_recognizes_lists_and_synthesis():
    assert requires_multiple_evidence("Hồ sơ gồm những giấy tờ gì?")
    assert requires_multiple_evidence("Liệt kê các bước cần thực hiện")
    assert requires_multiple_evidence("Tổng hợp thông tin từ các nguồn")


def test_ordinary_single_subject_query_keeps_standard_retrieval_breadth():
    assert not requires_multiple_evidence("Thủ tục thành lập công ty cổ phần là gì?")
    assert not requires_multiple_evidence("Địa điểm du lịch ở Huế")


def test_multi_evidence_intent_expands_candidates_without_disabling_reranking():
    plan = retrieval_breadth("Công ty cổ phần khác công ty TNHH ở điểm nào?")
    assert plan.multi_evidence is True
    assert plan.local_candidate_limit == 20
    assert plan.rerank_result_limit == 6
    assert plan.apply_rerank_margin is False

    ordinary = retrieval_breadth("Thủ tục thành lập công ty cổ phần là gì?")
    assert ordinary.local_candidate_limit == 10
    assert ordinary.rerank_result_limit == 3
    assert ordinary.apply_rerank_margin is True


def test_exact_match_shortcut_is_skipped_for_multi_evidence_or_list_queries():
    assert should_use_exact_match_shortcut(
        "Thủ tục thành lập công ty cổ phần là gì?",
        structured_list_query=False,
    )
    assert not should_use_exact_match_shortcut(
        "Công ty cổ phần khác công ty TNHH ở điểm nào?",
        structured_list_query=False,
    )
    assert not should_use_exact_match_shortcut(
        "Tourism query",
        structured_list_query=True,
    )
