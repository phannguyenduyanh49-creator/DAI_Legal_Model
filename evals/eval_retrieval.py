# -*- coding: utf-8 -*-
"""
Script Đánh giá Độc lập Retrieval (RAG Offline Evaluation)
Không sửa đổi code runtime.
"""
import json
import os
from typing import List, Dict, Any

def evaluate_retrieval(golden_data: List[Dict[str, Any]], retrieved_results: Dict[str, List[str]], top_k: int = 5) -> Dict[str, float]:
    """
    So sánh danh sách Evidence ID trích xuất thực tế với expected_evidence_ids trong Golden Set.
    """
    total_queries = len(golden_data)
    hits = 0
    total_precision = 0.0
    total_recall = 0.0

    print(f"\n=================== CẾT QUẢ ĐÁNH GIÁ RETRIEVAL (TOP-K = {top_k}) ===================")

    for item in golden_data:
        q_id = item["id"]
        expected_ids = set(item["expected_evidence_ids"])
        retrieved_ids = set(retrieved_results.get(q_id, [])[:top_k])

        # Tính toán
        intersection = expected_ids.intersection(retrieved_ids)
        
        is_hit = len(intersection) > 0
        if is_hit:
            hits += 1

        precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(intersection) / len(expected_ids) if expected_ids else 0.0

        total_precision += precision
        total_recall += recall

        status_str = "HIT" if is_hit else "MISS"
        print(f"[{status_str}] Query {q_id}: Trùng khớp {len(intersection)}/{len(expected_ids)} Evidence kỳ vọng.")

    avg_hit_rate = hits / total_queries if total_queries > 0 else 0.0
    avg_precision = total_precision / total_queries if total_queries > 0 else 0.0
    avg_recall = total_recall / total_queries if total_queries > 0 else 0.0

    metrics = {
        "hit_rate": avg_hit_rate,
        "mean_precision": avg_precision,
        "mean_recall": avg_recall
    }

    print("-------------------------------------------------------------------")
    print(f"-> Hit Rate@{top_k}      : {avg_hit_rate * 100:.2f}%")
    print(f"-> Mean Precision@{top_k} : {avg_precision * 100:.2f}%")
    print(f"-> Mean Recall@{top_k}    : {avg_recall * 100:.2f}%\n")

    return metrics

if __name__ == "__main__":
    # Test mẫu với dữ liệu mock retrieval
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "golden_set.json"), "r", encoding="utf-8") as f:
        golden = json.load(f)

    # Giả lập kết quả trả về từ RAG
    mock_retrieved = {
        "GOLDEN_001": ["doc_luat_xay_dung_2014_d102_k1_b", "doc_other_002"],
        "GOLDEN_002": ["doc_nd_101_2024_d28_k1"],
        "GOLDEN_003": ["doc_wrong_id_999"]
    }

    evaluate_retrieval(golden, mock_retrieved, top_k=3)