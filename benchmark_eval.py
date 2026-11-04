# -*- coding: utf-8 -*-

import math
from typing import List, Dict, Any, Set


def calculate_metrics_for_query(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5
) -> Dict[str, float]:
    """Tính các chỉ số đo lường cho 1 câu hỏi."""
    if k <= 0:
        raise ValueError("k must be positive")
    retrieved_k = list(dict.fromkeys(retrieved_ids))[:k]
    hits = [doc_id for doc_id in retrieved_k if doc_id in relevant_ids]
    num_hits = len(hits)

    # 1. Recall@K
    recall = num_hits / len(relevant_ids) if relevant_ids else 0.0

    # 2. Precision@K
    precision = num_hits / k

    # 3. MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_k, start=1):
        if doc_id in relevant_ids:
            mrr = 1.0 / rank
            break

    # 4. NDCG@K
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_k, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        f"Recall@{k}": recall,
        f"Precision@{k}": precision,
        f"MRR": mrr,
        f"NDCG@{k}": ndcg
    }


def run_benchmark_suite(
    retriever_instance: Any,
    eval_dataset: List[Dict[str, Any]],
    k: int = 5
) -> Dict[str, float]:
    """
    Chạy đánh giá trên toàn bộ tập Testset.
    
    :param eval_dataset: Danh sách các dict dạng:
           [{"query": "...", "relevant_ids": ["DOC_001", "DOC_002"]}, ...]
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not eval_dataset:
        raise ValueError("Evaluation set must contain at least one query")

    total_metrics = {f"Recall@{k}": 0.0, f"Precision@{k}": 0.0, "MRR": 0.0, f"NDCG@{k}": 0.0}
    num_queries = len(eval_dataset)

    for item in eval_dataset:
        query = item["query"]
        relevant_ids = set(item["relevant_ids"])
        if not query or not relevant_ids:
            raise ValueError("Each query needs text and at least one relevant ID")

        # Gọi Retriever
        results = retriever_instance.retrieve(query, top_k=k)
        retrieved_ids = []
        for doc in results:
            evidence_id = str(doc.get("evidence_id") or "").strip()
            if evidence_id:
                retrieved_ids.append(evidence_id)

        # Tính chỉ số
        q_metrics = calculate_metrics_for_query(retrieved_ids, relevant_ids, k=k)

        for key in total_metrics:
            total_metrics[key] += q_metrics[key]

    # Tính trung bình cộng (Mean)
    avg_metrics = {key: round(val / num_queries, 4) for key, val in total_metrics.items()}
    
    print(f"\n================ KẾT QUẢ BENCHMARK (N = {num_queries} Queries, K = {k}) ================")
    for metric_name, val in avg_metrics.items():
        print(f"  • {metric_name:<15}: {val:.4f} ({val * 100:.2f}%)")
    print("=========================================================================\n")

    return avg_metrics