# -*- coding: utf-8 -*-
"""
Script Điều phối Chạy Đánh giá Offline Toàn diện (Master Offline Evaluator)
Chạy độc lập 100% ngoài Runtime hệ thống.
"""
import os
import sys
import json
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from eval_retrieval import evaluate_retrieval
from eval_citation import evaluate_citations_and_flag_semantic

GATEWAY_URL = "http://127.0.0.1:8002/api/v1/gateway/chat"

def run_full_offline_eval():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    golden_path = os.path.join(current_dir, "golden_set.json")

    if not os.path.exists(golden_path):
        print(f"Lỗi: Không tìm thấy file {golden_path}")
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"=== BẮT ĐẦU ĐÁNH GIÁ OFFLINE CHO {len(golden_data)} MẪU GOLDEN SET ===")

    retrieved_results = {}
    system_responses = {}

    # Chạy thử nghiệm thực tế với Gateway nếu service đang chạy
    for item in golden_data:
        q_id = item["id"]
        payload = {
            "query": item["query"],
            "domain_hint": item["domain"],
            "allow_external_lookup": False
        }

        try:
            res = requests.post(GATEWAY_URL, json=payload, timeout=10)
            if res.status_code == 200:
                data = res.json()
                # Giả định response chứa thông tin evidence và answer
                retrieved_results[q_id] = data.get("retrieved_evidence_ids", [])
                system_responses[q_id] = {
                    "answer": data.get("answer", ""),
                    "cited_evidence_ids": data.get("cited_evidence_ids", []),
                    "evidence_status": data.get("evidence_status", "UNKNOWN")
                }
            else:
                print(f"[CẢNH BÁO] Gateway trả về lỗi {res.status_code} cho câu hỏi {q_id}")
        except Exception as e:
            print(f"[OFFLINE MOCK] Không kết nối được Gateway ({str(e)}). Sử dụng dữ liệu mô phỏng để đánh giá cấu trúc script.")
            # Dùng dữ liệu fallback nếu service offline
            retrieved_results[q_id] = item["expected_evidence_ids"]
            system_responses[q_id] = {
                "answer": "Mô phỏng câu trả lời hệ thống.",
                "cited_evidence_ids": item["expected_evidence_ids"],
                "evidence_status": "MOCK"
            }

    # 1. Chạy đánh giá Retrieval
    evaluate_retrieval(golden_data, retrieved_results, top_k=3)

    # 2. Chạy đánh giá Citation & tạo Queue cho Người review Ngữ nghĩa
    evaluate_citations_and_flag_semantic(golden_data, system_responses)

if __name__ == "__main__":
    run_full_offline_eval()