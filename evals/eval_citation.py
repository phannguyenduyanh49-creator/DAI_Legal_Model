# -*- coding: utf-8 -*-
"""
Script Đánh giá Citation Integrity & Gắn cờ Kiểm chứng Ngữ nghĩa Thủ công
"""
import json
import os
from typing import List, Dict, Any

def evaluate_citations_and_flag_semantic(golden_data: List[Dict[str, Any]], system_responses: Dict[str, Dict[str, Any]]):
    """
    1. Đánh giá tính hợp lệ về cấu trúc Citation và ID tồn tại.
    2. Xuất bảng gom nhóm yêu cầu Con người Đánh giá Ngữ nghĩa (Semantic Support).
    """
    print("\n=================== CẾT QUẢ ĐÁNH GIÁ CITATION & CẢNH BÁO NGỮ NGHĨA ===================")
    
    human_review_queue = []

    for item in golden_data:
        q_id = item["id"]
        query = item["query"]
        expected_keypoints = item["expected_keypoints"]
        
        resp = system_responses.get(q_id, {})
        raw_answer = resp.get("answer", "")
        cited_ids = resp.get("cited_evidence_ids", [])
        evidence_status = resp.get("evidence_status", "UNKNOWN")

        # 1. Kiểm tra ID Citation có khớp với danh sách Evidence thực tế không
        valid_citations = [cid for cid in cited_ids if cid in item["expected_evidence_ids"]]
        has_valid_citation_id = len(valid_citations) > 0

        print(f"\n[QUERY {q_id}]: {query}")
        print(f"  - Trạng thái Evidence    : {evidence_status}")
        print(f"  - Cited IDs từ System    : {cited_ids}")
        print(f"  - Citation Match ID status: {'ĐẠT (Khớp ID)' if has_valid_citation_id else 'KHÔNG KHỚP ID'}")
        
        # NGUYÊN TẮC VÀNG: Nhắc nhở không tự ý công nhận câu trả lời đúng
        print("  - [CẢNH BÁO] Việc Citation trùng ID chưa chứng minh câu trả lời đúng về mặt pháp lý.")

        # 2. Đưa vào hàng chờ Người đánh giá Ngữ nghĩa
        human_review_queue.append({
            "query_id": q_id,
            "query": query,
            "system_answer": raw_answer,
            "cited_ids": cited_ids,
            "expected_keypoints": expected_keypoints,
            "semantic_pass": None  # Dành cho chuyên gia tích chọn True/False
        })

    # Xuất file JSON để Người đánh giá (Annotator) mở ra chấm thủ công
    review_filepath = os.path.join(os.path.dirname(__file__), "human_semantic_review_queue.json")
    with open(review_filepath, "w", encoding="utf-8") as f:
        json.dump(human_review_queue, f, ensure_ascii=False, indent=2)

    print("\n-------------------------------------------------------------------")
    print(f"-> Đã xuất {len(human_review_queue)} câu hỏi ra file: 'evals/human_semantic_review_queue.json'")
    print("-> BẮT BỘC: Người đánh giá chuyên môn cần mở file này để đối chiếu 'system_answer' với 'expected_keypoints'.\n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "golden_set.json"), "r", encoding="utf-8") as f:
        golden = json.load(f)

    # Giả lập response nhận từ Gateway
    mock_responses = {
        "GOLDEN_001": {
            "answer": "Thời hạn cấp giấy phép xây dựng nhà ở riêng lẻ là không quá 15 ngày đối với đô thị.",
            "cited_evidence_ids": ["doc_luat_xay_dung_2014_d102_k1_b"],
            "evidence_status": "SUCCESS"
        },
        "GOLDEN_002": {
            "answer": "Hệ thống chưa tìm thấy đủ căn cứ pháp lý.",
            "cited_evidence_ids": [],
            "evidence_status": "MISSING"
        }
    }

    evaluate_citations_and_flag_semantic(golden, mock_responses)