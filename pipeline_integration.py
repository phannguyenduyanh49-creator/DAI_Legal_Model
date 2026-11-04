# -*- coding: utf-8 -*-
"""
Module: Pipeline Integration Example
Mục đích: Demo cách gọi HybridRetriever trong luồng RAG chính mà không gãy hệ thống.
"""

import os
from hybrid_retriever import HybridRetriever

# -----------------------------------------------------------------------------
# 1. MOCK RETRIEVERS (Thay thế bằng Qdrant & BM25 thật trong dự án)
# -----------------------------------------------------------------------------
class MockDenseRetriever:
    def search(self, query: str, top_k: int = 5):
        return [
            {"evidence_id": "DOC_137_2020_ND_CP#SEC_15", "content": "Thủ tục cấp GCN huấn luyện an toàn pháo hoa...", "metadata": {"domain": "Pháp lý"}},
            {"evidence_id": "DOC_56_2023_ND_CP#SEC_01", "content": "Sửa đổi nộp hồ sơ trực tuyến qua Cổng DVC...", "metadata": {"domain": "Pháp lý"}},
        ]

class MockBM25Retriever:
    def search(self, query: str, top_k: int = 5):
        return [
            {"evidence_id": "DOC_137_2020_ND_CP#SEC_15", "content": "Thủ tục cấp GCN huấn luyện an toàn pháo hoa...", "metadata": {"domain": "Pháp lý"}},
            {"evidence_id": "TTHC_TTPH_01", "content": "Mã thủ tục TTPH-01: Cấp giấy chứng nhận...", "metadata": {"domain": "Hành chính công"}},
        ]

# -----------------------------------------------------------------------------
# 2. KHỞI TẠO VÀ TÍCH HỢP
# -----------------------------------------------------------------------------
# Bật Feature Flag (Trong môi trường thực tế, cấu hình qua file .env hoặc Docker)
os.environ["ENABLE_HYBRID_RETRIEVAL"] = "true"  # Đổi sang "false" để kiểm tra Fallback

# Khởi tạo
dense_engine = MockDenseRetriever()
bm25_engine = MockBM25Retriever()

hybrid_retriever = HybridRetriever(
    dense_retriever=dense_engine,
    bm25_retriever=bm25_engine,
    rrf_k=60,
    feature_flag_var="ENABLE_HYBRID_RETRIEVAL"
)

# -----------------------------------------------------------------------------
# 3. LUỒNG THỰC THI TRONG RAG PIPELINE
# -----------------------------------------------------------------------------
def run_rag_pipeline(user_query: str):
    print(f"User Query: '{user_query}'")
    
    # 1. Retrieval Layer (Truy xuất Evidence)
    evidences = hybrid_retriever.retrieve(query=user_query, top_k=3)
    
    # 2. Xuất dữ liệu cấu trúc chuẩn bàn giao cho Duy Anh (Citation Parser) & An (MCP)
    print("\n--- OUTPUT EVIDENCE BÀN GIAO CHO GENERATION & CITATION PARSER ---")
    for idx, ev in enumerate(evidences, start=1):
        print(f"[{idx}] Evidence ID: {ev['evidence_id']}")
        print(f"    Content    : {ev['content'][:60]}...")
        print(f"    RRF Score  : {ev['score']}")
        print(f"    Domain     : {ev['metadata'].get('domain')}\n")

if __name__ == "__main__":
    run_rag_pipeline("Thủ tục 137/2020/NĐ-CP cấp chứng nhận pháo hoa")