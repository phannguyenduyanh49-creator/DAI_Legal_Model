# -*- coding: utf-8 -*-

import os
import logging
import math
from collections.abc import Mapping
from typing import List, Dict, Any, Optional

logger = logging.getLogger("HybridRetriever")


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Any,
        rrf_k: int = 60,
        min_rrf_score: float = 0.0,
        feature_flag_var: str = "ENABLE_HYBRID_RETRIEVAL"
    ):
        """
        Khởi tạo HybridRetriever.
        
        :param dense_retriever: Instance truy xuất Qdrant / BGE-M3
        :param bm25_retriever: Instance truy xuất BM25
        :param rrf_k: Hằng số làm mượt cho RRF (Mặc định 60)
        :param min_rrf_score: Ngưỡng điểm RRF tối thiểu để lọc bớt context rác
        :param feature_flag_var: Tên biến môi trường bật/tắt Hybrid Search
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.min_rrf_score = min_rrf_score
        self.feature_flag_var = feature_flag_var

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if not math.isfinite(min_rrf_score) or min_rrf_score < 0:
            raise ValueError("min_rrf_score must be finite and non-negative")

    @property
    def is_hybrid_enabled(self) -> bool:
        """Kiểm tra trạng thái Feature Flag từ biến môi trường."""
        flag_val = os.getenv(self.feature_flag_var, "false").strip().lower()
        return flag_val in ["true", "1", "yes"]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hàm truy xuất chính được gọi từ Pipeline RAG.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        # BƯỚC 1: FALLBACK SAFEGUARD
        # Nếu Feature Flag TẮT -> Trả về luồng Dense Search cũ 100%
        if not self.is_hybrid_enabled:
            logger.info("[HYBRID_FLAG=FALSE] Chạy luồng Local RAG Dense Search mặc định.")
            raw_dense = self.dense_retriever.search(query, top_k=top_k)
            return self._format_and_audit_output(raw_dense)

        # BƯỚC 2: TRUY XUẤT SONG SONG (Lấy gấp 3 lần top_k để RRF chấm điểm tối ưu)
        fetch_k = top_k * 3
        failures = []
        try:
            dense_results = self.dense_retriever.search(query, top_k=fetch_k)
        except Exception as e:
            logger.exception("Dense retrieval failed; continuing with BM25")
            dense_results = []
            failures.append(e)

        try:
            bm25_results = self.bm25_retriever.search(query, top_k=fetch_k)
        except Exception as e:
            logger.exception("BM25 retrieval failed; continuing with dense results")
            bm25_results = []
            failures.append(e)

        if len(failures) == 2:
            raise RuntimeError("Both hybrid retrieval channels failed") from failures[-1]

        # BƯỚC 3: MERGE KẾT QUẢ BẰNG RECIPROCAL RANK FUSION (RRF)
        fused_results = self._reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            top_k=top_k
        )

        # BƯỚC 4: BẢO TOÀN VÀ KIỂM TRA EVIDENCE ID
        audited_results = self._format_and_audit_output(fused_results)
        logger.info(f"[HYBRID_FLAG=TRUE] Đã truy xuất thành công {len(audited_results)} Evidence qua RRF.")
        
        return audited_results

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: List[List[Dict[str, Any]]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Thuật toán hợp nhất thứ hạng RRF."""
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        for ranked_list in ranked_lists:
            seen_in_list = set()
            for rank, doc in enumerate(ranked_list, start=1):
                if not isinstance(doc, Mapping):
                    raise TypeError("Retriever results must be mappings")

                ev_id = self._get_evidence_id(doc)
                if ev_id is None or ev_id in seen_in_list:
                    continue

                seen_in_list.add(ev_id)

                if ev_id not in rrf_scores:
                    rrf_scores[ev_id] = 0.0
                    doc_store[ev_id] = dict(doc)
                    doc_store[ev_id]["evidence_id"] = ev_id
                    metadata = doc.get("metadata", {})
                    if not isinstance(metadata, Mapping):
                        raise TypeError("Document metadata must be a mapping")
                    doc_store[ev_id]["metadata"] = dict(metadata)
                else:
                    self._merge_document(doc_store[ev_id], doc)

                # Công thức RRF: 1 / (k + r)
                rrf_scores[ev_id] += 1.0 / (self.rrf_k + rank)

        # Sắp xếp giảm dần theo RRF Score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for ev_id, score in sorted_docs:
            if score < self.min_rrf_score:
                continue
            item = doc_store[ev_id].copy()
            item["score"] = round(score, 6)
            fused.append(item)
            if len(fused) >= top_k:
                break

        return fused

    @staticmethod
    def _get_evidence_id(doc: Mapping[str, Any]) -> Optional[str]:
        for key in ("evidence_id", "doc_id", "id"):
            value = doc.get(key)
            if value is not None:
                normalized = str(value).strip()
                if normalized:
                    return normalized
        return None

    @staticmethod
    def _merge_document(
        existing: Dict[str, Any],
        incoming: Mapping[str, Any],
    ) -> None:
        for key, value in incoming.items():
            if key == "evidence_id":
                continue
            if key == "metadata":
                if not isinstance(value, Mapping):
                    raise TypeError("Document metadata must be a mapping")
                metadata = existing.setdefault("metadata", {})
                if not isinstance(metadata, dict):
                    raise TypeError("Document metadata must be a mapping")
                for metadata_key, metadata_value in value.items():
                    if metadata_key not in metadata or metadata[metadata_key] in (
                        None,
                        "",
                    ):
                        metadata[metadata_key] = metadata_value
                continue
            if key not in existing or existing[key] in (None, "", {}):
                existing[key] = value

    def _format_and_audit_output(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Đảm bảo định dạng đầu ra chuẩn 100% cho Citation Parser của Duy Anh
        và Local RAG / MCP Routing của An.
        """
        formatted = []
        for idx, doc in enumerate(raw_results):
            ev_id = self._get_evidence_id(doc)

            # Cảnh báo nếu phát hiện record thiếu Evidence ID
            if not ev_id:
                logger.warning(f"Bỏ qua record vị trí {idx} do thiếu Evidence ID!")
                continue

            clean_doc = dict(doc)
            clean_doc["evidence_id"] = ev_id
            content = doc.get("content") or doc.get("text", "")
            clean_doc["content"] = content.strip() if isinstance(content, str) else ""
            metadata = doc.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise TypeError("Document metadata must be a mapping")
            clean_doc["metadata"] = dict(metadata)
            clean_doc["score"] = doc.get("score", 0.0)
            formatted.append(clean_doc)

        return formatted