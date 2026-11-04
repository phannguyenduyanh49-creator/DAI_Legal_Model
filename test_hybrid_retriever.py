import os
import unittest
from unittest.mock import patch

from hybrid_retriever import HybridRetriever
from benchmark_eval import calculate_metrics_for_query
from pipeline_integration import hybrid_result_to_rag_context
from prototype import BM25Engine


class StaticRetriever:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        if self.error:
            raise self.error
        return self.results[:top_k]


class HybridRetrieverTests(unittest.TestCase):
    def make_retriever(self, dense, bm25, flag="TEST_HYBRID_RETRIEVAL"):
        return HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            rrf_k=60,
            feature_flag_var=flag,
        )

    def test_fusion_preserves_evidence_identity_and_merges_payload_fields(self):
        dense = StaticRetriever(
            [
                {
                    "evidence_id": " law:42 ",
                    "content": "",
                    "metadata": {"domain_id": "legal_ai"},
                    "dense_score": 0.81,
                }
            ]
        )
        bm25 = StaticRetriever(
            [
                {
                    "doc_id": "law:42",
                    "text": "Điều khoản nội dung nguồn",
                    "metadata": {"source_type": "rag"},
                    "legal_basis": "Luật mẫu",
                }
            ]
        )
        retriever = self.make_retriever(dense, bm25)

        with patch.dict(os.environ, {"TEST_HYBRID_RETRIEVAL": " true "}):
            results = retriever.retrieve("câu hỏi", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["evidence_id"], "law:42")
        self.assertEqual(results[0]["content"], "Điều khoản nội dung nguồn")
        self.assertEqual(results[0]["metadata"], {
            "domain_id": "legal_ai",
            "source_type": "rag",
        })
        self.assertEqual(results[0]["legal_basis"], "Luật mẫu")
        self.assertEqual(results[0]["dense_score"], 0.81)
        self.assertAlmostEqual(results[0]["score"], 2 / 61, places=6)

    def test_duplicate_ids_in_one_channel_do_not_inflate_rrf(self):
        dense = StaticRetriever([
            {"evidence_id": "a", "content": "first"},
            {"evidence_id": "a", "content": "duplicate"},
            {"evidence_id": "b", "content": "second"},
        ])
        bm25 = StaticRetriever([])
        retriever = self.make_retriever(dense, bm25)

        with patch.dict(os.environ, {"TEST_HYBRID_RETRIEVAL": "1"}):
            results = retriever.retrieve("query", top_k=3)

        self.assertEqual([item["evidence_id"] for item in results], ["a", "b"])
        self.assertAlmostEqual(results[0]["score"], 1 / 61, places=6)

    def test_missing_ids_are_dropped_and_alias_ids_are_normalized(self):
        dense = StaticRetriever([
            {"evidence_id": "  ", "content": "missing"},
            {"id": 0, "content": "zero id"},
        ])
        bm25 = StaticRetriever([])
        retriever = self.make_retriever(dense, bm25)

        with patch.dict(os.environ, {"TEST_HYBRID_RETRIEVAL": "yes"}):
            results = retriever.retrieve("query", top_k=3)

        self.assertEqual([item["evidence_id"] for item in results], ["0"])

    def test_disabled_feature_flag_uses_dense_only(self):
        dense = StaticRetriever([{"evidence_id": "dense", "content": "dense"}])
        bm25 = StaticRetriever([{"evidence_id": "lexical", "content": "lexical"}])
        retriever = self.make_retriever(dense, bm25)

        with patch.dict(os.environ, {"TEST_HYBRID_RETRIEVAL": "false"}):
            results = retriever.retrieve("query", top_k=2)

        self.assertEqual([item["evidence_id"] for item in results], ["dense"])
        self.assertEqual(len(dense.calls), 1)
        self.assertEqual(bm25.calls, [])

    def test_one_channel_failure_degrades_to_the_other_channel(self):
        dense = StaticRetriever(error=OSError("dense unavailable"))
        bm25 = StaticRetriever([{"evidence_id": "lexical", "content": "hit"}])
        retriever = self.make_retriever(dense, bm25)

        with patch.dict(os.environ, {"TEST_HYBRID_RETRIEVAL": "true"}):
            results = retriever.retrieve("query", top_k=2)

        self.assertEqual([item["evidence_id"] for item in results], ["lexical"])

    def test_both_channel_failures_are_reported(self):
        retriever = self.make_retriever(
            StaticRetriever(error=OSError("dense unavailable")),
            StaticRetriever(error=OSError("BM25 unavailable")),
        )

        with patch.dict(os.environ, {"TEST_HYBRID_RETRIEVAL": "true"}):
            with self.assertRaisesRegex(RuntimeError, "Both hybrid retrieval"):
                retriever.retrieve("query", top_k=2)

    def test_invalid_configuration_and_limits_fail_explicitly(self):
        with self.assertRaises(ValueError):
            HybridRetriever(StaticRetriever(), StaticRetriever(), rrf_k=0)

        retriever = self.make_retriever(StaticRetriever(), StaticRetriever())
        with self.assertRaises(ValueError):
            retriever.retrieve("query", top_k=-1)
        self.assertEqual(retriever.retrieve("query", top_k=0), [])

    def test_pipeline_mapping_preserves_evidence_id_and_current_context_fields(self):
        context = hybrid_result_to_rag_context({
            "evidence_id": "domain:legal_ai:document-1:1",
            "content": "Nguồn chứng cứ",
            "score": 0.03,
            "metadata": {
                "source_type": "domain_upload",
                "domain_id": "legal_ai",
                "title": "Luật mẫu",
            },
        })

        self.assertEqual(
            context["evidence_id"],
            "domain:legal_ai:document-1:1",
        )
        self.assertEqual(context["text"], "Nguồn chứng cứ")
        self.assertEqual(context["procedure_name"], "Luật mẫu")
        self.assertEqual(context["rrf_score"], 0.03)
        self.assertEqual(context["source_type"], "domain_upload")
        self.assertEqual(context["domain_id"], "legal_ai")

    def test_metrics_do_not_count_duplicate_evidence_ids_as_extra_hits(self):
        metrics = calculate_metrics_for_query(
            ["unrelated", "evidence-1", "evidence-1"],
            {"evidence-1"},
            k=2,
        )

        self.assertEqual(metrics["Recall@2"], 1.0)
        self.assertEqual(metrics["Precision@2"], 0.5)
        self.assertEqual(metrics["MRR"], 0.5)

    def test_bm25_retrieves_exact_evidence_id_in_a_large_diverse_corpus(self):
        topics = (
            "đăng ký đất đai",
            "giấy phép xây dựng",
            "hợp đồng lao động",
            "thủ tục hộ kinh doanh",
            "phòng cháy chữa cháy",
            "an toàn thực phẩm",
            "bảo hiểm xã hội",
            "đăng ký doanh nghiệp",
        )
        corpus = [
            {
                "doc_id": f"evidence-{index:05}",
                "title": f"{topics[index % len(topics)]} mã thủ tục {index:05}",
                "content": f"Quy định và hồ sơ liên quan mã thủ tục {index:05}.",
            }
            for index in range(5000)
        ]
        engine = BM25Engine(corpus)

        results = engine.search("mã thủ tục 04987", top_k=5)

        self.assertEqual(results[0]["doc_id"], "evidence-04987")
        self.assertEqual(len({result["doc_id"] for result in results}), len(results))


if __name__ == "__main__":
    unittest.main()
