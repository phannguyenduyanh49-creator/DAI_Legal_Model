import unittest
from pathlib import Path
import sys

from PROTOTYPE import (
    BM25Engine,
    evaluate_run,
    legal_tokenizer,
    load_dataset,
    reciprocal_rank_fusion,
    run_benchmark,
    search_exact_metadata,
)


class HybridRetrievalPrototypeTests(unittest.TestCase):
    def test_tokenizer_preserves_identifiers_and_accentless_variants(self):
        tokens = legal_tokenizer("137/2020/NĐ-CP Điều 15 TTPH-01")

        self.assertIn("137/2020/nd-cp", tokens)
        self.assertIn("nd", tokens)
        self.assertIn("15", tokens)
        self.assertIn("ttph-01", tokens)

    def test_bm25_returns_only_matching_documents_and_respects_top_k(self):
        corpus = [
            {"doc_id": "a", "title": "Huấn luyện pháo hoa", "content": "An toàn"},
            {"doc_id": "b", "title": "Đăng ký đất đai", "content": "Hồ sơ"},
        ]
        engine = BM25Engine(corpus)

        self.assertEqual(engine.search("huan luyen phao", top_k=1)[0]["doc_id"], "a")
        self.assertEqual(engine.search("không có từ trùng"), [])
        self.assertEqual(engine.search("huấn luyện", top_k=0), [])
        with self.assertRaises(ValueError):
            engine.search("huấn luyện", top_k=-1)

    def test_exact_metadata_matches_accentless_document_number_and_code(self):
        corpus = [
            {
                "doc_id": "a",
                "doc_number": "137/2020/NĐ-CP",
                "procedure_code": "TTPH-01",
                "article": "Điều 15",
            },
            {
                "doc_id": "b",
                "doc_number": "56/2023/NĐ-CP",
                "procedure_code": "TTPH-01-SD",
                "article": "Điều 1",
            },
        ]

        self.assertEqual(
            [hit["doc_id"] for hit in search_exact_metadata("137/2020/ND-CP", corpus)],
            ["a"],
        )
        self.assertEqual(
            [hit["doc_id"] for hit in search_exact_metadata("TTPH-01", corpus)],
            ["a"],
        )
        self.assertEqual(
            [hit["doc_id"] for hit in search_exact_metadata("TTPH-01-SD", corpus)],
            ["b"],
        )
        self.assertEqual(
            [hit["doc_id"] for hit in search_exact_metadata("Điều 1", corpus)],
            ["b"],
        )

    def test_rrf_rewards_a_document_ranked_by_multiple_retrievers(self):
        fused = reciprocal_rank_fusion(
            [
                [{"doc_id": "a"}, {"doc_id": "b"}],
                [{"doc_id": "b"}, {"doc_id": "c"}],
            ]
        )

        self.assertEqual(fused[0]["doc_id"], "b")
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 62 + 1 / 61)

    def test_metric_definitions_for_a_known_ranking(self):
        cases = [{"query_id": "q", "relevant_ids": ["b"]}]
        metrics = evaluate_run(cases, {"q": ["a", "b"]}, top_k=2)

        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertAlmostEqual(metrics["ndcg"], 1 / 1.584962500721156)

    def test_sample_benchmark_has_40_queries_and_reports_all_methods(self):
        dataset = load_dataset(Path(__file__).with_name("HYBRID_RETRIEVAL_EVAL.json"))
        result = run_benchmark(dataset, top_k=5)

        self.assertEqual(result["query_count"], 40)
        self.assertEqual(result["corpus_size"], 9)
        self.assertEqual(
            set(result["metrics"]),
            {"bm25", "exact_metadata", "bm25_plus_exact_rrf"},
        )
        self.assertTrue(
            all(
                0.0 <= value <= 1.0
                for scores in result["metrics"].values()
                for value in scores.values()
            )
        )
        self.assertAlmostEqual(
            result["metrics"]["bm25_plus_exact_rrf"]["recall"], 1.0
        )
        self.assertGreater(
            result["metrics"]["bm25_plus_exact_rrf"]["mrr"],
            result["metrics"]["bm25"]["mrr"],
        )


class ReportingTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.passed = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed += 1
        self.stream.writeln(f"[PASS] {test.id().rsplit('.', 1)[-1]}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.writeln(f"[FAIL] {test.id().rsplit('.', 1)[-1]}")

    def addError(self, test, err):
        super().addError(test, err)
        self.stream.writeln(f"[ERROR] {test.id().rsplit('.', 1)[-1]}")

    def stopTestRun(self):
        super().stopTestRun()
        failed = len(self.failures) + len(self.errors)
        self.stream.writeln(
            f"\nRESULT: {self.passed}/{self.testsRun} tests passed"
            + (f", {failed} failed" if failed else "")
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        HybridRetrievalPrototypeTests
    )
    runner = unittest.TextTestRunner(
        stream=sys.stdout,
        verbosity=0,
        resultclass=ReportingTestResult,
    )
    result = runner.run(suite)
    raise SystemExit(not result.wasSuccessful())
