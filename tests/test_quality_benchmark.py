import unittest

from evals.quality_benchmark import evaluate


def gold_row():
    return {
        "sample_id": "synthetic-1", "review_status": "expert_confirmed",
        "reviewers": ["reviewer-a", "reviewer-b"], "adjudicator": "adjudicator",
        "source_provenance": {"references": ["synthetic test fixture"]},
        "expected_domain": "tourism", "relevant_evidence": {"doc-a": 2, "doc-b": 1},
        "expected_answer_points": ["point-a"],
    }


class QualityBenchmarkTests(unittest.TestCase):
    def test_retrieval_metrics_and_latency(self):
        result = evaluate([gold_row(), {**gold_row(), "sample_id": "synthetic-2"}], [{
            "sample_id": "synthetic-1", "selected_domain": "tourism",
            "retrieved_evidence_ids": ["doc-a", "doc-x"],
            "latency_ms": {"retrieval": 10, "total": 20},
        }, {
            "sample_id": "synthetic-2", "selected_domain": "wrong",
            "retrieved_evidence_ids": ["doc-x"],
            "latency_ms": {"retrieval": 30, "total": 40},
        }], k=2)
        self.assertEqual(result["retrieval"]["recall@2"], 0.25)
        self.assertEqual(result["retrieval"]["precision@2"], 0.25)
        self.assertEqual(result["retrieval"]["mrr"], 0.5)
        self.assertEqual(result["retrieval"]["domain_accuracy"], 0.5)
        self.assertEqual(result["by_domain"]["tourism"]["sample_count"], 2)
        self.assertIsNone(result["answer"])
        self.assertEqual(result["latency_ms"]["total"]["p50"], 30)
        self.assertEqual(result["latency_ms"]["total"]["p95"], 40)

    def test_unreviewed_or_unpaired_data_fails_closed(self):
        row = gold_row()
        row["review_status"] = "synthetic"
        with self.assertRaisesRegex(ValueError, "expert_confirmed"):
            evaluate([row], [], k=2)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            evaluate([gold_row()], [], k=2)


if __name__ == "__main__":
    unittest.main()
