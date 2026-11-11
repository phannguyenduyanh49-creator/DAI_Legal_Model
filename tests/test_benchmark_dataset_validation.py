import json
from pathlib import Path

from benchmark.validate_dataset import validate_dataset
from benchmark.prepare_100_candidates import build


ROOT = Path(__file__).resolve().parents[1]


def test_current_30_case_dataset_is_valid_operational_input_but_not_100_case_target():
    dataset = json.loads((ROOT / "benchmark/dataset/benchmark_questions.json").read_text(encoding="utf-8"))
    report = validate_dataset(dataset)
    assert report["case_count"] == 30
    assert report["errors"] == []
    target_report = validate_dataset(dataset, target_100=True)
    assert any("100 cases" in error for error in target_report["errors"])


def test_duplicate_ids_queries_and_unreviewed_reviewer_claims_are_detected():
    case = {"case_id": "X1", "query": "Một câu hỏi", "expected_domain": "legal_ai", "group": "legal_rag"}
    duplicate = {**case, "query": "  MỘT   câu hỏi ", "review_status": "reviewed"}
    errors = validate_dataset({"cases": [case, duplicate]})["errors"]
    assert any("duplicate case_id" in error for error in errors)
    assert any("duplicate query" in error for error in errors)
    assert any("without reviewer" in error for error in errors)


def test_100_candidate_set_preserves_original_cases_without_creating_gold_labels():
    original = json.loads((ROOT / "benchmark/dataset/benchmark_questions.json").read_text(encoding="utf-8"))
    expanded = build()
    report = validate_dataset(expanded, target_100=True)
    assert report["errors"] == []
    assert report["case_count"] == 100
    assert report["reviewed_case_count"] == 0
    for old, new in zip(original["cases"], expanded["cases"]):
        assert old["case_id"] == new["case_id"]
        assert old["query"] == new["query"]
    assert all(case["review_status"] == "candidate_unreviewed" for case in expanded["cases"])
    assert all("candidate_evidence_ids" not in case for case in expanded["cases"])
    assert sum(case["allow_external"] is True for case in expanded["cases"]) == 5
    assert all(case.get("relevant_document_ids") == [] for case in expanded["cases"][30:])
