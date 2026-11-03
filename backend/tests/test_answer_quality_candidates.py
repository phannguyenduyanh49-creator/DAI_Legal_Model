import json
from pathlib import Path


def test_answer_quality_candidate_cases_are_unique_provenanced_and_unreviewed():
    path = Path(__file__).resolve().parents[2] / "evals" / "answer_quality_candidates.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    cases = dataset["cases"]

    assert dataset["review_status"] == "needs_human_review"
    assert dataset["requires_human_semantic_check"] is True
    assert len(cases) >= 3
    assert len({case["case_id"] for case in cases}) == len(cases)
    for case in cases:
        assert case["expected_domain"]
        assert case["collection"]
        assert case["expected_behavior"]
        assert case["evidence_candidates"]
        assert all(
            candidate["point_id"]
            and candidate["review_status"] == "unreviewed"
            for candidate in case["evidence_candidates"]
        )


def test_misplaced_education_candidate_is_explicitly_forbidden_as_evidence():
    path = Path(__file__).resolve().parents[2] / "evals" / "answer_quality_candidates.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    case = next(item for item in dataset["cases"] if item["case_id"] == "education-cross-domain-isolation")

    candidate = case["evidence_candidates"][0]
    assert case["expected_domain"] == "education"
    assert candidate["usable_as_expected_domain_evidence"] is False
    assert "healthcare" in candidate["observed_payload_facts"][-1]
