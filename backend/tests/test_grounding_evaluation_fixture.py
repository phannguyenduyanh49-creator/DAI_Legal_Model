import json
from pathlib import Path


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "verifier"
    / "evaluation"
    / "synthetic_grounding_cases.jsonl"
)


def test_synthetic_grounding_fixture_has_required_scenarios_and_fields():
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    required = {
        "case_id",
        "claim",
        "evidence_id",
        "evidence_text",
        "label",
        "domain",
        "annotator_notes",
    }

    assert len(rows) == 6
    assert all(required.issubset(row) for row in rows)
    assert {row["label"] for row in rows} == {
        "supported",
        "unsupported",
        "contradicted",
        "partially_supported",
        "insufficient_evidence",
        "citation_invalid",
    }
    assert all("synthetic" in row["annotator_notes"].lower() for row in rows)
