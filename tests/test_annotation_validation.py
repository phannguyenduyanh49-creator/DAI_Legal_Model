import pytest

from evals.validate_annotations import validate_judgment, validate_review_records


def judgment(key="catalog:legal_ai:doc-1", state="relevant", label=2):
    return {
        "document_key": key,
        "domain_id": "legal_ai",
        "document_id": "doc-1",
        "record_id": "record-1",
        "source_id": "source-1",
        "content_sha256": "a" * 64,
        "state": state,
        "label": label,
        "note": "The inspected source directly covers the query.",
        "source_checked_at": "2026-11-03T09:00:00Z",
    }


def review(annotator_id):
    return {
        "query_id": "CAND-001",
        "annotator_id": annotator_id,
        "annotated_at": "2026-11-03T09:00:00Z",
        "review_status": "submitted",
        "query_status": "answerable",
        "expected_domain": "legal_ai",
        "document_judgments": [judgment()],
        "notes": "Reviewed independently against the source.",
    }


def adjudication():
    return {
        "query_id": "CAND-001",
        "adjudicator_id": "judge-1",
        "adjudicated_at": "2026-11-03T10:00:00Z",
        "adjudication_status": "approved",
        "query_status": "answerable",
        "expected_domain": "legal_ai",
        "expected_record_type": "direct_catalog",
        "document_judgments": [judgment()],
        "notes": "Both reviews agree after source inspection.",
    }


def test_annotation_gate_requires_two_reviewers_and_adjudication():
    candidates = [{"query_id": "CAND-001", "query": "Procedure question"}]
    grouped, decisions = validate_review_records(
        candidates,
        [review("reviewer-a"), review("reviewer-b")],
        [adjudication()],
    )
    assert {row["annotator_id"] for row in grouped["CAND-001"]} == {"reviewer-a", "reviewer-b"}
    assert decisions["CAND-001"]["adjudication_status"] == "approved"


def test_insufficient_information_is_not_encoded_as_irrelevant_zero():
    validate_judgment(judgment(state="insufficient_information", label=None))
    with pytest.raises(ValueError, match="null relevance label"):
        validate_judgment(judgment(state="insufficient_information", label=0))


def test_annotation_gate_rejects_more_than_two_reviewers():
    with pytest.raises(ValueError, match="exactly two independent annotator reviews"):
        validate_review_records(
            [{"query_id": "CAND-001", "query": "Procedure question"}],
            [review("reviewer-a"), review("reviewer-b"), review("reviewer-c")],
            [adjudication()],
        )


def test_automatic_suggestion_cannot_be_submitted_as_a_human_review():
    first = review("reviewer-a")
    first["auto_candidate_judgments"] = [{"suggested_relevance": 2}]
    with pytest.raises(ValueError, match="automatic candidate suggestions"):
        validate_review_records(
            [{"query_id": "CAND-001", "query": "Procedure question"}],
            [first, review("reviewer-b")],
            [adjudication()],
        )


def test_adjudicator_must_be_independent_and_review_pools_must_match():
    decision = adjudication()
    decision["adjudicator_id"] = "reviewer-a"
    with pytest.raises(ValueError, match="adjudicator must be independent"):
        validate_review_records(
            [{"query_id": "CAND-001", "query": "Procedure question"}],
            [review("reviewer-a"), review("reviewer-b")],
            [decision],
        )

