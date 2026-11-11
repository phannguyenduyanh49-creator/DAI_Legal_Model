from backend.verifier.evaluation.claim_dataset import candidate_from_catalog, validate


def _candidate():
    return candidate_from_catalog({
        "sample_id": "sample-1", "domain": "administrative_law",
        "claim": "A claim to review.", "evidence_text": "An exact excerpt.",
        "evidence_id": "catalog:record-1", "source_record_id": "record-1",
        "source_field": "processing_time", "source_url": "https://example.org/source",
    })


def test_catalog_candidate_has_no_fabricated_review_or_document_identity():
    row = _candidate()
    assert row["question"] is None
    assert row["source_document_id"] is None
    assert row["label"] is None
    assert row["reviewer"] is None
    assert validate([row])["errors"] == []


def test_validator_rejects_duplicate_pair_and_conflicting_source_identity():
    first = _candidate()
    second = {**first, "sample_id": "sample-2"}
    assert any("duplicate claim" in error for error in validate([first, second])["errors"])
    broken = {**first, "source_metadata": {**first["source_metadata"], "source_record_id": "other"}}
    assert any("contradicts" in error for error in validate([broken])["errors"])


def test_unreviewed_candidate_cannot_carry_a_gold_label():
    row = {**_candidate(), "label": "supported"}
    assert any("unapproved" in error for error in validate([row])["errors"])


def test_reused_evidence_id_cannot_change_domain_or_source():
    first = _candidate()
    second = {
        **first,
        "sample_id": "sample-2",
        "claim": "A different claim.",
        "source_metadata": {**first["source_metadata"], "domain_id": "legal_ai"},
    }
    assert any("conflicting domain/source" in error for error in validate([first, second])["errors"])
