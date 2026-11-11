from types import SimpleNamespace

import pytest

from backend.search.repair_education_domain import digest, repaired_payload


def education_point(**overrides):
    payload = {
        "sheet_name": "Giao_duc_100",
        "domain_id": "healthcare",
        "source_type": "domain_upload",
        "source_id": "workbook-id",
        "document_id": "workbook-id",
        "chunk_index": 42,
        "evidence_id": "domain:healthcare:workbook-id:42",
        "record_id": "EDU-42",
        "text": "Original Education record",
    }
    payload.update(overrides)
    return SimpleNamespace(payload=payload)


def test_repair_changes_only_domain_identity_and_preserves_original():
    point = education_point()
    repaired = repaired_payload(point)

    assert repaired["domain_id"] == "education"
    assert repaired["evidence_id"] == "domain:education:workbook-id:42"
    assert repaired["record_id"] == point.payload["record_id"]
    assert repaired["text"] == point.payload["text"]
    assert point.payload["domain_id"] == "healthcare"


@pytest.mark.parametrize("overrides", [
    {"sheet_name": "Y_te_100"},
    {"source_id": "another-document"},
    {"evidence_id": "domain:healthcare:other:42"},
    {"chunk_index": 0},
    {"text": ""},
])
def test_repair_rejects_unreviewed_or_ambiguous_points(overrides):
    with pytest.raises(ValueError):
        repaired_payload(education_point(**overrides))


def test_backup_digest_is_order_independent_for_mapping_keys():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
