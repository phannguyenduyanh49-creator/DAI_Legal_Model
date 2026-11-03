from backend.evidence.builder import EvidenceBuilder
from backend.evidence.schemas import Evidence


def test_builder_deduplicates_repeated_copy_without_changing_provenance():
    first = Evidence(
        evidence_id="domain:shopping:doc-1:1",
        source_type="domain_upload",
        source_id="doc-1",
        title="products.xlsx",
        location="Sheet 1 · row 2",
        content="product: kettle",
        score=0.7,
        metadata={"domain_id": "shopping", "document_id": "doc-1"},
    )
    repeated = Evidence(
        evidence_id="domain:shopping:doc-1:1",
        source_type="domain_upload",
        source_id="doc-1",
        title="products.xlsx",
        location="Sheet 1 · row 2",
        content="product: kettle",
        score=0.9,
        metadata={"domain_id": "shopping", "document_id": "doc-1"},
    )

    bundle = EvidenceBuilder().build(rag_evidences=[first, repeated])

    assert bundle.total == 1
    assert bundle.evidences[0].evidence_id == first.evidence_id
    assert bundle.evidences[0].score == 0.9
    assert bundle.metadata["integrity_issues"] == []


def test_builder_excludes_same_id_with_conflicting_source_or_content():
    first = Evidence(
        evidence_id="domain:shopping:doc-1:1",
        source_type="domain_upload",
        source_id="doc-1",
        content="product: kettle",
        score=0.9,
        metadata={"domain_id": "shopping", "document_id": "doc-1"},
    )
    conflict = Evidence(
        evidence_id="domain:shopping:doc-1:1",
        source_type="domain_upload",
        source_id="doc-2",
        content="product: toaster",
        score=0.8,
        metadata={"domain_id": "shopping", "document_id": "doc-2"},
    )

    bundle = EvidenceBuilder().build(rag_evidences=[first, conflict])

    assert bundle.total == 0
    assert bundle.metadata["integrity_excluded_count"] == 1
    assert bundle.metadata["integrity_issues"] == [
        "EVIDENCE_ID_PROVENANCE_CONFLICT:domain:shopping:doc-1:1"
    ]


def test_builder_excludes_evidence_without_an_id_and_reports_integrity_issue():
    bundle = EvidenceBuilder().build(rag_evidences=[
        Evidence(evidence_id="", source_type="rag", content="Source text", score=0.9)
    ])

    assert bundle.total == 0
    assert bundle.metadata["integrity_issues"] == ["EVIDENCE_ID_MISSING"]


def test_builder_rejects_uploaded_evidence_with_missing_source_identity():
    bundle = EvidenceBuilder().build(rag_evidences=[Evidence(
        evidence_id="domain:shopping:doc-1:1",
        source_type="domain_upload",
        content="product: kettle",
        score=0.9,
        metadata={"domain_id": "shopping", "document_id": "doc-1"},
    )])

    assert bundle.total == 0
    assert bundle.metadata["integrity_issues"] == [
        "EVIDENCE_SOURCE_IDENTITY_MISSING:source_id"
    ]


def test_invalid_duplicate_identity_cannot_be_replaced_by_a_valid_copy():
    invalid = Evidence(
        evidence_id="domain:shopping:doc-1:1", source_type="domain_upload",
        content="same content", score=0.9,
        metadata={"domain_id": "shopping", "document_id": "doc-1"},
    )
    valid = Evidence(
        evidence_id="domain:shopping:doc-1:1", source_type="domain_upload",
        source_id="doc-1", content="same content", score=0.8,
        metadata={"domain_id": "shopping", "document_id": "doc-1"},
    )

    bundle = EvidenceBuilder().build(rag_evidences=[invalid, valid])

    assert bundle.total == 0
    assert bundle.metadata["integrity_issues"] == [
        "EVIDENCE_SOURCE_IDENTITY_MISSING:source_id"
    ]


def test_builder_rejects_uploaded_evidence_with_conflicting_record_identity():
    first = Evidence(
        evidence_id="domain:shopping:doc-1:1", source_type="domain_upload",
        source_id="doc-1", content="product: kettle", score=0.9,
        metadata={"domain_id": "shopping", "document_id": "doc-1", "record_id": "SKU-1"},
    )
    conflicting = Evidence(
        evidence_id="domain:shopping:doc-1:1", source_type="domain_upload",
        source_id="doc-1", content="product: kettle", score=0.8,
        metadata={"domain_id": "shopping", "document_id": "doc-1", "record_id": "SKU-2"},
    )

    bundle = EvidenceBuilder().build(rag_evidences=[first, conflicting])

    assert bundle.total == 0
    assert bundle.metadata["integrity_issues"] == [
        "EVIDENCE_ID_PROVENANCE_CONFLICT:domain:shopping:doc-1:1"
    ]
