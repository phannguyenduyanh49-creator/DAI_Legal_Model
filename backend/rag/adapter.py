from typing import Any

from backend.evidence.schemas import Evidence


def rag_result_to_evidence(
    item: dict[str, Any],
) -> Evidence:

    procedure_name = item.get("title") or item.get("procedure_name", "")

    content = item.get(
        "text",
        ""
    )

    source_type = str(item.get("source_type") or "rag")
    raw_id = item.get("evidence_id")
    if not raw_id and source_type != "domain_upload":
        raw_id = item.get("formality_id") or item.get("code") or procedure_name
    raw_id = str(raw_id or "").strip()

    evidence_id = (
        raw_id if raw_id.startswith(("rag:", "domain:", "catalog:", "mcp:"))
        else f"rag:{raw_id}" if raw_id
        else ""
    )

    record_fields = item.get("record_fields")
    if not isinstance(record_fields, dict):
        record_fields = item.get("fields")
    fields = record_fields if isinstance(record_fields, dict) else {}
    source_metadata = item.get("metadata")
    metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    metadata.update({
        "retrieval_score": item.get("score"),
        "rerank_score": item.get("rerank_score"),
        "code": item.get("code"),
        "formality_id": item.get("formality_id"),
        "category_name": item.get("category_name"),
        "target_type": item.get("target_type"),
        "department_promulgate": item.get("department_promulgate"),
        "fees": item.get("fees"),
        "legal_basis": item.get("legal_basis"),
        "executing_agencies": item.get("executing_agencies"),
    })
    for key in (
        "domain_id", "document_id", "record_id", "chunk_index", "point_id",
        "sheet_name", "row_number", "record_name",
    ):
        if item.get(key) is not None:
            metadata[key] = item[key]
    if metadata.get("record_id") is None and fields.get("record_id") is not None:
        metadata["record_id"] = fields["record_id"]
    if fields:
        metadata["record_fields"] = fields

    return Evidence(
        evidence_id=evidence_id,
        source_type=source_type,
        # For uploaded records, source_id is an independent source identity.
        # Do not silently infer it from document_id: the verifier must be able
        # to detect a missing source_id instead of accepting an alias.
        source_id=(item.get("source_id") if source_type == "domain_upload" else (
            item.get("source_id")
            or item.get("document_id")
            or item.get("formality_id")
            or item.get("code")
        )),
        source_url=item.get("source_url"),
        title=procedure_name,
        content=content,
        location=item.get("location"),
        score=(
            item.get("rerank_score")
            if item.get("rerank_score") is not None
            else item.get("score")
        ),
        validity_from=item.get("validity_from"),
        validity_to=item.get("validity_to"),
        fields=fields,
        completeness=(
            item.get("completeness")
            if isinstance(item.get("completeness"), dict)
            else {}
        ),
        metadata=metadata,
    )


def rag_results_to_evidence(
    results: list[dict[str, Any]],
) -> list[Evidence]:

    evidences = []

    for item in results:
        evidence = rag_result_to_evidence(item)
        evidences.append(evidence)

    return evidences
