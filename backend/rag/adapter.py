import hashlib
import re
from typing import Any

from backend.evidence.schemas import Evidence


def build_concise_evidence_id(
    item: dict[str, Any],
    raw_id: str | None = None,
    source_type: str = "rag",
) -> tuple[str, str]:
    """Generate a stable, concise, collision-safe Evidence ID while preserving raw identity.

    Returns:
        (concise_evidence_id, original_raw_id)
    """
    procedure_name = str(item.get("title") or item.get("procedure_name") or "").strip()
    content = str(item.get("text") or item.get("full_text") or "").strip()

    if not raw_id and source_type != "domain_upload":
        raw_id = item.get("formality_id") or item.get("code") or procedure_name
    original_raw_id = str(raw_id or "").strip()

    # Uploaded documents already carry an externally assigned evidence
    # identity (for example ``domain:tourism:<document>:<chunk>``). Preserve
    # that identity exactly when supplied; deriving a hash here would sever
    # the link back to the indexed point and make the Evidence provenance
    # contract fail. If it is absent, return an empty ID so the Evidence
    # builder can reject the record instead of fabricating an identity.
    if source_type == "domain_upload":
        return original_raw_id, original_raw_id

    # 1. If raw_id is already a concise, valid prefixed identifier without whitespace, keep it
    if (
        original_raw_id.startswith(("rag:", "domain:", "catalog:", "mcp:"))
        and len(original_raw_id) <= 64
        and not any(c.isspace() for c in original_raw_id)
    ):
        return original_raw_id, original_raw_id

    # 2. Check for formalityId in source_url (DVC procedures)
    source_url = str(item.get("source_url") or "").strip()
    url_formality_match = re.search(r"formalityId=([0-9a-fA-F-]+)", source_url)
    formality_id = str(
        item.get("formality_id")
        or (url_formality_match.group(1) if url_formality_match else "")
    ).strip()

    code = str(item.get("code") or "").strip()
    point_id = str(item.get("point_id") or item.get("record_id") or "").strip()

    # 3. Prefer official code if concise and clean (e.g. 1.002847)
    if code and len(code) <= 32 and not any(c.isspace() for c in code):
        clean_code = re.sub(r"[^\w.-]", "_", code)
        return f"rag:code_{clean_code}", original_raw_id

    # 4. Prefer formality_id if available (UUID or slug)
    if formality_id and len(formality_id) <= 40 and not any(c.isspace() for c in formality_id):
        return f"rag:dvc_{formality_id[:18]}", original_raw_id

    # 5. Check if raw_id is non-empty, short, and has no whitespace
    if (
        original_raw_id
        and len(original_raw_id) <= 32
        and not any(c.isspace() for c in original_raw_id)
        and not original_raw_id.startswith(("rag:", "domain:", "catalog:", "mcp:"))
    ):
        return f"rag:{original_raw_id}", original_raw_id

    # 6. Fallback: Deterministic SHA-256 hash of procedure_name/content to ensure stability and collision safety
    anchor_text = procedure_name or content or original_raw_id or point_id or "unknown"
    digest = hashlib.sha256(anchor_text.encode("utf-8")).hexdigest()[:12]
    return f"rag:doc_{digest}", original_raw_id


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

    evidence_id, original_raw_id = build_concise_evidence_id(item, raw_id, source_type)

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
        "canonical_id": evidence_id,
        "original_procedure_name": procedure_name,
    })
    if original_raw_id:
        metadata["original_evidence_id"] = original_raw_id
    if item.get("retrieval_provenance"):
        metadata["retrieval_provenance"] = item["retrieval_provenance"]

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
