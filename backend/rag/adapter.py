from typing import Any

from backend.evidence.schemas import Evidence


def rag_result_to_evidence(
    item: dict[str, Any],
) -> Evidence:

    procedure_name = item.get(
        "procedure_name",
        ""
    )

    content = item.get(
        "text",
        ""
    )

    evidence_id = (
        item.get("evidence_id")
        or item.get("formality_id")
        or item.get("code")
        or f"rag:{procedure_name}"
    )
    source_type = str(item.get("source_type") or "rag")
    if source_type != "domain_upload":
        evidence_id = (
        item.get("evidence_id")
        or item.get("formality_id")
        or item.get("code")
        or f"rag:{procedure_name}"
        )

    return Evidence(
        evidence_id=(
            evidence_id
            if source_type == "domain_upload"
            else f"rag:{evidence_id}"
        ),
        source_type=source_type,
        source_id=item.get("source_id") or (
            item.get("formality_id")
            or item.get("code")
        ),
        source_url=item.get("source_url"),
        title=item.get("title") or procedure_name,
        content=content,
        location=item.get("location"),
        score=(
            item.get("rerank_score")
            if item.get("rerank_score") is not None
            else item.get("score")
        ),
        validity_from=None,
        validity_to=None,
        metadata={
            "domain_id": item.get("domain_id"),
            "retrieval_score": item.get("score"),
            "rerank_score": item.get("rerank_score"),
            "code": item.get("code"),
            "formality_id": item.get("formality_id"),
            "category_name": item.get("category_name"),
            "target_type": item.get("target_type"),
            "department_promulgate": item.get(
                "department_promulgate"
            ),
            "fees": item.get("fees"),
            "legal_basis": item.get("legal_basis"),
            "executing_agencies": item.get(
                "executing_agencies"
            ),
        },
    )


def rag_results_to_evidence(
    results: list[dict[str, Any]],
) -> list[Evidence]:

    evidences = []

    for item in results:
        evidence = rag_result_to_evidence(item)
        evidences.append(evidence)

    return evidences
