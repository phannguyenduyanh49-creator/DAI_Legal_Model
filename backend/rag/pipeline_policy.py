from backend.rag.data_sufficiency import check_enough_data


def build_catalog_fallback_after_retrieval_error(
    question: str,
    catalog_evidences: list,
    evidence_builder,
    allow_external: bool,
) -> dict:
    """Keep catalog evidence usable when vector retrieval fails."""
    candidates = [
        {**evidence.fields, "text": evidence.content}
        for evidence in catalog_evidences
    ]
    enough_data, required_fields, missing_fields = check_enough_data(
        question,
        candidates,
    )
    bundle = evidence_builder.build(
        catalog_evidences=catalog_evidences,
        query=question,
    )
    return {
        "contexts": [],
        "evidence_bundle": bundle,
        "retrieval_status": "error",
        "reranking_status": "not_run",
        "evidence_builder_status": "completed",
        "enough_data": enough_data,
        "required_fields": required_fields,
        "missing_fields": missing_fields,
        "external_fallback": (
            "not_needed" if enough_data
            else "pending" if allow_external
            else "not_allowed"
        ),
    }


def choose_data_source(
    has_local_evidence: bool,
    enough_data: bool | None,
    allow_external: bool,
) -> str:
    """Choose local, external, or permission flow without implicit MCP access."""
    if has_local_evidence and (enough_data is not False or not allow_external):
        return "local"
    if allow_external:
        return "external"
    return "permission"
