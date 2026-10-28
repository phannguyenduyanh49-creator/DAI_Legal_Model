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
    mcp_tool_allowed: bool = True,
) -> str:
    """Choose local, external, or permission flow without implicit MCP access."""
    if not mcp_tool_allowed:
        return "local"
    if has_local_evidence and (enough_data is not False or not allow_external):
        return "local"
    if allow_external:
        return "external"
    return "permission"


def check_evidence_sufficiency(question: str, evidence_items: list) -> tuple[bool, list[str], list[str]]:
    """Reassess requested facts against the final evidence set.

    For Evidence objects without structured field metadata, their non-empty
    content can satisfy only the generic ``full_text`` requirement. Explicitly
    requested structured facts remain insufficient unless those fields exist.
    """
    candidates = []
    for evidence in evidence_items:
        content = getattr(evidence, "content", None)
        if isinstance(content, str) and content.strip():
            candidates.append({"full_text": content, "text": content})
    return check_enough_data(question, candidates)
