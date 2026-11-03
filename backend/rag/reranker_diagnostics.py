"""Pure helpers for auditable reranker inputs and candidate decisions."""

from __future__ import annotations

from typing import Any


def build_reranker_document_text(item: dict[str, Any]) -> str:
    """Avoid legal-procedure labels and workbook filenames for uploaded records."""
    text = str(item.get("text") or "").strip()
    if item.get("source_type") == "domain_upload":
        record_name = str(item.get("record_name") or "").strip()
        return "\n".join(value for value in (record_name, text) if value)

    name = str(item.get("procedure_name") or "").strip()
    return f"Tên thủ tục: {name}\n{text}" if name else text


def score_and_select_candidates(
    contexts: list[dict[str, Any]],
    scores: list[float],
    *,
    threshold: float,
    max_results: int,
    apply_margin: bool,
    margin_threshold: float,
) -> list[dict[str, Any]]:
    """Return sorted candidate decisions without changing the configured policy."""
    if len(contexts) != len(scores):
        raise ValueError("Reranker scores must match the candidate count")
    scored = [
        {"item": dict(item), "score": float(score)}
        for item, score in zip(contexts, scores)
    ]
    scored.sort(key=lambda candidate: candidate["score"], reverse=True)

    eligible = []
    for candidate in scored:
        candidate["selected"] = False
        candidate["rejection_reason"] = None
        if candidate["score"] < threshold:
            candidate["rejection_reason"] = "score_below_threshold"
        else:
            eligible.append(candidate)

    if eligible and apply_margin:
        top_score = eligible[0]["score"]
        within_margin = [eligible[0]]
        for candidate in eligible[1:]:
            if top_score - candidate["score"] <= margin_threshold:
                within_margin.append(candidate)
            else:
                candidate["rejection_reason"] = "score_gap_exceeds_margin"
        eligible = within_margin

    selected = eligible[:max(0, max_results)]
    selected_ids = {id(candidate) for candidate in selected}
    for candidate in eligible:
        if id(candidate) in selected_ids:
            candidate["selected"] = True
        elif candidate["rejection_reason"] is None:
            candidate["rejection_reason"] = "top_k_limit"

    for rank, candidate in enumerate(scored, start=1):
        candidate["rank"] = rank
    return scored


def format_reranker_diagnostic(
    item: dict[str, Any],
    *,
    rank: int,
    score: float,
    threshold: float,
    selected: bool,
    rejection_reason: str | None,
) -> str:
    fields = item.get("record_fields")
    record_id = item.get("record_id") or (
        fields.get("record_id") if isinstance(fields, dict) else None
    )
    record_identity = (
        f"record_id:{record_id}" if record_id else
        f"sheet_row:{item.get('sheet_name')}:{item.get('row_number')}"
        if item.get("sheet_name") is not None and item.get("row_number") is not None else
        f"chunk_index:{item.get('chunk_index')}"
        if item.get("chunk_index") is not None else "missing"
    )
    return " ".join((
        "[RERANK CANDIDATE]",
        f"candidate_id={str(item.get('point_id') or 'unknown')[:100]}",
        f"record_id={str(record_id or 'absent')[:100]}",
        f"record_identity={record_identity[:160]}",
        f"source_id={str(item.get('source_id') or 'unknown')[:100]}",
        f"candidate_rank={rank}",
        f"reranker_score={score:.6f}",
        "score_normalized=true",
        f"threshold={threshold:.6f}",
        f"decision={'selected' if selected else 'rejected'}",
        f"rejection_reason={rejection_reason or 'none'}",
    ))
