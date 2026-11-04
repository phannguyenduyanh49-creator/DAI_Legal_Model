"""Shared citation instructions for initial generation and repair attempts."""

CITATION_RULES = (
    "Start with a concise direct answer when Evidence supports one, then address "
    "the requested parts in the order asked. Avoid repeating facts. Use short sections or "
    "bullet points when they make the answer easier to follow; there is no fixed "
    "sentence or bullet limit. Answer using only the supplied Evidence. Include "
    "only factual claims supported by Evidence. Keep the answer concise and omit "
    "unsupported elaboration. End every factual sentence on the same line with "
    "exactly [Evidence ID: <exact allowed ID>]; never put a citation marker on "
    "its own line or separate it from the claim with a blank line. Cite the Evidence that supports "
    "that sentence; do not reuse one sentence's citation for another claim. Do not "
    "copy a citation to an unsupported sentence. Do not put "
    "citations in headings. Do not combine independent legal claims in one sentence. "
    "Never invent or alter an Evidence ID, legal source, fact, amount, condition, or "
    "deadline. If Evidence supports only part of the request, omit unsupported facts "
    "and put only missing requested detail names in the neutral heading "
    "### Thông tin nguồn chưa nêu, with no prose beneath it. If no requested fact "
    "is supported, do not make a factual claim; use only that neutral heading."
)


def format_allowed_evidence_ids(evidence_ids) -> str:
    """Format an already prompt-filtered ID sequence without widening it."""
    ids = tuple(str(evidence_id) for evidence_id in evidence_ids)
    return ", ".join(ids) if ids else "none"
