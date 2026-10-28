"""Shared citation instructions for initial generation and repair attempts."""

CITATION_RULES = (
    "Output only a concise bullet list, with exactly one short sentence per bullet; "
    "do not add an uncited introduction, conclusion, or heading. Answer using only "
    "the supplied Evidence. Include only legal factual claims supported by Evidence. "
    "End every legal factual sentence "
    "with exactly [Evidence ID: <exact allowed ID>]. Cite the Evidence that supports "
    "that sentence; do not reuse one sentence's citation for another claim. Do not "
    "copy a citation to an unsupported sentence. Do not put "
    "citations in headings. Do not combine independent legal claims in one sentence. "
    "Never invent or alter an Evidence ID, legal source, fact, amount, condition, or "
    "deadline. If Evidence does not support an answer, state that the available "
    "Evidence is insufficient and identify the missing information without stating "
    "unsupported legal facts."
)


def format_allowed_evidence_ids(evidence_ids) -> str:
    """Format an already prompt-filtered ID sequence without widening it."""
    ids = tuple(str(evidence_id) for evidence_id in evidence_ids)
    return ", ".join(ids) if ids else "none"
