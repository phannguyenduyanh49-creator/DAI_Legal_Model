import re

from backend.answer.schemas import Claim, StructuredAnswer


_CITATION_RE = re.compile(r"\[Evidence ID:\s*([^\]]+)\]", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_BOLD_HEADING_RE = re.compile(r"\*\*[^*]+\*\*")


def structure_answer_with_explicit_evidence(answer: str) -> StructuredAnswer:
    """Parse explicit inline citations without inferring or inventing evidence links."""
    claims: list[Claim] = []
    previous_line_had_claim = False

    for line in answer.splitlines():
        line = line.strip()
        if not line:
            previous_line_had_claim = False
            continue

        line_without_list_marker = re.sub(
            r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line
        ).strip()

        citation_matches = list(_CITATION_RE.finditer(line_without_list_marker))
        citation_ids = [match.group(1).strip() for match in citation_matches]
        if citation_matches and line_without_list_marker[
            citation_matches[-1].end():
        ].strip():
            # A citation followed by more prose is not in the required position.
            citation_ids = []
        text_without_citations = _CITATION_RE.sub(
            "", line_without_list_marker
        ).strip()

        # Citations on a heading do not support the claims that follow it.
        if _HEADING_RE.match(text_without_citations):
            previous_line_had_claim = False
            continue
        if _BOLD_HEADING_RE.fullmatch(text_without_citations):
            previous_line_had_claim = False
            continue

        # A standalone citation immediately after a paragraph belongs to the
        # preceding final claim only. It is never copied to sibling claims.
        if not text_without_citations:
            if citation_ids and claims and previous_line_had_claim:
                claims[-1].evidence_ids.extend(citation_ids)
            previous_line_had_claim = False
            continue

        # Keep sentence-to-citation association local: a citation at the end
        # of a line applies only to the final sentence on that line.
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text_without_citations)
            if part.strip()
        ]
        for index, sentence in enumerate(sentences):
            sentence_ids = citation_ids if index == len(sentences) - 1 else []
            claims.append(Claim(claim=sentence, evidence_ids=sentence_ids))
        previous_line_had_claim = bool(sentences)

    return StructuredAnswer(answer=answer, claims=claims)
