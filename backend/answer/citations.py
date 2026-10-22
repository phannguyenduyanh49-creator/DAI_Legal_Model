
import re

from backend.answer.schemas import Claim, StructuredAnswer


_CITATION_RE = re.compile(
    r"\[Evidence ID:\s*([^\]]+)\]",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_BOLD_HEADING_RE = re.compile(r"\*\*[^*]+\*\*")


def structure_answer_with_explicit_evidence(
    answer: str,
) -> StructuredAnswer:
    """Parse explicit citations without inventing evidence links."""
    claims: list[Claim] = []
    previous_line_had_claim = False

    for line in answer.splitlines():
        line = line.strip()

        if not line:
            previous_line_had_claim = False
            continue

        # Remove Markdown list markers.
        line = re.sub(
            r"^(?:[-*+]\s+|\d+[.)]\s+)",
            "",
            line,
        ).strip()

        citation_matches = list(_CITATION_RE.finditer(line))
        citation_ids = [
            match.group(1).strip()
            for match in citation_matches
        ]

        text_without_citations = _CITATION_RE.sub("", line).strip()

        # Ignore headings; they do not support claims.
        if (
            _HEADING_RE.match(text_without_citations)
            or _BOLD_HEADING_RE.fullmatch(text_without_citations)
        ):
            previous_line_had_claim = False
            continue

        # A standalone citation binds only to the immediately preceding
        # claim, never to earlier claims in the paragraph.
        if not text_without_citations:
            if citation_ids and claims and previous_line_had_claim:
                claims[-1].evidence_ids.extend(citation_ids)

            previous_line_had_claim = False
            continue

        # A citation is accepted only if it is at the end of the line.
        citation_at_end = bool(
            citation_matches
            and not line[citation_matches[-1].end():].strip()
        )

        # Split the line into individual sentences.
        sentences = [
            part.strip()
            for part in re.split(
                r"(?<=[.!?])\s+",
                text_without_citations,
            )
            if part.strip()
        ]

        for index, sentence in enumerate(sentences):
            # The trailing citation supports only the final sentence.
            sentence_ids = (
                citation_ids
                if citation_at_end and index == len(sentences) - 1
                else []
            )

            claims.append(
                Claim(
                    claim=sentence,
                    evidence_ids=sentence_ids,
                )
            )

        previous_line_had_claim = bool(sentences)

    return StructuredAnswer(
        answer=answer,
        claims=claims,
    )
