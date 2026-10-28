import re

from backend.answer.schemas import Claim, StructuredAnswer


# Evidence IDs are opaque, but the ID syntax used by the current producers is
# restricted to printable non-whitespace characters other than the closing ].
_CITATION_RE = re.compile(
    r"\[Evidence ID:\s*([^\]\r\n]+?)\s*\]",
    re.IGNORECASE,
)
_CITATION_LIKE_RE = re.compile(r"\[Evidence\b", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_BOLD_HEADING_RE = re.compile(r"^\*\*[^*]+\*\*\s*:?[ \t]*$")
_LIST_MARKER_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")
_SUBSTANTIVE_HEADING_RE = re.compile(
    r"\b(?:ph\u1ea3i|\u0111\u01b0\u1ee3c|c\u00f3 quy\u1ec1n|"
    r"c\u00f3 ngh\u0129a v\u1ee5|c\u00f3 tr\u00e1ch nhi\u1ec7m|"
    r"l\u00e0|c\u00f3|kh\u00f4ng|g\u1ed3m|bao g\u1ed3m|"
    r"trong th\u1eddi h\u1ea1n|trong v\u00f2ng|"
    r"must|shall|is required to|has the right to|is|are|includes|within)\b",
    re.IGNORECASE,
)
_LIST_INTRO_RE = re.compile(
    r"^(?:"
    r".+\bc\u1ea7n\s+(?:nh\u1eefng|c\u00e1c)\s+"
    r"(?:gi\u1ea5y t\u1edd|h\u1ed3 s\u01a1|t\u00e0i li\u1ec7u|"
    r"th\u00e0nh ph\u1ea7n h\u1ed3 s\u01a1|b\u01b0\u1edbc|\u0111i\u1ec1u ki\u1ec7n|"
    r"tr\u01b0\u1eddng h\u1ee3p|n\u1ed9i dung)\s+sau"
    r"|(?:h\u1ed3 s\u01a1|danh s\u00e1ch|danh m\u1ee5c)\s+"
    r"(?:g\u1ed3m|bao g\u1ed3m|nh\u01b0 sau|sau \u0111\u00e2y)"
    r"|(?:c\u00e1c|nh\u1eefng)\s+"
    r"(?:gi\u1ea5y t\u1edd|h\u1ed3 s\u01a1|t\u00e0i li\u1ec7u|b\u01b0\u1edbc|"
    r"\u0111i\u1ec1u ki\u1ec7n|tr\u01b0\u1eddng h\u1ee3p|n\u1ed9i dung)\s+"
    r"(?:g\u1ed3m|bao g\u1ed3m|nh\u01b0 sau|sau \u0111\u00e2y)"
    r"|(?:the following (?:documents|steps|items)|"
    r"(?:the )?(?:required )?(?:documents|steps|items) (?:as follows|are as follows))"
    r")\s*:$",
    re.IGNORECASE,
)
_TRAILING_PUNCTUATION_RE = re.compile(r"[.!?,;:…]+[\"'’”)}\]]*$")


def _split_sentences(text: str) -> list[str]:
    """Split prose on sentence punctuation while retaining the source text."""
    pieces = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def _is_list_introduction(lines: list[str], line_index: int, text: str) -> bool:
    """Recognize explicit list labels immediately followed by a marked item."""
    text = re.sub(r"^#{1,6}\s+", "", text).strip()
    text = re.sub(r"^\*\*|\*\*$", "", text).strip()
    if not text.endswith(":") or _CITATION_LIKE_RE.search(text):
        return False
    if not _LIST_INTRO_RE.fullmatch(text):
        return False

    for following in lines[line_index + 1:]:
        candidate = following.strip()
        if not candidate:
            continue
        return bool(_LIST_MARKER_RE.match(candidate))
    return False


def structure_answer_with_explicit_evidence(answer: str) -> StructuredAnswer:
    """Extract claims and only their adjacent explicit citations.

    Raw answer text is retained byte-for-byte in ``StructuredAnswer.answer``.
    A citation on its own next line can qualify the immediately preceding
    claim, but blank lines/headings/list items prevent citation carry-over.
    """
    claims: list[Claim] = []
    parse_errors: list[str] = []
    previous_claim_index: int | None = None

    raw_lines = answer.splitlines()
    for line_index, raw_line in enumerate(raw_lines):
        line = raw_line.strip()
        if not line:
            previous_claim_index = None
            continue

        line = _LIST_MARKER_RE.sub("", line, count=1).strip()
        citation_matches = list(_CITATION_RE.finditer(line))
        malformed = len(list(_CITATION_LIKE_RE.finditer(line))) > len(citation_matches)
        if malformed:
            parse_errors.append("MALFORMED_CITATION")

        # Treat nominal headings as formatting, but keep assertion-like
        # headings in the claim path so they still need explicit Evidence.
        without_citations = _CITATION_RE.sub("", line).strip()
        heading_text = re.sub(r"^#{1,6}\s+", "", without_citations).strip()
        is_heading = bool(
            _HEADING_RE.match(line) or _BOLD_HEADING_RE.fullmatch(heading_text)
        )
        if is_heading and not _SUBSTANTIVE_HEADING_RE.search(heading_text):
            if citation_matches:
                parse_errors.append("CITATION_ON_HEADING")
            previous_claim_index = None
            continue

        if not citation_matches and _is_list_introduction(
            raw_lines, line_index, line
        ):
            # A colon alone is not enough: only explicit list-intro language
            # followed by a marked item is treated as structural.
            previous_claim_index = None
            continue

        if not citation_matches:
            sentence_indexes = []
            for sentence in _split_sentences(line):
                claims.append(Claim(claim=sentence))
                sentence_indexes.append(len(claims) - 1)
            previous_claim_index = sentence_indexes[-1] if sentence_indexes else None
            continue

        # Reject malformed/empty IDs explicitly. Do not normalize or invent IDs.
        citation_ids = [match.group(1).strip() for match in citation_matches]
        if any(not evidence_id for evidence_id in citation_ids):
            parse_errors.append("EMPTY_EVIDENCE_ID")

        citation_only_line = bool(citation_matches) and not re.sub(
            r"[.!?,;:…]+[\"'’”)}\]]*$", "", without_citations
        ).strip()
        if not without_citations or citation_only_line:
            # A citation on the immediately following line applies only to the
            # last claim on the previous nonblank line.
            if citation_ids and previous_claim_index is not None:
                claims[previous_claim_index].evidence_ids.extend(
                    evidence_id for evidence_id in citation_ids if evidence_id
                )
            elif citation_ids:
                parse_errors.append("ORPHAN_CITATION")
            previous_claim_index = None
            continue

        # Split around citations so only the final sentence before each
        # citation can receive it. Prose after a citation makes that citation
        # structurally ambiguous and is reported as invalid.
        cursor = 0
        line_claim_indexes: list[int] = []
        for citation_index, match in enumerate(citation_matches):
            preceding = line[cursor:match.start()].strip()
            following = line[match.end():]
            between_citations = following.split("[Evidence ID:", 1)[0]
            trailing_is_malformed_citation = bool(
                _CITATION_LIKE_RE.search(following)
                and not _CITATION_RE.search(following)
            )
            adjacent_next_citation = (
                citation_index + 1 < len(citation_matches)
                and not line[match.end():citation_matches[citation_index + 1].start()].strip()
            )
            citation_is_terminal = (
                not following.strip()
                or bool(_TRAILING_PUNCTUATION_RE.fullmatch(between_citations.strip()))
                or trailing_is_malformed_citation
                or adjacent_next_citation
            )
            sentences = _split_sentences(preceding)
            if sentences:
                for sentence in sentences:
                    claims.append(Claim(claim=sentence))
                    line_claim_indexes.append(len(claims) - 1)
                before_citation = preceding.rstrip()
                if citation_is_terminal:
                    if citation_ids:
                        claims[line_claim_indexes[-1]].evidence_ids.append(
                            citation_ids[citation_index]
                        )
                else:
                    parse_errors.append("CITATION_NOT_AT_CLAIM_END")
            elif citation_ids and line_claim_indexes and citation_is_terminal:
                # Adjacent citation tokens after one sentence can all support
                # that same claim, but never a later claim.
                claims[line_claim_indexes[-1]].evidence_ids.append(
                    citation_ids[citation_index]
                )
            else:
                # A citation without preceding claim text is either an orphan
                # or an invalid prefix; never attach it to following prose.
                parse_errors.append("CITATION_WITHOUT_PRECEDING_CLAIM")
            cursor = match.end()

        tail = line[cursor:].strip()
        if tail and not (
            _CITATION_LIKE_RE.search(tail)
            and not _CITATION_RE.search(tail)
        ):
            # If the citation is followed only by punctuation, it belongs to
            # the preceding claim. Any substantive trailing text stays a
            # separate, uncited claim.
            if citation_matches and _TRAILING_PUNCTUATION_RE.fullmatch(tail):
                pass
            else:
                tail_sentences = _split_sentences(tail)
                for sentence in tail_sentences:
                    claims.append(Claim(claim=sentence))
                    line_claim_indexes.append(len(claims) - 1)
                if citation_matches:
                    parse_errors.append("PROSE_AFTER_CITATION")

        previous_claim_index = line_claim_indexes[-1] if line_claim_indexes else None

    return StructuredAnswer(
        answer=answer,
        claims=claims,
        metadata={"citation_parse_errors": sorted(set(parse_errors))},
    )
