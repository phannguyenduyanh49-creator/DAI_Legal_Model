
"""Vietnamese title relevance checks for local procedure retrieval."""

from __future__ import annotations

import re
import unicodedata


STOPWORDS = frozenset({
    "thu", "tuc", "thuc", "hien", "can", "nhung", "giay", "to", "gi",
    "nao", "bao", "lau", "o", "dau", "the", "co", "khong", "muc", "phi",
    "mat", "thoi", "han", "cho", "toi", "xin", "vui", "long", "hoi", "la",
    "mot", "loai", "cua", "nguoi", "duoc", "nhu", "theo", "ve", "voi",
    "trong", "tai", "ma", "hoac", "da", "lam",

    # Các từ mô tả câu hỏi, không nhất thiết xuất hiện trong tên thủ tục.
    "do", "tuoi", "thieu", "viet", "nam", "nhieu", "bao",
})

MIN_QUERY_COVERAGE = 0.25
MIN_MATCHED_TERMS = 1


def normalize_vietnamese(text: str) -> str:
    value = str(text or "").lower().strip()
    # Normalize the Vietnamese đ and repair the common legacy mojibake form
    # before stripping combining marks.
    value = value.replace("đ", "d").replace("ä\u2018", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(
        char for char in value
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9\s]", " ", value),
    ).strip()


def extract_keywords(text: str) -> set[str]:
    return {
        word
        for word in normalize_vietnamese(text).split()
        if len(word) >= 2 and word not in STOPWORDS
    }


def procedure_relevance(question: str, procedure_name: str) -> dict:
    """Check query-term coverage against the procedure title."""
    query_norm = normalize_vietnamese(question)
    title_norm = normalize_vietnamese(procedure_name)

    query_terms = extract_keywords(question)
    title_terms = extract_keywords(procedure_name)
    matched = query_terms & title_terms

    coverage = len(matched) / max(len(query_terms), 1)
    exact = bool(title_norm and title_norm in query_norm)

    # A long natural-language question often contains age, location, and
    # response-format details absent from the official procedure title.
    # Accept a strong title-anchored match when at least two meaningful terms
    # match and at least 45% of the title's meaningful terms are covered.
    # This avoids globally lowering query coverage and keeps generic matches out.
    title_anchored = (
        len(matched) >= 2
        and (len(matched) / max(len(title_terms), 1)) >= 0.45
    )
    relevant = exact or title_anchored or (
        len(matched) >= MIN_MATCHED_TERMS
        and coverage >= MIN_QUERY_COVERAGE
    )

    return {
        "relevant": relevant,
        "match_type": (
            "exact"
            if exact
            else "title_anchored"
            if relevant and title_anchored and coverage < MIN_QUERY_COVERAGE
            else "query_coverage"
            if relevant
            else "below_query_coverage"
        ),
        "query_term_count": len(query_terms),
        "title_term_count": len(title_terms),
        "matched_terms": sorted(matched),
        "matched_term_count": len(matched),
        "query_coverage": coverage,
        "title_precision": len(matched) / max(len(title_terms), 1),
    }
