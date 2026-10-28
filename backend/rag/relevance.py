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
})
MIN_QUERY_COVERAGE = 0.60
MIN_MATCHED_TERMS = 2


def normalize_vietnamese(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text or "").lower().strip())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value)).strip()


def extract_keywords(text: str) -> set[str]:
    return {
        word for word in normalize_vietnamese(text).split()
        if len(word) >= 2 and word not in STOPWORDS
    }


def procedure_relevance(question: str, procedure_name: str) -> dict:
    """Use query-term recall so long qualified legal titles remain eligible.

    Only the procedure title participates. Description, fees, and procedural
    text can contain generic legal terms that would inflate a false match.
    """
    query_norm = normalize_vietnamese(question)
    title_norm = normalize_vietnamese(procedure_name)
    query_terms = extract_keywords(question)
    title_terms = extract_keywords(procedure_name)
    matched = query_terms & title_terms
    coverage = len(matched) / max(len(query_terms), 1)
    exact = bool(title_norm and title_norm in query_norm)
    relevant = exact or (
        len(matched) >= MIN_MATCHED_TERMS
        and coverage >= MIN_QUERY_COVERAGE
    )
    return {
        "relevant": relevant,
        "match_type": "exact" if exact else "query_coverage" if relevant else "below_query_coverage",
        "query_term_count": len(query_terms),
        "title_term_count": len(title_terms),
        "matched_terms": sorted(matched),
        "matched_term_count": len(matched),
        "query_coverage": coverage,
        "title_precision": len(matched) / max(len(title_terms), 1),
    }
