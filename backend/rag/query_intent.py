"""Conservative query intent helpers for retrieval breadth decisions."""

from dataclasses import dataclass
import re

from backend.rag.relevance import normalize_vietnamese


MULTI_EVIDENCE_MARKERS = (
    "so sanh",
    "khac nhau",
    "khac biet",
    "doi chieu",
    "tong hop",
    "liet ke",
    "nhung giay to gi",
    "nhung loai nao",
    "cac loai",
    "bao gom nhung gi",
    "compare",
    "comparison",
    "versus",
)


@dataclass(frozen=True)
class RetrievalBreadth:
    multi_evidence: bool
    local_candidate_limit: int
    rerank_result_limit: int
    apply_rerank_margin: bool


def requires_multiple_evidence(query: str) -> bool:
    """Recognize explicit comparison/list/synthesis requests, not generic length."""
    normalized = normalize_vietnamese(query)
    if not normalized:
        return False
    return (
        any(marker in normalized for marker in MULTI_EVIDENCE_MARKERS)
        or bool(re.search(r"\bkhac\b.*\bdiem nao\b", normalized))
        or bool(re.search(r"\bdifference\b.*\bbetween\b|\bcompare\b", normalized))
    )


def retrieval_breadth(
    query: str,
    *,
    standard_local_limit: int = 10,
    standard_rerank_limit: int = 3,
    multi_local_limit: int = 20,
    multi_rerank_limit: int = 6,
) -> RetrievalBreadth:
    multi_evidence = requires_multiple_evidence(query)
    return RetrievalBreadth(
        multi_evidence=multi_evidence,
        local_candidate_limit=(multi_local_limit if multi_evidence else standard_local_limit),
        rerank_result_limit=(multi_rerank_limit if multi_evidence else standard_rerank_limit),
        apply_rerank_margin=not multi_evidence,
    )


def should_use_exact_match_shortcut(query: str, *, structured_list_query: bool) -> bool:
    """Exact-match selection is unsafe for requests that need comparison or synthesis."""
    return not structured_list_query and not requires_multiple_evidence(query)
