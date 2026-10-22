import re
import unicodedata

from .schemas import CatalogQuery, CatalogResult
from .repository import CatalogRepository


_STOPWORDS = frozenset(
    "toi minh muon hoi cho can lam la ve thu tuc thu tuc cua ve tai o dau "
    "nhu the nao ra sao bao lau giay to gom nhung gi duoc khong co "
    "xin vui long hay".split()
)
_MIN_COVERAGE = 0.55
_MIN_TITLE_PRECISION = 0.45
_MIN_SCORE = 0.58
_AMBIGUITY_MARGIN = 0.08


def _normalize(value: str | None) -> str:
    text = str(value or "").lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _tokens(value: str | None) -> set[str]:
    return {
        token for token in _normalize(value).split()
        if len(token) > 1 and token not in _STOPWORDS
    }


def _score_title(query_tokens: set[str], procedure_name: str | None) -> float | None:
    title_tokens = _tokens(procedure_name)
    if len(query_tokens) < 2 or len(title_tokens) < 2:
        return None
    overlap = query_tokens & title_tokens
    if len(overlap) < 2:
        return None
    coverage = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(title_tokens)
    harmonic = 2 * coverage * precision / (coverage + precision)
    if (
        coverage < _MIN_COVERAGE
        or precision < _MIN_TITLE_PRECISION
        or harmonic < _MIN_SCORE
    ):
        return None
    return harmonic


class CatalogMatcher:

    def __init__(self):
        self.repository = CatalogRepository()

    def match(
        self,
        query: CatalogQuery,
    ) -> CatalogResult:

        if query.domain != "legal":
            return CatalogResult(
                matched=False,
                match_type="unsupported_domain",
                confidence=0.0,
                reason="Domain not supported by this catalog",
            )

        record = None
        match_type = "none"
        match_confidence = 1.0

        if query.formality_id:
            record = self.repository.find_by_formality_id(
                query.formality_id
            )
            match_type = "exact_formality_id"

        elif query.code:
            record = self.repository.find_by_code(
                query.code
            )
            match_type = "exact_code"

        elif query.procedure_name:
            record = self.repository.find_by_name(
                query.procedure_name
            )
            match_type = "exact_name"

        else:
            return CatalogResult(
                matched=False,
                match_type="missing_lookup_key",
                confidence=0.0,
                reason="formality_id, code, or procedure_name is required",
            )

        if record is None and query.procedure_name:
            query_tokens = _tokens(query.procedure_name)
            candidates = self.repository.find_name_candidates(query_tokens)
            scored = []
            for candidate in candidates:
                score = _score_title(query_tokens, candidate.procedure_name)
                if score is not None:
                    scored.append((score, candidate))
            scored.sort(key=lambda item: item[0], reverse=True)

            if scored:
                if (
                    len(scored) > 1
                    and scored[0][0] - scored[1][0] < _AMBIGUITY_MARGIN
                ):
                    return CatalogResult(
                        matched=False,
                        match_type="ambiguous",
                        confidence=0.0,
                        reason="Multiple catalog procedure names match the query",
                    )
                record = scored[0][1]
                match_type = "natural_language_name"
                match_confidence = scored[0][0]

        if record is None:
            return CatalogResult(
                matched=False,
                match_type="none",
                confidence=0.0,
                reason="No sufficiently relevant catalog procedure name found",
            )

        return CatalogResult(
            matched=True,
            match_type=match_type,
            confidence=match_confidence,
            record=record,
        )
