"""Pure matching and response validation for the official DVC search flow."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Iterable


class DVCUpstreamError(RuntimeError):
    """The DVC endpoint failed or returned a response outside its contract."""


_QUERY_STOPWORDS = frozenset(
    "toi minh ban cho can lam la ve thu tuc so gom nhung gi va bao lau "
    "thoi han giai quyet truc tuyen nhu the nao ra sao duoc khong co "
    "thuc hien tai dau nao mot cac thong tin xin hoi".split()
)
_ID_FIELDS = ("id", "formality_id", "formalityId", "procedureId", "uuid")
_NAME_FIELDS = (
    "procedure_name", "procedureName", "name", "title", "formalityName",
    "formality_name",
)
_CODE_FIELDS = ("code", "codeNotation", "procedure_code", "procedureCode")
MIN_QUERY_COVERAGE = 0.55
MIN_TITLE_SIMILARITY = 0.42
AMBIGUITY_MARGIN = 0.025
_CANCELLATION_TERMS = ("dung", "huy", "ngung", "xoa", "thu hoi", "cham dut")


def normalize_vietnamese(value: Any) -> str:
    text = str(value or "").lower()
    # Repair the legacy mojibake form of đ that is present in existing source
    # data, then remove Vietnamese diacritics for accent-insensitive matching.
    text = text.replace("\u0111", "d").replace("\u00e4\u2018", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def meaningful_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_vietnamese(value).split()
        if len(token) > 1 and token not in _QUERY_STOPWORDS
    }


def _first_value(row: Any, fields: tuple[str, ...]) -> str:
    if not isinstance(row, dict):
        return ""
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def procedure_match_score(row: Any, query: str) -> float | None:
    """Score a candidate using query coverage and title precision.

    Matching against the title prevents generic words in a long description
    (such as "giải quyết") from making an unrelated procedure eligible.
    """
    if not isinstance(row, dict):
        return None

    query_norm = normalize_vietnamese(query)
    name = _first_value(row, _NAME_FIELDS)
    code = _first_value(row, _CODE_FIELDS)
    code_norm = normalize_vietnamese(code)
    if code_norm and re.search(rf"(?:^| ){re.escape(code_norm)}(?: |$)", query_norm):
        return 1.0

    query_terms = meaningful_tokens(query)
    title_terms = meaningful_tokens(name)
    if len(query_terms) < 2 or len(title_terms) < 2:
        return None

    query_norm = f" {normalize_vietnamese(query)} "
    title_norm = f" {normalize_vietnamese(name)} "
    query_actions = {action for action in _CANCELLATION_TERMS if f" {action} " in query_norm}
    title_actions = {action for action in _CANCELLATION_TERMS if f" {action} " in title_norm}
    if title_actions and not query_actions:
        return None
    if query_actions and not (query_actions & title_actions):
        return None

    overlap = query_terms & title_terms
    if len(overlap) < 2:
        return None

    coverage = len(overlap) / len(query_terms)
    precision = len(overlap) / len(title_terms)
    harmonic = 2 * coverage * precision / (coverage + precision)
    if coverage < MIN_QUERY_COVERAGE or harmonic < MIN_TITLE_SIMILARITY:
        return None
    return harmonic


def select_best_procedure(rows: Iterable[Any], query: str) -> dict | None:
    scored = []
    seen_ids: set[str] = set()
    for row in rows:
        score = procedure_match_score(row, query)
        if score is None:
            continue
        row_id = _first_value(row, _ID_FIELDS)
        identity = row_id or repr(sorted(row.items()))
        if identity in seen_ids:
            continue
        seen_ids.add(identity)
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < AMBIGUITY_MARGIN:
        return None
    return scored[0][1]


def rank_local_index(dataset: Iterable[Any], query: str, limit: int = 5) -> list[dict]:
    ranked = []
    for row in dataset:
        score = procedure_match_score(row, query)
        if score is None:
            continue
        procedure_id = _first_value(row, _ID_FIELDS)
        name = _first_value(row, _NAME_FIELDS)
        if procedure_id and name:
            ranked.append({"id": procedure_id, "name": name, "score": score, "row": row})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    if len(ranked) > 1 and ranked[0]["score"] - ranked[1]["score"] < AMBIGUITY_MARGIN:
        return []
    return ranked[: max(0, limit)]


def extract_dvc_rows(response: Any) -> list[dict]:
    if isinstance(response, list):
        rows = response
    elif isinstance(response, dict):
        code = response.get("code")
        if code and str(code).upper() not in {"OK", "200", "SUCCESS"}:
            raise DVCUpstreamError("DVC search returned a non-success status")
        data = response.get("data")
        if isinstance(data, dict):
            row_key = next((key for key in ("rows", "items", "content") if key in data), None)
            if row_key is None:
                raise DVCUpstreamError("DVC search response is missing result rows")
            rows = data[row_key]
        elif isinstance(data, list):
            rows = data
        else:
            row_key = next((key for key in ("rows", "items") if key in response), None)
            if row_key is None:
                raise DVCUpstreamError("DVC search response has an invalid shape")
            rows = response[row_key]
    else:
        raise DVCUpstreamError("DVC search response has an invalid type")

    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise DVCUpstreamError("DVC search result rows are malformed")
    return rows


def fetch_dvc_rows(query: str, post_json: Callable[[str, dict], Any], endpoint: str) -> list[dict]:
    payload = {
        "limit": 10,
        "lastId": "",
        "q": query,
        "categoryId": "",
        "departmentCode": "",
        "formalityType": "STANDARD",
    }
    try:
        response = post_json(endpoint, payload)
    except Exception as exc:
        raise DVCUpstreamError("DVC search request failed") from exc
    return extract_dvc_rows(response)


def validate_dvc_detail_response(response: Any, expected_id: str) -> None:
    if not isinstance(response, dict):
        raise DVCUpstreamError("DVC detail response has an invalid type")
    code = response.get("code")
    if code and str(code).upper() not in {"OK", "200", "SUCCESS"}:
        raise DVCUpstreamError("DVC detail returned a non-success status")
    data = response.get("data")
    if not isinstance(data, (dict, list)) or not data:
        raise DVCUpstreamError("DVC detail response is missing detail data")
    if isinstance(data, dict):
        returned_id = _first_value(data, _ID_FIELDS)
        if returned_id and str(returned_id) != str(expected_id):
            raise DVCUpstreamError("DVC detail ID does not match the selected procedure")
