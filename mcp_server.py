import sys
import re
import json
import unicodedata
import urllib.request
import urllib.error
import urllib.parse
import os
import time
from pathlib import Path

from typing import Any

from qdrant_client import QdrantClient
from mcp.server import MCPServer
from backend.mcp.runtime import guarded_tool
from backend.mcp.dvc_search import (
    DVCUpstreamError,
    fetch_dvc_rows,
    normalize_vietnamese as normalize_dvc_text,
    rank_local_index,
    select_best_procedure,
    validate_dvc_detail_response,
)


# ============================================================
# CONFIG
# ============================================================

QDRANT_URL = os.getenv("MCP_QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("MCP_QDRANT_COLLECTION", "legal_docs")

EMBEDDING_MODEL = os.getenv(
    "MCP_EMBEDDING_MODEL",
    "bkai-foundation-models/vietnamese-bi-encoder",
)

_DVC_DATASET_OVERRIDE = os.getenv("DVC_DATASET_PATH")
_DVC_DATASET_CANDIDATES = (
    [Path(_DVC_DATASET_OVERRIDE).expanduser()]
    if _DVC_DATASET_OVERRIDE
    else [
        Path(__file__).resolve().parent / "data" / "dichvucong_procedures",
        Path(__file__).resolve().parent / "backend" / "mcp" / "data" / "dichvucong_procedures",
        Path(__file__).resolve().parent / "backend" / "data" / "dichvucong_procedures",
    ]
)
DVC_DATASET_PATH = str(_DVC_DATASET_CANDIDATES[0])

DVC_BASE_URL = "https://dichvucong.gov.vn"

DVC_SEARCH_API = (
    DVC_BASE_URL
    + "/api/v1/submitting/formality-top-search/list-by-citizen"
)

DVC_DETAIL_API = (
    DVC_BASE_URL
    + "/api/v1/configuring/formality/get-formality-by-citizen"
)

ALLOWED_HOST = "dichvucong.gov.vn"


# ============================================================
# INITIALIZE
# ============================================================

embedding_model = None


def get_embedding_model():
    """Load the local retrieval encoder only for tools that need embeddings."""
    global embedding_model
    if embedding_model is None:
        started_at = time.perf_counter()
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(
            "[MCP STAGE]",
            "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
            "stage=embedding_model_load",
            "status=completed",
            "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
            file=sys.stderr,
            flush=True,
        )
    return embedding_model


qdrant = QdrantClient(url=QDRANT_URL)

dvc_dataset = None


def get_dvc_dataset():
    global dvc_dataset
    if dvc_dataset is not None:
        return dvc_dataset

    started_at = time.perf_counter()
    from datasets import load_from_disk

    errors = []
    for dataset_path in _DVC_DATASET_CANDIDATES:
        try:
            if not dataset_path.exists():
                errors.append(f"{dataset_path}: path does not exist")
                continue

            dvc_dataset = load_from_disk(str(dataset_path))
            print(
                "[MCP STAGE]",
                "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
                "stage=dvc_local_index_load", "status=completed",
                "path=", str(dataset_path),
                "count=", len(dvc_dataset),
                "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
                file=sys.stderr,
                flush=True,
            )
            return dvc_dataset
        except Exception as exc:
            errors.append(f"{dataset_path}: {type(exc).__name__}: {exc}")

    print(
        "[MCP STAGE]",
        "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
        "stage=dvc_local_index_load", "status=failed",
        "tried_paths=", " | ".join(str(p) for p in _DVC_DATASET_CANDIDATES),
        "errors=", " | ".join(errors),
        "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
        file=sys.stderr,
        flush=True,
    )
    return None


mcp = MCPServer("DAI Legal MCP")


# ============================================================
# NORMALIZE
# ============================================================

def normalize_vietnamese(text: str):
    text = str(text or "").lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# PROCEDURE DETECTION
# ============================================================

PROCEDURE_PATTERNS = [
    "thủ tục",
    "hồ sơ",
    "giấy tờ",
    "đăng ký",
    "cấp",
    "xin cấp",
    "nộp hồ sơ",
    "làm ở đâu",
    "thực hiện ở đâu",
    "cách làm",
    "cách thực hiện",
    "trình tự",
    "thời hạn giải quyết",
    "mất bao lâu",
    "phí",
    "lệ phí",
    "điều kiện",
    "cần những gì",
    "cần gì",
    "đăng ký ở đâu",
    "nộp ở đâu",
    "cấp lại",
    "đổi",
    "gia hạn",
    "xác nhận",
    "chứng thực",
    "khai sinh",
    "kết hôn",
    "tạm trú",
    "tạm vắng",
    "hộ chiếu",
    "giấy phép",
    "lý lịch tư pháp",
]

NON_PROCEDURE_PATTERNS = [
    "phạt bao nhiêu",
    "mức phạt",
    "bị phạt",
    "phạt tiền",
    "vi phạm",
    "xử phạt",
    "án bao nhiêu",
    "mức án",
    "tội gì",
    "có bị đi tù",
    "đi tù bao lâu",
]


def is_procedure_question(question: str) -> bool:
    """Avoid rejecting short but meaningful procedure queries before DVC search.

    Core calls this MCP tool only after local retrieval is insufficient and
    external lookup is permitted. Therefore the gate should reject explicit
    non-procedure/criminal-penalty questions, not require an exact phrase such
    as "thủ tục" or "đăng ký". Short queries like "khai sinh" or "hộ chiếu"
    must be allowed through to the official search endpoint.
    """
    q = normalize_vietnamese(question)
    if not q:
        return False

    for pattern in NON_PROCEDURE_PATTERNS:
        if normalize_vietnamese(pattern) in q:
            print(
                f"[MCP DVC GATE] decision=reject reason=explicit_non_procedure pattern={pattern!r}",
                file=sys.stderr,
                flush=True,
            )
            return False

    for pattern in PROCEDURE_PATTERNS:
        if normalize_vietnamese(pattern) in q:
            print(
                f"[MCP DVC GATE] decision=allow reason=procedure_phrase pattern={pattern!r}",
                file=sys.stderr,
                flush=True,
            )
            return True

    # Accept a compact query with at least two meaningful words. This covers
    # procedure names entered directly without question framing.
    tokens = [token for token in re.findall(r"[a-z0-9]+", q) if len(token) > 1]
    allowed = len(tokens) >= 2
    print(
        "[MCP DVC GATE]",
        f"decision={'allow' if allowed else 'reject'}",
        f"reason={'short_procedure_terms' if allowed else 'insufficient_query_terms'}",
        f"token_count={len(tokens)}",
        file=sys.stderr,
        flush=True,
    )
    return allowed


# ============================================================
# EXTRACT KEYWORD
# ============================================================

REMOVE_PHRASES = [
    "thủ tục",
    "cho tôi biết",
    "cho tôi hỏi",
    "tôi muốn hỏi",
    "xin hỏi",
    "hỏi về",
    "cần những giấy tờ gì",
    "cần giấy tờ gì",
    "cần những gì",
    "cần gì",
    "gồm những gì",
    "gồm những giấy tờ gì",
    "mất bao lâu",
    "bao lâu",
    "thực hiện ở đâu",
    "làm ở đâu",
    "đăng ký ở đâu",
    "nộp ở đâu",
    "như thế nào",
    "thế nào",
    "ra sao",
    "có mất phí không",
    "mất phí không",
    "bao nhiêu tiền",
    "có lệ phí bao nhiêu",
    "lệ phí bao nhiêu",
    "phí bao nhiêu",
    "có lệ phí",
    "lệ phí",
    "phí",
]


def extract_dvc_keyword(question: str) -> str:
    keyword = str(question or "").strip()

    for phrase in REMOVE_PHRASES:
        keyword = re.sub(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            " ",
            keyword,
            flags=re.IGNORECASE,
        )

    keyword = re.sub(
        r"\b(có|là|bao|nhiêu|không)\b",
        " ",
        keyword,
        flags=re.IGNORECASE,
    )
    keyword = re.sub(r"[?？!！,.。:;]+", " ", keyword)
    keyword = re.sub(r"\s+", " ", keyword).strip()

    return keyword


# ============================================================
# HTTP JSON
# ============================================================

def post_json(url: str, payload: dict, timeout: int = 30) -> Any:
    """POST JSON to the official DVC API with bounded transient retries.

    Retries only transport failures and transient HTTP statuses. Permanent 4xx
    responses and invalid JSON are surfaced immediately with diagnostic logs.
    """
    started_at = time.perf_counter()
    request_id = os.getenv("DAI_REQUEST_ID", "unknown")
    step = "detail" if url == DVC_DETAIL_API else "search"

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("DVC request chỉ được phép dùng HTTPS.")
    if parsed.hostname != ALLOWED_HOST:
        raise ValueError(f"Host không được phép: {parsed.hostname}")

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36"
        ),
        "Origin": DVC_BASE_URL,
        "Referer": DVC_BASE_URL + "/",
    }
    max_attempts = 3
    transient_statuses = {408, 425, 429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                text = raw.decode("utf-8", errors="replace")
                print(
                    "[MCP HTTP]", "request_id=", request_id,
                    "source=dichvucong", "step=", step,
                    "status_code=", response.status,
                    "attempt=", attempt, "retry_count=", attempt - 1,
                    "response_chars=", len(text),
                    "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
                    file=sys.stderr, flush=True,
                )
                if not text.strip():
                    raise ValueError("DVC trả về response rỗng.")
                try:
                    parsed_json = json.loads(text)
                except json.JSONDecodeError as exc:
                    print(
                        "[MCP HTTP ERROR]", "request_id=", request_id,
                        "step=", step, "reason=invalid_json",
                        "response_preview=", repr(text[:500]),
                        file=sys.stderr, flush=True,
                    )
                    raise DVCUpstreamError("DVC returned invalid JSON") from exc

                if isinstance(parsed_json, dict):
                    data_value = parsed_json.get("data")
                    data_keys = sorted(data_value.keys())[:30] if isinstance(data_value, dict) else []
                    print(
                        "[MCP HTTP JSON]", "request_id=", request_id,
                        "step=", step,
                        "top_level_keys=", sorted(parsed_json.keys())[:30],
                        "data_type=", type(data_value).__name__,
                        "data_keys=", data_keys,
                        file=sys.stderr, flush=True,
                    )
                elif isinstance(parsed_json, list):
                    print(
                        "[MCP HTTP JSON]", "request_id=", request_id,
                        "step=", step, "top_level_type=list",
                        "top_level_count=", len(parsed_json),
                        file=sys.stderr, flush=True,
                    )
                else:
                    print(
                        "[MCP HTTP JSON]", "request_id=", request_id,
                        "step=", step,
                        "top_level_type=", type(parsed_json).__name__,
                        file=sys.stderr, flush=True,
                    )
                return parsed_json

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            retryable = exc.code in transient_statuses and attempt < max_attempts
            print(
                "[MCP HTTP ERROR]", "request_id=", request_id,
                "source=dichvucong", "step=", step,
                "status_code=", exc.code, "reason=", exc.reason,
                "attempt=", attempt, "retry=", retryable,
                "body=", body,
                "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
                file=sys.stderr, flush=True,
            )
            if not retryable:
                raise DVCUpstreamError(
                    f"DVC HTTP {exc.code}: {exc.reason}; body={body[:300]}"
                ) from exc
            time.sleep(0.4 * (2 ** (attempt - 1)))

        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            retryable = attempt < max_attempts
            print(
                "[MCP HTTP TRANSPORT ERROR]", "request_id=", request_id,
                "source=dichvucong", "step=", step,
                "exception_type=", type(exc).__name__,
                "reason=", repr(getattr(exc, "reason", exc)),
                "attempt=", attempt, "retry=", retryable,
                "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
                file=sys.stderr, flush=True,
            )
            if not retryable:
                raise DVCUpstreamError(
                    f"DVC transport failed after {attempt} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            time.sleep(0.4 * (2 ** (attempt - 1)))

    raise DVCUpstreamError("DVC request exhausted retry attempts")

# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_text_from_object(obj: Any, max_length: int = 30000) -> str:
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj[:max_length]

    if isinstance(obj, (int, float, bool)):
        return str(obj)

    if isinstance(obj, list):
        parts = []
        for item in obj:
            text = extract_text_from_object(item, max_length)
            if text:
                parts.append(text)
        return "\n".join(parts)[:max_length]

    if isinstance(obj, dict):
        parts = []
        for key, value in obj.items():
            if value is None:
                continue

            text = extract_text_from_object(value, max_length)
            if text:
                parts.append(f"{key}: {text}")

            if sum(len(x) for x in parts) >= max_length:
                break

        return "\n".join(parts)[:max_length]

    return str(obj)[:max_length]


# ============================================================
# GET NAME
# ============================================================

def get_procedure_name(row: dict) -> str:
    fields = [
        "name",
        "procedureName",
        "procedure_name",
        "title",
        "formalityName",
        "formality_name",
    ]

    for field in fields:
        value = row.get(field)
        if value:
            return str(value).strip()

    return ""


# ============================================================
# GET ID
# ============================================================

def get_procedure_id(row: dict):
    fields = [
        "id",
        "formality_id",
        "formalityId",
        "procedure_id",
        "procedureId",
        "uuid",
    ]

    for field in fields:
        value = row.get(field)
        if value:
            return value

    return None


# ============================================================
# FIND BEST PROCEDURE
# ============================================================

def find_best_procedure(rows, keyword):
    selected = select_best_procedure(rows, keyword)
    if selected:
        print(
            "DVC MATCH: relevant official search result",
            file=sys.stderr,
        )
    return selected


# ============================================================
# BUILD DETAIL
# ============================================================

def build_detail_text(detail) -> str:
    if not detail:
        return ""

    if isinstance(detail, dict):
        if detail.get("data") is not None:
            detail = detail.get("data")

    return extract_text_from_object(detail, 30000)


# ============================================================
# LOCAL QDRANT
# ============================================================

@mcp.tool()
@guarded_tool("search_legal_documents")
def search_legal_documents(query: str, limit: int = 5) -> str:
    query = str(query or "").strip()

    if not query:
        return "Query rỗng."

    try:
        vector = get_embedding_model().encode(query).tolist()

        result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        points = result.points

        if not points:
            return "Không tìm thấy dữ liệu pháp luật phù hợp."

        output = []

        for index, point in enumerate(points, start=1):
            payload = point.payload or {}

            output.append(
                f"""
--- RESULT {index} ---
Score: {point.score}

Thủ tục:
{payload.get("thu_tuc", "")}

Nội dung:
{payload.get("text", "")}

Nguồn:
{payload.get("source", "")}
""".strip()
            )

        return "\n\n".join(output)

    except Exception as e:
        print(
            f"LOCAL RAG ERROR: {repr(e)}",
            file=sys.stderr,
        )
        return "Lỗi khi tìm kiếm Qdrant: " + str(e)


# ============================================================
# DVC MULTI-QUERY + LOCAL INDEX FALLBACK
# ============================================================

def build_dvc_search_queries(keyword: str) -> list[str]:
    """Build conservative search variants without broadening to generic terms.

    DVC's public search may return no rows when a query includes location or
    question wording. Try the full phrase first, then a small number of
    increasingly concise procedure-name phrases. Never issue one-word queries.
    """
    query = re.sub(r"\s+", " ", str(keyword or "")).strip(" .?!;,:-")
    if not query:
        return []

    candidates = [query]
    simplified = query
    # Location qualifiers often appear in the question but not in the official
    # procedure title. Keep the full query as the first attempt.
    for phrase in (
        "ở trong nước", "tại trong nước", "trong nước",
        "ở nước ngoài", "tại nước ngoài",
    ):
        simplified = re.sub(re.escape(phrase), " ", simplified, flags=re.IGNORECASE)
    simplified = re.sub(r"\s+", " ", simplified).strip(" .?!;,:-")
    if simplified and simplified.casefold() != query.casefold():
        candidates.append(simplified)

    # Strip only common question framing; keep the procedure's core nouns.
    core = simplified or query
    for phrase in (
        "đăng ký thủ tục", "thủ tục về", "thủ tục", "hồ sơ về",
        "hướng dẫn về", "hướng dẫn",
    ):
        core = re.sub(re.escape(phrase), " ", core, flags=re.IGNORECASE)
    core = re.sub(r"\s+", " ", core).strip(" .?!;,:-")
    if core and len(core.split()) >= 2:
        candidates.append(core)

    # Do not arbitrarily truncate the end of a procedure name: terms such as
    # "Việt Nam" can be essential to the official title. The full phrase,
    # location-stripped phrase, and question-framing-stripped phrase are enough.

    unique = []
    seen = set()
    for item in candidates:
        item = re.sub(r"\s+", " ", item).strip()
        key = item.casefold()
        if len(item.split()) >= 2 and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:4]


def search_dvc_rows(query: str) -> list:
    started_at = time.perf_counter()

    rows = fetch_dvc_rows(
        query,
        lambda url, payload: post_json(url, payload),
        DVC_SEARCH_API,
    )

    print(
        "[MCP DVC]",
        "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
        "stage=search_parse",
        "status=completed",
        "raw_count=", len(rows),
        "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
        file=sys.stderr,
        flush=True,
    )

    return rows


def search_local_dvc_index(query: str, limit: int = 5) -> list:
    """Resolve relevant official IDs from the local index, then fetch DVC detail."""
    dataset = get_dvc_dataset()

    # FIX: never pass None to rank_local_index().
    if dataset is None:
        print(
            "[MCP DVC]",
            "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
            "stage=local_id_resolution",
            "status=skipped",
            "reason=local_dataset_unavailable",
            file=sys.stderr,
            flush=True,
        )
        return []

    ranked = rank_local_index(dataset, query, limit=limit)

    print(
        "[MCP DVC]",
        "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
        "stage=local_id_resolution",
        "status=completed",
        "match_count=", len(ranked),
        file=sys.stderr,
        flush=True,
    )

    return ranked


def get_dvc_detail_by_id(procedure_id: str) -> Any:
    return post_json(
        DVC_DETAIL_API,
        {"id": procedure_id},
    )


# ============================================================
# SEARCH PROCEDURE
# ============================================================

@mcp.tool()
@guarded_tool("search_procedure")
def search_procedure(keyword: str, limit: int = 5) -> str:
    return search_legal_documents.__wrapped__(keyword, limit)


# ============================================================
# DỊCH VỤ CÔNG
# ============================================================

@mcp.tool()
@guarded_tool("search_dichvucong")
def search_dichvucong(keyword: str) -> str:
    tool_started_at = time.perf_counter()
    original_question = str(keyword or "").strip()

    if not is_procedure_question(original_question):
        return (
            "DVC_NO_RELEVANT_RESULT: "
            "query is not an administrative procedure request."
        )

    dvc_keyword = extract_dvc_keyword(original_question)
    queries = build_dvc_search_queries(dvc_keyword)

    if not queries:
        return (
            "DVC_NO_RELEVANT_RESULT: "
            "query contains no searchable procedure terms."
        )

    # An upstream failure is propagated by guarded_tool as an MCP error.
    # It must not be treated as an empty search result.
    rows = []
    for query_index, query in enumerate(queries, start=1):
        print(
            "[MCP DVC]",
            "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
            "stage=search_query",
            "query_index=", query_index,
            "query_count=", len(queries),
            "query=", repr(query),
            file=sys.stderr,
            flush=True,
        )
        query_rows = search_dvc_rows(query)
        print(
            "[MCP DVC]",
            "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
            "stage=search_query_result",
            "query_index=", query_index,
            "result_count=", len(query_rows),
            file=sys.stderr,
            flush=True,
        )
        rows.extend(query_rows)

    raw_search_count = len(rows)
    print(
        "[MCP DVC]",
        "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
        "stage=official_search_aggregate",
        "raw_count=", raw_search_count,
        "keyword=", repr(dvc_keyword),
        file=sys.stderr,
        flush=True,
    )
    unique_rows = {}

    for row in rows:
        procedure_id = get_procedure_id(row)
        identity = (
            str(procedure_id)
            if procedure_id
            else repr(sorted(row.items()))
        )
        unique_rows[identity] = row

    official_rows = list(unique_rows.values())
    selected = find_best_procedure(official_rows, dvc_keyword)

    selected_source = "official_search_matcher"

    # The official endpoint is itself a ranked search. Do not turn a valid
    # official result set into NO_MATCH merely because our local lexical
    # matcher cannot recognize a synonym or a different title formulation.
    if selected is None and official_rows:
        for official_row in official_rows:
            if get_procedure_id(official_row) and get_procedure_name(official_row):
                selected = official_row
                selected_source = "official_search_ranked_fallback"
                break

    if selected is None:
        fallback_started_at = time.perf_counter()
        fallback_by_id = {}
        for query in queries:
            candidates = search_local_dvc_index(query, limit=5)
            for candidate in candidates:
                candidate_id = str(candidate.get("id") or get_procedure_id(candidate.get("row", {})) or "")
                if not candidate_id:
                    continue
                previous = fallback_by_id.get(candidate_id)
                if previous is None or float(candidate.get("score", 0)) > float(previous.get("score", 0)):
                    fallback_by_id[candidate_id] = candidate

        fallback_candidates = sorted(
            fallback_by_id.values(),
            key=lambda candidate: float(candidate.get("score", 0)),
            reverse=True,
        )

        print(
            "[MCP DVC]",
            "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
            "stage=local_id_resolution",
            "status=completed" if fallback_candidates else "status=no_match",
            "candidate_count=", len(fallback_candidates),
            "query_count=", len(queries),
            "elapsed_ms=", round((time.perf_counter() - fallback_started_at) * 1000, 2),
            file=sys.stderr,
            flush=True,
        )

        if not fallback_candidates:
            return (
                "DVC_NO_RELEVANT_RESULT: "
                "no sufficiently relevant official procedure matched."
            )

        # Do not silently choose between near-tied, different procedures.
        if len(fallback_candidates) > 1 and (
            float(fallback_candidates[0].get("score", 0))
            - float(fallback_candidates[1].get("score", 0))
        ) < 0.025:
            return (
                "DVC_NO_RELEVANT_RESULT: multiple similarly relevant procedures; "
                "please provide a more specific procedure name or administrative level."
            )

        selected = fallback_candidates[0]["row"]
        selected_source = "local_index_id_resolver"

    procedure_id = get_procedure_id(selected)
    procedure_name = get_procedure_name(selected)

    if not procedure_id or not procedure_name:
        raise DVCUpstreamError(
            "Selected DVC procedure is missing its official ID or name"
        )

    procedure_code = str(
        selected.get("code")
        or selected.get("codeNotation")
        or ""
    )

    detail_started_at = time.perf_counter()
    detail_response = get_dvc_detail_by_id(str(procedure_id))

    validate_dvc_detail_response(
        detail_response,
        str(procedure_id),
    )

    detail_text = build_detail_text(detail_response)

    if not detail_text.strip():
        raise DVCUpstreamError(
            "DVC detail response contains no procedure detail"
        )

    print(
        f"DVC MATCH SOURCE: {selected_source}",
        file=sys.stderr,
        flush=True,
    )

    print(
        "[MCP DVC]",
        "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
        "stage=detail_mapping",
        "status=completed",
        "detail_elapsed_ms=", round((time.perf_counter() - detail_started_at) * 1000, 2),
        "raw_search_count=", raw_search_count,
        "unique_search_count=", len(unique_rows),
        "tool_elapsed_ms=", round((time.perf_counter() - tool_started_at) * 1000, 2),
        file=sys.stderr,
        flush=True,
    )

    return f"""
DVC_OFFICIAL_PROCEDURE
DVC_PROCEDURE_NAME: {procedure_name}
DVC_PROCEDURE_CODE: {procedure_code}
DVC_PROCEDURE_ID: {procedure_id}
DVC_SOURCE_URL: {DVC_BASE_URL}
DVC_DETAIL_BEGIN
{detail_text}
DVC_DETAIL_END
""".strip()
# ============================================================
# START MCP
# ============================================================

if __name__ == "__main__":
    print("============================================", file=sys.stderr)
    print("DAI LEGAL MCP SERVER", file=sys.stderr)
    print("Local Qdrant + Official DVC API", file=sys.stderr)
    print("============================================", file=sys.stderr)
    print(f"Qdrant: {QDRANT_URL}", file=sys.stderr)
    print(f"Collection: {COLLECTION_NAME}", file=sys.stderr)
    print(f"DVC: {DVC_BASE_URL}", file=sys.stderr)
    print("Starting MCP stdio server...", file=sys.stderr, flush=True)

    mcp.run(transport="stdio")    