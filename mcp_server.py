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

# DVC local index: chỉ dùng để RESOLVE formality_id khi endpoint top-search
# không trả đúng thủ tục. Dữ liệu cuối cùng vẫn lấy từ DVC API chính thức.
DVC_DATASET_PATH = os.getenv(
    "DVC_DATASET_PATH",
    str(Path(__file__).resolve().parent / "data" / "dichvucong_procedures"),
)

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

qdrant = QdrantClient(
    url=QDRANT_URL
)

# This is NOT the external source of truth. It is loaded only if the official
# top-search endpoint cannot resolve a title to an ID.
dvc_dataset = None


def get_dvc_dataset():
    global dvc_dataset
    if dvc_dataset is not None:
        return dvc_dataset
    started_at = time.perf_counter()
    try:
        from datasets import load_from_disk

        dvc_dataset = load_from_disk(DVC_DATASET_PATH)
        print(
            "[MCP STAGE]",
            "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
            "stage=dvc_local_index_load", "status=completed",
            "count=", len(dvc_dataset),
            "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(
            "[MCP STAGE]",
            "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
            "stage=dvc_local_index_load", "status=failed",
            "exception_type=", type(exc).__name__,
            "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
            file=sys.stderr,
            flush=True,
        )
        return None
    return dvc_dataset

mcp = MCPServer(
    "DAI Legal MCP"
)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_vietnamese(text: str) -> str:

    text = str(text or "").lower()

    text = text.replace(
        "đ",
        "d"
    )

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        c
        for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

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


def is_procedure_question(
    question: str
) -> bool:

    q = normalize_vietnamese(
        question
    )

    for pattern in NON_PROCEDURE_PATTERNS:

        if normalize_vietnamese(
            pattern
        ) in q:

            print(
                f"DVC FILTER: KHONG PHAI THU TUC -> {pattern}",
                file=sys.stderr
            )

            return False

    for pattern in PROCEDURE_PATTERNS:

        if normalize_vietnamese(
            pattern
        ) in q:

            return True

    return False


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

    # Cụm hỏi về phí/lệ phí: chỉ giữ lại chủ đề thủ tục cho DVC.
    "có lệ phí bao nhiêu",
    "lệ phí bao nhiêu",
    "phí bao nhiêu",
    "có lệ phí",
    "lệ phí",
    "phí",
]


def extract_dvc_keyword(
    question: str
) -> str:

    keyword = str(
        question or ""
    ).strip()

    for phrase in REMOVE_PHRASES:

        keyword = re.sub(
            re.escape(phrase),
            " ",
            keyword,
            flags=re.IGNORECASE
        )

    # Loại các từ hỏi còn sót sau khi bỏ cụm phí/lệ phí.
    keyword = re.sub(
        r"\b(có|là|bao|nhiêu|không)\b",
        " ",
        keyword,
        flags=re.IGNORECASE
    )

    keyword = re.sub(
        r"[?？!！,.。:;]+",
        " ",
        keyword
    )

    keyword = re.sub(
        r"\s+",
        " ",
        keyword
    ).strip()

    return keyword


# ============================================================
# HTTP JSON
# ============================================================

def post_json(
    url: str,
    payload: dict,
    timeout: int = 30
) -> Any:

    started_at = time.perf_counter()
    request_id = os.getenv("DAI_REQUEST_ID", "unknown")
    step = "detail" if url == DVC_DETAIL_API else "search"

    parsed = urllib.parse.urlparse(
        url
    )

    if parsed.scheme != "https":
        raise ValueError(
            "DVC request chỉ được phép dùng HTTPS."
        )

    if parsed.hostname != ALLOWED_HOST:
        raise ValueError(
            f"Host không được phép: {parsed.hostname}"
        )

    data = json.dumps(
        payload,
        ensure_ascii=False
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0.0.0 Safari/537.36"
            ),
            "Origin": DVC_BASE_URL,
            "Referer": DVC_BASE_URL + "/",
        },
        method="POST",
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=timeout
        ) as response:

            raw = response.read()

            text = raw.decode(
                "utf-8",
                errors="replace"
            )

            print(
                "[MCP HTTP]",
                "request_id=", request_id,
                "source=dichvucong",
                "step=", step,
                "status_code=", response.status,
                "attempt=1", "retry_count=0",
                "response_chars=", len(text),
                "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
                file=sys.stderr,
                flush=True,
            )

            print(
                f"DVC CONTENT TYPE: "
                f"{response.headers.get('Content-Type')}",
                file=sys.stderr
            )

            print(
                f"DVC RESPONSE LENGTH: "
                f"{len(text)}",
                file=sys.stderr
            )


            if not text.strip():

                raise ValueError(
                    "DVC trả về response rỗng."
                )

            try:

                return json.loads(
                    text
                )

            except json.JSONDecodeError:

                print(
                    "DVC RESPONSE KHONG PHAI JSON",
                    file=sys.stderr
                )

                raise

    except urllib.error.HTTPError as e:

        error_body = e.read().decode(
            "utf-8",
            errors="replace"
        )

        print(
            "[MCP HTTP]",
            "request_id=", request_id,
            "source=dichvucong", "step=", step,
            "status_code=", e.code,
            "attempt=1", "retry_count=0",
            "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
            file=sys.stderr,
            flush=True,
        )


        raise

    except Exception as e:

        print(
            "[MCP HTTP]",
            "request_id=", request_id,
            "source=dichvucong", "step=", step,
            "status_code=unavailable",
            "exception_type=", type(e).__name__,
            "attempt=1", "retry_count=0",
            "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
            file=sys.stderr,
            flush=True,
        )

        raise


# ============================================================
# EXTRACT TEXT
# ============================================================

def extract_text_from_object(
    obj: Any,
    max_length: int = 30000
) -> str:

    if obj is None:
        return ""

    if isinstance(
        obj,
        str
    ):
        return obj[:max_length]

    if isinstance(
        obj,
        (int, float, bool)
    ):
        return str(obj)

    if isinstance(
        obj,
        list
    ):

        parts = []

        for item in obj:

            text = extract_text_from_object(
                item,
                max_length
            )

            if text:
                parts.append(text)

        return "\n".join(
            parts
        )[:max_length]

    if isinstance(
        obj,
        dict
    ):

        parts = []

        for key, value in obj.items():

            if value is None:
                continue

            text = extract_text_from_object(
                value,
                max_length
            )

            if text:

                parts.append(
                    f"{key}: {text}"
                )

            if sum(
                len(x)
                for x in parts
            ) >= max_length:
                break

        return "\n".join(
            parts
        )[:max_length]

    return str(
        obj
    )[:max_length]


# ============================================================
# GET NAME
# ============================================================

def get_procedure_name(
    row: dict
) -> str:

    fields = [
        "name",
        "procedureName",
        "procedure_name",
        "title",
        "formalityName",
        "formality_name",
    ]

    for field in fields:

        value = row.get(
            field
        )

        if value:
            return str(
                value
            ).strip()

    return ""


# ============================================================
# GET ID
# ============================================================

def get_procedure_id(
    row: dict
):

    fields = [
        "id",
        "formalityId",
        "procedureId",
        "uuid",
    ]

    for field in fields:

        value = row.get(
            field
        )

        if value:
            return value

    return None


# ============================================================
# FIND BEST PROCEDURE
# ============================================================

def find_best_procedure(rows, keyword):
    selected = select_best_procedure(rows, keyword)
    if selected:
        print("DVC MATCH: relevant official search result", file=sys.stderr)
    return selected


# ============================================================
# BUILD DETAIL
# ============================================================

def build_detail_text(
    detail
) -> str:

    if not detail:
        return ""

    if isinstance(
        detail,
        dict
    ):

        if detail.get(
            "data"
        ) is not None:

            detail = detail.get(
                "data"
            )

    return extract_text_from_object(
        detail,
        30000
    )


# ============================================================
# LOCAL QDRANT
# ============================================================

@mcp.tool()
@guarded_tool("search_legal_documents")
def search_legal_documents(
    query: str,
    limit: int = 5
) -> str:

    query = str(
        query or ""
    ).strip()

    if not query:
        return "Query rỗng."

    try:

        vector = get_embedding_model().encode(
            query
        ).tolist()

        result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        points = result.points

        if not points:

            return (
                "Không tìm thấy dữ liệu "
                "pháp luật phù hợp."
            )

        output = []

        for index, point in enumerate(
            points,
            start=1
        ):

            payload = (
                point.payload
                or {}
            )

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

        return "\n\n".join(
            output
        )

    except Exception as e:

        print(
            f"LOCAL RAG ERROR: {repr(e)}",
            file=sys.stderr
        )

        return (
            "Lỗi khi tìm kiếm Qdrant: "
            + str(e)
        )



# ============================================================
# DVC MULTI-QUERY + LOCAL INDEX FALLBACK
# ============================================================

def build_dvc_search_queries(keyword: str) -> list[str]:
    query = re.sub(r"\s+", " ", str(keyword or "")).strip()
    return [query] if query else []


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
        "stage=search_parse", "status=completed",
        "raw_count=", len(rows),
        "elapsed_ms=", round((time.perf_counter() - started_at) * 1000, 2),
        file=sys.stderr,
    )
    return rows


def search_local_dvc_index(query: str, limit: int = 5) -> list:
    """Resolve relevant official IDs from the local index, then fetch DVC detail."""
    dataset = get_dvc_dataset()
    ranked = rank_local_index(dataset, query, limit=limit)
    print(f"DVC LOCAL INDEX MATCH COUNT: {len(ranked)}", file=sys.stderr)
    return ranked


def get_dvc_detail_by_id(procedure_id: str) -> Any:
    return post_json(
        DVC_DETAIL_API,
        {"id": procedure_id}
    )


# ============================================================
# SEARCH PROCEDURE
# ============================================================

@mcp.tool()
@guarded_tool("search_procedure")
def search_procedure(
    keyword: str,
    limit: int = 5
) -> str:

    return search_legal_documents.__wrapped__(
        keyword,
        limit
    )


# ============================================================
# DỊCH VỤ CÔNG
# ============================================================

@mcp.tool()
@guarded_tool("search_dichvucong")
def search_dichvucong(keyword: str) -> str:
    tool_started_at = time.perf_counter()
    original_question = str(keyword or "").strip()
    if not is_procedure_question(original_question):
        return "DVC_NO_RELEVANT_RESULT: query is not an administrative procedure request."

    dvc_keyword = extract_dvc_keyword(original_question)
    queries = build_dvc_search_queries(dvc_keyword)
    if not queries:
        return "DVC_NO_RELEVANT_RESULT: query contains no searchable procedure terms."

    # An upstream failure is propagated by guarded_tool as an MCP error. It must
    # never be treated as an empty search and converted into a local-index hit.
    rows = []
    for query in queries:
        rows.extend(search_dvc_rows(query))
    raw_search_count = len(rows)
    unique_rows = {}
    for row in rows:
        procedure_id = get_procedure_id(row)
        identity = str(procedure_id) if procedure_id else repr(sorted(row.items()))
        unique_rows[identity] = row

    selected = find_best_procedure(list(unique_rows.values()), dvc_keyword)
    selected_source = "official_search"
    if selected is None:
        fallback_started_at = time.perf_counter()
        fallback_candidates = search_local_dvc_index(dvc_keyword, limit=5)
        print(
            "[MCP DVC]",
            "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
            "stage=local_id_resolution", "status=completed" if fallback_candidates else "no_match",
            "candidate_count=", len(fallback_candidates),
            "elapsed_ms=", round((time.perf_counter() - fallback_started_at) * 1000, 2),
            file=sys.stderr,
        )
        if not fallback_candidates:
            return "DVC_NO_RELEVANT_RESULT: no sufficiently relevant official procedure matched."
        selected = fallback_candidates[0]["row"]
        selected_source = "local_index_id_resolver"

    procedure_id = get_procedure_id(selected)
    procedure_name = get_procedure_name(selected)
    if not procedure_id or not procedure_name:
        raise DVCUpstreamError("Selected DVC procedure is missing its official ID or name")
    procedure_code = str(selected.get("code") or selected.get("codeNotation") or "")

    detail_started_at = time.perf_counter()
    detail_response = get_dvc_detail_by_id(str(procedure_id))
    validate_dvc_detail_response(detail_response, str(procedure_id))

    detail_text = build_detail_text(detail_response)
    if not detail_text.strip():
        raise DVCUpstreamError("DVC detail response contains no procedure detail")

    print(f"DVC MATCH SOURCE: {selected_source}", file=sys.stderr)
    print(
        "[MCP DVC]",
        "request_id=", os.getenv("DAI_REQUEST_ID", "unknown"),
        "stage=detail_mapping", "status=completed",
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

    print(
        "============================================",
        file=sys.stderr
    )

    print(
        "DAI LEGAL MCP SERVER",
        file=sys.stderr
    )

    print(
        "Local Qdrant + Official DVC API",
        file=sys.stderr
    )

    print(
        "============================================",
        file=sys.stderr
    )

    print(
        f"Qdrant: {QDRANT_URL}",
        file=sys.stderr
    )

    print(
        f"Collection: {COLLECTION_NAME}",
        file=sys.stderr
    )

    print(
        f"DVC: {DVC_BASE_URL}",
        file=sys.stderr
    )

    print(
        "Starting MCP stdio server...",
        file=sys.stderr
    )

    mcp.run(
        transport="stdio"
    )
