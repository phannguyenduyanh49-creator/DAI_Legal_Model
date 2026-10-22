import sys
import re
import json
import unicodedata
import urllib.request
import urllib.error
import urllib.parse

from typing import Any

from datasets import load_from_disk

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from mcp.server import MCPServer

from backend.mcp.runtime import guarded_tool


# ============================================================
# CONFIG
# ============================================================

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_docs"

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"

# DVC local index: chỉ dùng để RESOLVE formality_id khi endpoint top-search
# không trả đúng thủ tục. Dữ liệu cuối cùng vẫn lấy từ DVC API chính thức.
DVC_DATASET_PATH = r"D:\DAI-Legal-Model\data\dichvucong_procedures"

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

print(
    "Loading embedding model...",
    file=sys.stderr
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model OK",
    file=sys.stderr
)

qdrant = QdrantClient(
    url=QDRANT_URL
)

# Load DVC procedure index once. This is NOT the external source of truth.
# It is only a local lookup index for finding the official DVC formality ID.
dvc_dataset = None
try:
    dvc_dataset = load_from_disk(DVC_DATASET_PATH)
    print(
        f"DVC LOCAL INDEX OK: {len(dvc_dataset)} procedures",
        file=sys.stderr
    )
except Exception as e:
    print(
        f"DVC LOCAL INDEX WARNING: {repr(e)}",
        file=sys.stderr
    )

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
                f"DVC HTTP STATUS: "
                f"{response.status}",
                file=sys.stderr
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

            print(
                "DVC RESPONSE PREVIEW:",
                file=sys.stderr
            )

            print(
                text[:1000],
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
            f"DVC HTTP ERROR: {e.code}",
            file=sys.stderr
        )

        print(
            error_body[:2000],
            file=sys.stderr
        )

        raise

    except Exception as e:

        print(
            f"DVC REQUEST ERROR: {repr(e)}",
            file=sys.stderr
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

def find_best_procedure(
    rows,
    keyword
):

    if not rows:
        return None

    q = normalize_vietnamese(
        keyword
    )

    # Không loại "dang ky" ở bước exact phrase.
    # Exact phrase phải được ưu tiên trước.

    important_tokens = [
        token
        for token in q.split()
        if len(token) > 1
    ]

    print(
        f"DVC NORMALIZED KEYWORD: {q}",
        file=sys.stderr
    )

    print(
        f"DVC IMPORTANT TOKENS: "
        f"{important_tokens}",
        file=sys.stderr
    )

    candidates = []

    important_phrases = [
        "tam tru",
        "tam vang",
        "khai sinh",
        "ket hon",
        "ly lich tu phap",
        "ho chieu",
        "giay phep xay dung",
        "dang ky xe",
        "chung thuc",
        "cap lai",
        "gia han",
    ]

    for row in rows:

        if not isinstance(
            row,
            dict
        ):
            continue

        name = get_procedure_name(
            row
        )

        if not name:
            continue

        name_norm = normalize_vietnamese(
            name
        )

        # ====================================================
        # EXACT
        # ====================================================

        if q in name_norm:

            candidates.append({
                "row": row,
                "name": name,
                "score": 1000,
                "reason": "EXACT PHRASE",
            })

            continue

        # ====================================================
        # IMPORTANT PHRASE
        # ====================================================

        matched_phrase = None

        for phrase in important_phrases:

            if (
                phrase in q
                and phrase in name_norm
            ):

                matched_phrase = phrase
                break

        if matched_phrase:

            candidates.append({
                "row": row,
                "name": name,
                "score": 950,
                "reason": (
                    "IMPORTANT PHRASE: "
                    + matched_phrase
                ),
            })

            continue

        # ====================================================
        # TOKEN MATCH
        # ====================================================

        name_tokens = set(
            name_norm.split()
        )

        matched = [
            token
            for token in important_tokens
            if token in name_tokens
        ]

        if len(
            important_tokens
        ) >= 2:

            if len(matched) < 2:
                continue

            overlap = (
                len(matched)
                /
                len(important_tokens)
            )

            if overlap < 0.75:
                continue

            score = overlap * 100

        else:

            if not matched:
                continue

            score = 50

        candidates.append({
            "row": row,
            "name": name,
            "score": score,
            "reason": (
                f"TOKEN MATCH {matched}"
            ),
        })

    if not candidates:

        print(
            "DVC: KHONG CO THU TUC PHU HOP",
            file=sys.stderr
        )

        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print(
        "DVC RANKED RESULTS:",
        file=sys.stderr
    )

    for item in candidates[:10]:

        print(
            f"  [{item['score']:.2f}] "
            f"{item['reason']} -> "
            f"{item['name']}",
            file=sys.stderr
        )

    best = candidates[0]

    print(
        f"DVC BEST: {best['name']} "
        f"({best['reason']}, "
        f"score={best['score']:.2f})",
        file=sys.stderr
    )

    return best["row"]


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

        vector = embedding_model.encode(
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
    """Tạo các biến thể tìm kiếm theo cách người dùng hỏi."""
    original = str(keyword or "").strip()
    normalized = normalize_vietnamese(original)
    queries = []

    def add(q: str):
        q = re.sub(r"\s+", " ", str(q or "")).strip()
        if q and q not in queries:
            queries.append(q)

    add(original)

    if "ho tro" in normalized and "hoc" in normalized:
        add("hỗ trợ chi phí học tập")
        add("hỗ trợ học phí")
        add("chi phí học tập")
        add("hỗ trợ học sinh sinh viên")

    if "hoc dai hoc" in normalized:
        add("giải quyết chế độ hỗ trợ để theo học đến trình độ đại học")
        add("hỗ trợ học tập đại học")
        add("hỗ trợ chi phí học tập cho sinh viên")
        add("hỗ trợ học tập sinh viên")

    return queries


def extract_dvc_rows(response: Any) -> list:
    rows = []

    if isinstance(response, dict):
        data = response.get("data")
        if isinstance(data, dict):
            rows = (
                data.get("rows")
                or data.get("items")
                or data.get("content")
                or []
            )
        elif isinstance(data, list):
            rows = data

        if not rows:
            rows = response.get("rows") or response.get("items") or []

    elif isinstance(response, list):
        rows = response

    return rows


def search_dvc_rows(query: str) -> list:
    """Gọi endpoint top-search hiện có của DVC. Không dùng lastId vì API đã
    được kiểm chứng là trả lại cùng 10 rows khi đổi lastId."""
    payload = {
        "limit": 10,
        "lastId": "",
        "q": query,
        "categoryId": "",
        "departmentCode": "",
        "formalityType": "STANDARD",
    }

    try:
        response = post_json(DVC_SEARCH_API, payload)
    except Exception as e:
        print(
            f"DVC SEARCH ERROR [{query}]: {repr(e)}",
            file=sys.stderr
        )
        return []

    rows = extract_dvc_rows(response)
    print(
        f"DVC TOP-SEARCH [{query}] -> {len(rows)} rows",
        file=sys.stderr
    )
    return rows


def _row_text(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    fields = [
        row.get("procedure_name"),
        row.get("procedureName"),
        row.get("name"),
        row.get("keywords"),
        row.get("description"),
        row.get("content_text"),
    ]
    return " ".join(str(x) for x in fields if x)


def search_local_dvc_index(query: str, limit: int = 5) -> list:
    """Resolve procedure candidates from the DVC dataset index.

    This is only an ID resolver. After selecting a candidate, the MCP MUST
    fetch its detail from the official DVC API before returning context.
    """
    if dvc_dataset is None:
        return []

    q = normalize_vietnamese(query)
    q_tokens = {t for t in q.split() if len(t) >= 2}

    if not q_tokens:
        return []

    scored = []

    try:
        for row in dvc_dataset:
            name = str(row.get("procedure_name") or "").strip()
            if not name:
                continue

            text = normalize_vietnamese(_row_text(row))
            name_norm = normalize_vietnamese(name)
            name_tokens = set(name_norm.split())
            matched = q_tokens & name_tokens

            score = 0.0

            # Exact phrase in procedure name.
            if q in name_norm:
                score += 1000.0

            # Prefer matching procedure title over description/content.
            score += len(matched) * 40.0

            # Coverage of query terms.
            score += (len(matched) / max(len(q_tokens), 1)) * 100.0

            # Bonus for the key concept combination.
            if {"ho", "tro"}.issubset(name_tokens):
                score += 25.0
            if "hoc" in q_tokens and "hoc" in name_tokens:
                score += 35.0
            if "dai" in q_tokens and "dai" in name_tokens:
                score += 35.0

            # Do not consider a weak candidate.
            if len(matched) < 2 and q not in name_norm:
                continue

            # Retrieve ID from the canonical dataset field.
            formality_id = (
                row.get("formality_id")
                or row.get("id")
                or row.get("formalityId")
            )
            if not formality_id:
                continue

            scored.append({
                "id": str(formality_id),
                "name": name,
                "score": score,
                "row": row,
            })

    except Exception as e:
        print(
            f"DVC LOCAL INDEX ERROR: {repr(e)}",
            file=sys.stderr
        )
        return []

    scored.sort(key=lambda x: x["score"], reverse=True)

    print(
        f"DVC LOCAL INDEX MATCHES [{query}]: {len(scored)}",
        file=sys.stderr
    )
    for item in scored[:5]:
        print(
            f"  [{item['score']:.1f}] {item['name']} | ID={item['id']}",
            file=sys.stderr
        )

    return scored[:limit]


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

    # Internal delegation must not re-authorize as a second public tool: the
    # caller is authorized as search_procedure, and this invokes its implementation.
    return search_legal_documents.__wrapped__(
        keyword,
        limit
    )


# ============================================================
# DỊCH VỤ CÔNG
# ============================================================

@mcp.tool()
@guarded_tool("search_dichvucong")
def search_dichvucong(
    keyword: str
) -> str:

    original_question = str(keyword or "").strip()

    print(
        f"DVC ORIGINAL QUESTION: {original_question}",
        file=sys.stderr
    )

    if not is_procedure_question(original_question):
        return (
            "Câu hỏi không được xác định là yêu cầu "
            "tra cứu thủ tục hành chính."
        )

    dvc_keyword = extract_dvc_keyword(original_question)
    print(
        f"DVC KEYWORD: {normalize_vietnamese(dvc_keyword)}",
        file=sys.stderr
    )

    if not dvc_keyword:
        return "Không xác định được từ khóa thủ tục để tra cứu."

    # --------------------------------------------------------
    # 1. Try official DVC top-search endpoint.
    # --------------------------------------------------------
    queries = build_dvc_search_queries(dvc_keyword)
    all_rows = []

    for q in queries:
        rows = search_dvc_rows(q)
        for row in rows:
            if isinstance(row, dict):
                row_copy = dict(row)
                row_copy["_query"] = q
                all_rows.append(row_copy)

    # De-duplicate official search rows by ID.
    unique_rows = {}
    for row in all_rows:
        rid = get_procedure_id(row)
        if rid:
            unique_rows[str(rid)] = row

    rows = list(unique_rows.values())

    print(
        f"DVC UNIQUE TOP-SEARCH RESULTS: {len(rows)}",
        file=sys.stderr
    )

    best = find_best_procedure(rows, dvc_keyword)

    # --------------------------------------------------------
    # 2. If top-search ignores q, resolve the ID using the DVC
    #    procedure index, then fetch the authoritative detail from
    #    DVC. This is NOT a Qdrant wrapper: Qdrant is not used here.
    # --------------------------------------------------------
    if not best:
        print(
            "DVC TOP-SEARCH KHONG TIM THAY -> LOCAL DVC INDEX FALLBACK",
            file=sys.stderr
        )

        fallback_candidates = []
        for q in queries:
            fallback_candidates.extend(
                search_local_dvc_index(q, limit=5)
            )

        # De-duplicate and keep best score.
        candidate_map = {}
        for item in fallback_candidates:
            rid = item["id"]
            if rid not in candidate_map or item["score"] > candidate_map[rid]["score"]:
                candidate_map[rid] = item

        fallback_candidates = sorted(
            candidate_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        if not fallback_candidates:
            return (
                "Không tìm thấy thủ tục phù hợp trong chỉ mục DVC. "
                "Không gọi nguồn khác ngoài Cổng Dịch vụ công Quốc gia."
            )

        selected = fallback_candidates[0]
        procedure_id = selected["id"]
        procedure_name = selected["name"]
        procedure_code = str(
            selected["row"].get("code") or ""
        )

        print(
            f"DVC FALLBACK SELECTED: {procedure_name}",
            file=sys.stderr
        )
        print(
            f"DVC FALLBACK ID: {procedure_id}",
            file=sys.stderr
        )

        try:
            detail_response = get_dvc_detail_by_id(procedure_id)
        except Exception as e:
            print(
                f"DVC DETAIL ERROR: {repr(e)}",
                file=sys.stderr
            )
            return (
                f"Đã xác định thủ tục trên chỉ mục DVC: {procedure_name}, "
                "nhưng không lấy được dữ liệu chi tiết từ Cổng Dịch vụ công Quốc gia."
            )

    else:
        procedure_id = get_procedure_id(best)
        procedure_name = get_procedure_name(best)
        procedure_code = str(
            best.get("code") or best.get("codeNotation") or ""
        )

        print(
            f"DVC SELECTED: {procedure_name}",
            file=sys.stderr
        )
        print(
            f"DVC ID: {procedure_id}",
            file=sys.stderr
        )

        if not procedure_id:
            return (
                "Tìm thấy thủ tục nhưng không có ID để lấy chi tiết.\n\n"
                + extract_text_from_object(best, 20000)
            )

        try:
            detail_response = get_dvc_detail_by_id(procedure_id)
        except Exception as e:
            print(
                f"DVC DETAIL ERROR: {repr(e)}",
                file=sys.stderr
            )
            return (
                f"Tìm thấy thủ tục: {procedure_name}\n\n"
                "Nhưng không lấy được dữ liệu chi tiết."
            )

    detail_text = build_detail_text(detail_response)

    print(
        f"DVC DETAIL OK: {len(detail_text)} chars",
        file=sys.stderr
    )

    if not detail_text:
        return (
            f"Tìm thấy thủ tục: {procedure_name}, nhưng DVC không trả dữ liệu chi tiết."
        )

    return f"""
=== CỔNG DỊCH VỤ CÔNG QUỐC GIA ===

Tên thủ tục:
{procedure_name}

Mã thủ tục:
{procedure_code}

ID:
{procedure_id}

Nguồn chính thức:
{DVC_BASE_URL}

DỮ LIỆU LẤY TRỰC TIẾP TỪ DVC API:

{detail_text}

============================================================

CHỈ DÙNG DỮ LIỆU TRÊN ĐỂ TRẢ LỜI.

Không tự bịa:
- giấy tờ
- phí/lệ phí
- thời hạn
- cơ quan thực hiện
- điều kiện

Nếu dữ liệu không có thông tin cần thiết, phải nói rõ dữ liệu chưa đủ.

============================================================
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