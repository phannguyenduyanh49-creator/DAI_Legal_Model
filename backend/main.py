from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from backend.catalog.service import CatalogService
from backend.mcp.adapter import (
    mcp_results_to_evidence
)

from backend.prompt.assembler import PromptAssembler

from backend.catalog.schemas import CatalogQuery

from backend.evidence.normalizer import (
    procedure_to_evidence,
    catalog_record_to_evidence,
)

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from backend.rag.adapter import rag_results_to_evidence
from backend.evidence.builder import EvidenceBuilder
import httpx
import json
import asyncio
import os
import sys
import re
import unicodedata
import uuid
import sqlite3
from datetime import datetime, timezone


catalog_service = CatalogService()
evidence_builder = EvidenceBuilder(
    min_score=0.3,
    max_evidences=10,
)

prompt_assembler = PromptAssembler(
    max_evidence_chars=12000
)
# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="DAI Legal Core API"
)

# =========================================================
# SESSION / COOKIE
# =========================================================
# Khong can login. Moi browser/user duoc cap mot session_id rieng.
# Cookie chi luu session_id, khong luu toan bo lich su chat.
SESSION_COOKIE_NAME = "session_id"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 ngay
SESSION_COOKIE_SAMESITE = "lax"
SESSION_COOKIE_SECURE = False  # True khi deploy HTTPS production



# =========================================================
# PROJECT BASE DIRECTORY
# =========================================================

BASE_DIR = r"D:\Prj\DAI-Legal-Model"


# =========================================================
# CHAT HISTORY DATABASE - SQLITE
# =========================================================
# Giai đoạn hiện tại dùng SQLite để test nhanh.
# Khi Docker production sẽ chuyển sang PostgreSQL.
CHAT_DB_DIR = os.path.join(BASE_DIR, "data")
CHAT_DB_PATH = os.path.join(CHAT_DB_DIR, "chat_history.db")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_chat_database():
    os.makedirs(CHAT_DB_DIR, exist_ok=True)

    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages(session_id)
        """)

        conn.commit()

    print("CHAT DATABASE READY:", CHAT_DB_PATH)


def ensure_session(session_id: str):
    now = utc_now()

    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, created_at, last_active)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET last_active = excluded.last_active
        """, (session_id, now, now))

        conn.commit()


def save_message(session_id: str, role: str, content: str):
    content = str(content or "").strip()

    if not content:
        return

    now = utc_now()

    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute("""
            INSERT INTO messages
            (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, now))

        conn.execute("""
            UPDATE sessions
            SET last_active = ?
            WHERE session_id = ?
        """, (now, session_id))

        conn.commit()


def get_pending_external_question(session_id: str) -> str | None:
    """
    Nếu tin nhắn cuối cùng của AI là yêu cầu xin phép tra cứu
    dữ liệu bên ngoài, lấy lại câu hỏi pháp lý ngay trước đó.
    """

    permission_marker = (
        "Bạn có muốn mình sử dụng "
        "dữ liệu bên ngoài từ "
        "Cổng Dịch vụ công Quốc gia"
    )

    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT 2
        """, (session_id,)).fetchall()

    if len(rows) < 2:
        return None

    latest = rows[0]
    previous = rows[1]

    if (
        latest["role"] == "assistant"
        and permission_marker in str(latest["content"] or "")
        and previous["role"] == "user"
    ):
        return str(previous["content"] or "").strip() or None

    return None


def get_chat_history(session_id: str, limit: int = 20) -> list[dict]:
    limit = max(1, min(int(limit), 100))

    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT id, session_id, role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()

    rows = list(reversed(rows))

    return [dict(row) for row in rows]


def clear_chat_history(session_id: str):
    now = utc_now()

    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,)
        )

        conn.execute("""
            UPDATE sessions
            SET last_active = ?
            WHERE session_id = ?
        """, (now, session_id))

        conn.commit()

def get_or_create_session_id(request: Request) -> tuple[str, bool]:
    """Lay session_id tu cookie; neu chua co thi tao UUID moi."""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)

    if session_id:
        session_id = session_id.strip()

    if session_id:
        return session_id, False

    return str(uuid.uuid4()), True




# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = r"D:\Prj\DAI-Legal-Model"


MCP_SERVER_PATH = os.path.join(
    BASE_DIR,
    "mcp_server.py"
)

MODEL_API_URL = (
    "http://localhost:8001/generate"
)

QDRANT_URL = (
    "http://localhost:6333"
)


# =========================================================
# INITIALIZE CHAT DATABASE
# =========================================================

init_chat_database()


# =========================================================
# BGE-M3 + QDRANT
# =========================================================

QDRANT_COLLECTION = (
    "legal_docs_bge_m3"
)

EMBEDDING_MODEL = (
    "BAAI/bge-m3"
)


# =========================================================
# RERANKER
# =========================================================

RERANKER_MODEL = (
    "BAAI/bge-reranker-v2-m3"
)

# Điểm BGE-M3 tối thiểu để được xét
RAG_SCORE_THRESHOLD = 0.45

# Số document lấy từ Qdrant trước khi rerank
MAX_LOCAL_RESULTS = 10

# Chỉ giữ document có điểm reranker đủ cao
RERANK_SCORE_THRESHOLD = 0.50

# Số document tối đa đưa cho Qwen
MAX_RERANK_RESULTS = 3

# Khoảng cách điểm cho phép giữa kết quả đầu và các kết quả sau.
# Nếu kết quả đầu vượt kết quả thứ hai quá margin này, chỉ giữ kết quả đầu.
RERANK_MARGIN_THRESHOLD = 0.15

MAX_CONTEXT_CHARS = 5000


# =========================================================
# QDRANT
# =========================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL
)


# =========================================================
# LOCAL PROCEDURE INDEX
# =========================================================

LOCAL_PROCEDURE_INDEX = []
LOCAL_PROCEDURE_INDEX_LOADED = False

def load_local_procedure_index():
    global LOCAL_PROCEDURE_INDEX, LOCAL_PROCEDURE_INDEX_LOADED
    if LOCAL_PROCEDURE_INDEX_LOADED:
        return LOCAL_PROCEDURE_INDEX
    try:
        items = []
        offset = None
        while True:
            points, next_offset = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                name = str(payload.get("thu_tuc", payload.get("procedure_name", "")) or "").strip()
                if name:
                    items.append({"id": point.id, "procedure_name": name, "payload": payload})
            if next_offset is None:
                break
            offset = next_offset
        LOCAL_PROCEDURE_INDEX = items
        LOCAL_PROCEDURE_INDEX_LOADED = True
        print("LOCAL PROCEDURE INDEX LOADED:", len(items))
    except Exception as exc:
        print("LOCAL PROCEDURE INDEX ERROR:", exc)
    return LOCAL_PROCEDURE_INDEX

def find_exact_local_procedures(question: str):
    question_normalized = normalize_vietnamese(question)
    if not question_normalized:
        return []
    matches = []
    for item in load_local_procedure_index():
        name = item["procedure_name"]
        normalized = normalize_vietnamese(name)
        without_prefix = re.sub(r"^thu tuc\s+", "", normalized).strip()
        length = 0
        if normalized and normalized in question_normalized:
            length = len(normalized)
        elif without_prefix and without_prefix in question_normalized:
            length = len(without_prefix)
        if length:
            matches.append((length, item))
    matches.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in matches[:5]]



# =========================================================
# BGE-M3 EMBEDDING
# =========================================================

print(
    "Loading BGE-M3 embedding model..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cuda"
)

embedding_model.max_seq_length = 1024

print(
    "BGE-M3 embedding model loaded."
)

print(
    "Embedding dimension:",
    embedding_model.get_sentence_embedding_dimension()
)


# =========================================================
# BGE RERANKER
# =========================================================

print(
    "Loading BGE Reranker..."
)

reranker = FlagReranker(
    RERANKER_MODEL,
    use_fp16=True
)

print(
    "BGE Reranker loaded."
)


# =========================================================
# HOME
# =========================================================

@app.get("/")
async def root():

    return {
        "success": True,
        "message": "DAI Legal Core API đang chạy",
        "architecture": (
            "Frontend -> Core API -> "
            "BGE-M3 -> Qdrant Top-K -> "
            "BGE Reranker -> Local RAG -> "
            "Permission -> MCP -> Qwen"
        ),
        "mcp_server": MCP_SERVER_PATH,
        "model_api": MODEL_API_URL,
        "qdrant": QDRANT_URL,
        "collection": QDRANT_COLLECTION,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL
    }


# =========================================================
# NORMALIZE VIETNAMESE
# =========================================================

def normalize_vietnamese(
    text: str
) -> str:

    text = str(
        text
    ).lower().strip()

    text = unicodedata.normalize(
        "NFD",
        text
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char)
        != "Mn"
    )

    text = text.replace(
        "đ",
        "d"
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SYNONYM / ABBREVIATION MAPPING
# =========================================================
# Mo rong tu viet tat/synonym cho retrieval.
# Cau hoi goc cua nguoi dung van duoc giu nguyen.

SYNONYM_MAPPING = {
    "cccd": "căn cước công dân",
    "can cuoc": "căn cước công dân",
    "cmnd": "chứng minh nhân dân",
    "cmt": "chứng minh nhân dân",
    "cmtnd": "chứng minh nhân dân",

    "dkkh": "đăng ký kết hôn",
    "đkkh": "đăng ký kết hôn",

    "dkks": "đăng ký khai sinh",
    "đkks": "đăng ký khai sinh",

    "dkt": "đăng ký tạm trú",
    "dktt": "đăng ký tạm trú",
    "đktt": "đăng ký thường trú",

    "gplx": "giấy phép lái xe",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "mst": "mã số thuế",
    "tthc": "thủ tục hành chính",
    "dvc": "dịch vụ công",
    "hkd": "hộ kinh doanh",
    "gcn": "giấy chứng nhận",
    "gcnqsdd": "giấy chứng nhận quyền sử dụng đất",
    "sdd": "sử dụng đất",
    "vneid": "ứng dụng định danh điện tử VNeID",
}


def expand_synonyms_for_retrieval(text: str) -> str:
    """Mo rong tu viet tat nhung giu nguyen dau tieng Viet cho BGE-M3."""
    original = str(text or "").strip()
    if not original:
        return original

    expanded = original
    mappings = sorted(
        SYNONYM_MAPPING.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for abbreviation, full_form in mappings:
        pattern = rf"(?<!\w){re.escape(abbreviation)}(?!\w)"
        expanded = re.sub(
            pattern,
            full_form,
            expanded,
            flags=re.IGNORECASE
        )

    if expanded != original:
        print("SYNONYM MAPPING:", original, "->", expanded)

    return expanded


# =========================================================
# STOPWORDS
# =========================================================

RAG_STOPWORDS = {
    "thu",
    "tuc",
    "thuc",
    "hien",
    "can",
    "nhung",
    "giay",
    "to",
    "gi",
    "nao",
    "bao",
    "lau",
    "o",
    "dau",
    "the",
    "co",
    "khong",
    "muc",
    "phi",
    "mat",
    "thoi",
    "han",
    "cho",
    "toi",
    "xin",
    "vui",
    "long",
    "hoi",
    "la",
    "mot",
    "loai",
    "cua",
    "nguoi",
    "duoc",
    "nhu",
    "theo",
    "ve",
    "voi",
    "trong",
    "tai"
}


# =========================================================
# KEYWORDS
# =========================================================

def extract_keywords(
    text: str
) -> set[str]:

    words = normalize_vietnamese(
        text
    ).split()

    return {
        word
        for word in words
        if len(word) >= 2
        and word not in RAG_STOPWORDS
    }


# =========================================================
# CHECK PROCEDURE RELEVANCE
# =========================================================

def is_relevant_procedure(
    question: str,
    procedure_name: str,
    score: float
) -> bool:

    question_normalized = (
        normalize_vietnamese(
            question
        )
    )

    procedure_normalized = (
        normalize_vietnamese(
            procedure_name
        )
    )

    if not procedure_normalized:
        return False

    # -----------------------------------------------------
    # EXACT NAME
    # -----------------------------------------------------

    if procedure_normalized in question_normalized:

        print(
            "RAG EXACT MATCH:",
            procedure_name
        )

        return True

    # -----------------------------------------------------
    # KEYWORDS
    # -----------------------------------------------------

    question_keywords = (
        extract_keywords(question)
    )

    procedure_keywords = (
        extract_keywords(procedure_name)
    )

    if not question_keywords:
        return False

    if not procedure_keywords:
        return False

    matched = (
        question_keywords
        &
        procedure_keywords
    )

    overlap = (
        len(matched)
        /
        max(
            len(procedure_keywords),
            1
        )
    )

    print(
        "RAG RELEVANCE:",
        procedure_name,
        "| score=",
        round(score, 4),
        "| matched=",
        matched,
        "| overlap=",
        round(overlap, 2)
    )

    # -----------------------------------------------------
    # STRONG KEYWORD MATCH
    # -----------------------------------------------------

    if overlap >= 0.60:
        return True

    # -----------------------------------------------------
    # SHORT PROCEDURE NAME
    # -----------------------------------------------------

    if (
        len(procedure_keywords) <= 2
        and len(matched) >= 1
        and score >= 0.65
    ):
        return True

    # -----------------------------------------------------
    # VERY HIGH SCORE
    # -----------------------------------------------------

    if score >= 0.78:
        return True

    return False


# =========================================================
# RERANK LOCAL RESULTS
# =========================================================

async def rerank_local_results(
    question: str,
    contexts: list[dict]
):

    if not contexts:
        return []

    print(
        "\n=============================="
    )

    print(
        "RERANKER INPUT:",
        len(contexts),
        "documents"
    )



    pairs = []

    for item in contexts:

        document_text = (
            f"Tên thủ tục: "
            f"{item['procedure_name']}\n"
            f"{item['text']}"
        )

        pairs.append(
            [
                question,
                document_text
            ]
        )

    try:

        # -------------------------------------------------
        # RUN RERANKER
        # -------------------------------------------------

        scores = await asyncio.to_thread(
            reranker.compute_score,
            pairs,
            normalize=True
        )

        # Trường hợp chỉ có 1 document
        if isinstance(
            scores,
            (float, int)
        ):
            scores = [float(scores)]

        # -------------------------------------------------
        # ATTACH SCORE
        # -------------------------------------------------

        reranked = []

        for item, score in zip(
            contexts,
            scores
        ):

            item_copy = dict(item)

            item_copy[
                "rerank_score"
            ] = float(score)

            reranked.append(
                item_copy
            )

        # -------------------------------------------------
        # SORT BY RERANK SCORE
        # -------------------------------------------------

        reranked.sort(
            key=lambda x:
                x["rerank_score"],
            reverse=True
        )

        print(
            "\n=============================="
        )

        print(
            "RERANK RESULTS:"
        )

        for i, item in enumerate(
            reranked,
            start=1
        ):

            print(
                f"#{i} "
                f"{item['procedure_name']} | "
                f"BGE={item['score']:.4f} | "
                f"RERANK={item['rerank_score']:.4f}"
            )

        # -------------------------------------------------
        # FILTER
        # -------------------------------------------------

        filtered = [
            item
            for item in reranked
            if item["rerank_score"]
            >= RERANK_SCORE_THRESHOLD
        ]

        # -------------------------------------------------
        # ADAPTIVE MARGIN FILTER
        # -------------------------------------------------
        # Nếu kết quả #1 vượt kết quả #2 quá xa, chỉ giữ #1.
        # Nếu điểm các kết quả gần nhau, vẫn cho phép giữ nhiều
        # document để Qwen có đủ context.

        if filtered:

            top_score = filtered[0]["rerank_score"]

            adaptive_filtered = [
                filtered[0]
            ]

            for item in filtered[1:]:

                score_gap = (
                    top_score
                    - item["rerank_score"]
                )

                if score_gap <= RERANK_MARGIN_THRESHOLD:

                    adaptive_filtered.append(
                        item
                    )

                    print(
                        "→ KEEP BY MARGIN:",
                        item["procedure_name"],
                        "| gap=",
                        round(score_gap, 4)
                    )

                else:

                    print(
                        "→ REJECT BY MARGIN:",
                        item["procedure_name"],
                        "| gap=",
                        round(score_gap, 4)
                    )

            filtered = adaptive_filtered[:MAX_RERANK_RESULTS]

        # -------------------------------------------------
        # TOP N
        # -------------------------------------------------

        filtered = filtered[
            :MAX_RERANK_RESULTS
        ]

        print(
            "\nRERANK FILTERED RESULTS:",
            len(filtered)
        )

        for item in filtered:

            print(
                "→ ACCEPT:",
                item["procedure_name"],
                "| rerank=",
                round(
                    item["rerank_score"],
                    4
                )
            )

        return filtered

    except Exception as e:

        print(
            "RERANKER ERROR:",
            repr(e)
        )

        # Nếu reranker lỗi thì không làm
        # toàn bộ Local RAG chết.
        return contexts


# =========================================================
# DETECT QUESTION DATA REQUIREMENT
# =========================================================

def detect_required_fields(
    question: str
) -> list[str]:

    """
    Xác định field dữ liệu cần có để trả lời câu hỏi.
    Đây là lớp ENOUGH DATA, không thay thế RAG/Reranker.
    """

    q = normalize_vietnamese(question)

    required = []

    # -----------------------------------------------------
    # HỒ SƠ / GIẤY TỜ
    # -----------------------------------------------------
    document_patterns = [
        "giay to",
        "giay",
        "ho so",
        "thanh phan ho so",
        "can chuan bi",
        "chuan bi nhung gi",
        "can nhung gi",
        "bao gom nhung gi",
        "can nop",
        "nop nhung gi",
    ]

    if any(
        pattern in q
        for pattern in document_patterns
    ):
        required.append("profile_components")

    # -----------------------------------------------------
    # THỜI GIAN
    # -----------------------------------------------------
    time_patterns = [
        "bao lau",
        "thoi gian",
        "thoi han",
        "may ngay",
        "bao nhieu ngay",
        "giai quyet trong",
        "mat bao lau",
    ]

    if any(
        pattern in q
        for pattern in time_patterns
    ):
        required.append("processing_time")

    # -----------------------------------------------------
    # PHÍ / LỆ PHÍ
    # -----------------------------------------------------
    fee_patterns = [
        "le phi",
        "phi bao nhieu",
        "muc phi",
        "co phi",
        "mat bao nhieu",
        "ton bao nhieu",
        "bao nhieu tien",
    ]

    if any(
        pattern in q
        for pattern in fee_patterns
    ):
        required.append("fees")

    # -----------------------------------------------------
    # HÌNH THỨC NỘP
    # -----------------------------------------------------
    submission_patterns = [
        "hinh thuc nop",
        "nop o dau",
        "nop ho so o dau",
        "nop truc tuyen",
        "nop truc tiep",
        "nop online",
        "cach nop",
    ]

    if any(
        pattern in q
        for pattern in submission_patterns
    ):
        required.append("execution_methods")

    # -----------------------------------------------------
    # TRÌNH TỰ THỰC HIỆN
    # -----------------------------------------------------
    execution_patterns = [
        "trinh tu",
        "cac buoc",
        "thuc hien nhu the nao",
        "lam nhu the nao",
        "quy trinh",
    ]

    if any(
        pattern in q
        for pattern in execution_patterns
    ):
        required.append("execution_steps")

    # -----------------------------------------------------
    # ĐIỀU KIỆN
    # -----------------------------------------------------
    condition_patterns = [
        "dieu kien",
        "yeu cau",
        "dieu kien thuc hien",
        "ai duoc",
        "doi tuong nao",
    ]

    if any(
        pattern in q
        for pattern in condition_patterns
    ):
        required.append("requirements_conditions")

    # -----------------------------------------------------
    # Nếu không nhận diện được intent cụ thể,
    # yêu cầu full_text làm fallback.
    # -----------------------------------------------------
    if not required:
        required.append("full_text")

    # Loại duplicate nhưng giữ thứ tự.
    return list(
        dict.fromkeys(required)
    )


# =========================================================
# ENOUGH DATA
# =========================================================

def check_enough_data(
    question: str,
    reranked_items: list[dict]
) -> tuple[bool, list[str]]:
    """Local RAG la nguon chinh; thieu field khong tu dong day sang MCP."""
    required_fields = detect_required_fields(question)

    print("\n==============================")
    print("ENOUGH DATA CHECK")
    print("REQUIRED FIELDS:", required_fields)

    if not reranked_items:
        print("→ ENOUGH DATA: FALSE")
        print("→ REASON: khong co local document")
        return False, required_fields

    best_item = reranked_items[0]
    print("BEST PROCEDURE:", best_item.get("procedure_name", ""))

    has_content = bool(
        str(best_item.get("full_text", "")).strip()
        or str(best_item.get("text", "")).strip()
    )

    if not has_content:
        print("→ ENOUGH DATA: FALSE")
        print("→ REASON: local document khong co noi dung")
        return False, required_fields

    missing_fields = [
        field for field in required_fields
        if not str(best_item.get(field, "")).strip()
    ]

    if missing_fields:
        print("MISSING REQUESTED FIELDS:", missing_fields)
        print("→ ENOUGH DATA: TRUE (LOCAL DATA LA NGUON CHINH; KHONG FALLBACK MCP)")
        return True, missing_fields

    print("→ ENOUGH DATA: TRUE")
    return True, required_fields


# =========================================================
# LOCAL RAG
# =========================================================
async def search_local_rag(
    question: str,
    catalog_evidences: list | None = None,
):
    if catalog_evidences is None:
        catalog_evidences = []



    print(
        "\n=============================="
    )

    print(
        "LOCAL RAG QUESTION:",
        question
    )

    # =====================================================
    # SYNONYM MAPPING CHO RETRIEVAL
    # =====================================================
    # Chi mo rong query dung cho BGE-M3/Qdrant/Reranker.
    # Cau hoi goc van duoc giu nguyen cho Qwen va session.
    retrieval_question = expand_synonyms_for_retrieval(question)

    print(
        "RETRIEVAL QUESTION:",
        retrieval_question
    )

    try:

        # -------------------------------------------------
        # BGE-M3 QUERY EMBEDDING
        # -------------------------------------------------

        query_vector = (
            await asyncio.to_thread(
                embedding_model.encode,
                retrieval_question,
                normalize_embeddings=True
            )
        )

        # -------------------------------------------------
        # QDRANT TOP-K
        # -------------------------------------------------

        response = (
            await asyncio.to_thread(

                qdrant_client.query_points,

                collection_name=
                    QDRANT_COLLECTION,

                query=
                    query_vector.tolist(),

                limit=
                    MAX_LOCAL_RESULTS,

                with_payload=True
            )
        )

        results = response.points

        contexts = []

        # -------------------------------------------------
        # LOCAL PROCEDURE EXACT LOOKUP
        # -------------------------------------------------
        exact_local_items = await asyncio.to_thread(
            find_exact_local_procedures,
            retrieval_question
        )

        if exact_local_items:
            print("LOCAL EXACT PROCEDURE MATCHES:", len(exact_local_items))
            for local_item in exact_local_items:
                payload = local_item.get("payload", {}) or {}
                text = str(payload.get("text", "")).strip()
                if not text:
                    continue
                contexts.append({
                    "procedure_name": local_item["procedure_name"],
                    "text": text,
                    "score": 0.99,
                    "full_text": str(payload.get("full_text", "")).strip(),
                    "processing_time": str(payload.get("processing_time", "")).strip(),
                    "fees": str(payload.get("fees", "")).strip(),
                    "execution_methods": str(payload.get("execution_methods", "")).strip(),
                    "profile_components": str(payload.get("profile_components", "")).strip(),
                    "location": str(payload.get("location", "")).strip(),
                    "requirements_conditions": str(payload.get("requirements_conditions", "")).strip(),
                    "execution_steps": str(payload.get("execution_steps", "")).strip(),
                    "legal_basis": str(payload.get("legal_basis", "")).strip(),
                    "results": str(payload.get("results", "")).strip(),
                    "source_url": str(payload.get("source_url", "")).strip(),
                })

        # -------------------------------------------------
        # PROCESS BGE-M3 RESULTS
        # -------------------------------------------------

        for result in results:

            payload = (
                result.payload
                or {}
            )

            print(
                "PAYLOAD FIELDS:",
                "processing_time=", payload.get("processing_time", ""),
                "fees=", payload.get("fees", "")
            )

            procedure_name = str(
                payload.get(
                    "thu_tuc",
                    payload.get(
                        "procedure_name",
                        ""
                    )
                )
            ).strip()

            text = str(
                payload.get(
                    "text",
                    ""
                )
            ).strip()

            score = float(
                result.score
                or 0
            )

            print(
                f"RAG RESULT: "
                f"{procedure_name} | "
                f"score={score:.4f}"
            )

            if not text:

                print(
                    "→ REJECT: empty text"
                )

                continue

            if (
                score
                < RAG_SCORE_THRESHOLD
            ):
                exact_names = {
                    item.get("procedure_name", "")
                    for item in contexts
                    if item.get("score", 0) >= 0.99
                }
                if procedure_name not in exact_names:
                    print("→ REJECT: low score")
                    continue
                print("→ KEEP: exact local procedure despite low BGE score")

            if not is_relevant_procedure(
                retrieval_question,
                procedure_name,
                score
            ):

                print(
                    "→ REJECT: "
                    "không liên quan"
                )

                continue

            print(
                "→ ACCEPT:",
                procedure_name
            )

            contexts.append(
                {
                    "procedure_name": procedure_name,
                    "text": text,
                    "score": score,

                    # Structured fields from FULL Qdrant payload
                    "full_text": str(payload.get("full_text", "")).strip(),
                    "processing_time": str(payload.get("processing_time", "")).strip(),
                    "fees": str(payload.get("fees", "")).strip(),
                    "execution_methods": str(payload.get("execution_methods", "")).strip(),
                    "profile_components": str(payload.get("profile_components", "")).strip(),
                    "location": str(payload.get("location", "")).strip(),
                    "requirements_conditions": str(payload.get("requirements_conditions", "")).strip(),
                    "execution_steps": str(payload.get("execution_steps", "")).strip(),
                    "legal_basis": str(payload.get("legal_basis", "")).strip(),
                    "results": str(payload.get("results", "")).strip(),
                    "source_url": str(payload.get("source_url", "")).strip(),
                }
            )

        # -------------------------------------------------
        # SORT BY BGE-M3
        # -------------------------------------------------

        contexts.sort(
            key=lambda x:
                x["score"],
            reverse=True
        )

        print(
            "BGE-M3 RELEVANT RESULTS:",
            len(contexts)
        )

        # -------------------------------------------------
        # EXACT PROCEDURE MATCH
        # -------------------------------------------------
        # Nếu Top-K đã chứa đúng tên thủ tục trong câu hỏi,
        # ưu tiên procedure exact match và bỏ qua reranker.
        # Điều này tránh đưa các thủ tục gần giống như:
        # "Gia hạn...", "Xóa...", "... lưu động" vào context.

        exact_procedure = None

        question_normalized = normalize_vietnamese(
            retrieval_question
        )

        exact_candidates = []

        for item in contexts:

            procedure_name = str(
                item.get(
                    "procedure_name",
                    ""
                )
            ).strip()

            if not procedure_name:
                continue

            procedure_normalized = normalize_vietnamese(
                procedure_name
            )

            # Cho phép câu hỏi bỏ tiền tố "thủ tục".
            procedure_without_prefix = re.sub(
                r"^thu tuc\s+",
                "",
                procedure_normalized
            ).strip()

            matched_length = 0

            if (
                procedure_normalized
                and procedure_normalized in question_normalized
            ):
                matched_length = len(
                    procedure_normalized
                )

            elif (
                procedure_without_prefix
                and procedure_without_prefix in question_normalized
            ):
                matched_length = len(
                    procedure_without_prefix
                )

            if matched_length > 0:
                exact_candidates.append(
                    (
                        matched_length,
                        item
                    )
                )

        if exact_candidates:

            # Nếu có nhiều exact match, ưu tiên tên cụ thể/dài hơn.
            exact_candidates.sort(
                key=lambda x: x[0],
                reverse=True
            )

            exact_procedure = dict(
                exact_candidates[0][1]
            )

            # Không dùng điểm reranker giả; chỉ đánh dấu để formatter
            # có thể hiển thị rõ đây là exact match.
            exact_procedure[
                "rerank_score"
            ] = 1.0

            print(
                "\n=============================="
            )
            print(
                "EXACT PROCEDURE MATCH:",
                exact_procedure[
                    "procedure_name"
                ]
            )
            print(
                "→ BỎ QUA RERANKER"
            )
            print(
                "→ CHỈ DÙNG PROCEDURE EXACT MATCH"
            )

            reranked_contexts = [
                exact_procedure
            ]

        else:

            # Không có exact match → dùng reranker như trước.
            reranked_contexts = (
                await rerank_local_results(
                    retrieval_question,
                    contexts
                )
            )
        # ============================================================
        # EVIDENCE LAYER
        # Direct Catalog + RAG + MCP → Evidence Builder
        # ============================================================

        rag_evidences = rag_results_to_evidence(
            reranked_contexts
        )







        # -------------------------------------------------
        # ENOUGH DATA CHECK
        # -------------------------------------------------

        enough_data, enough_data_fields = check_enough_data(
            question,
            reranked_contexts
        )

        mcp_evidences = []

        if not enough_data:

            print(
                "LOCAL RAG DOCUMENT CÓ NHƯNG "
                "KHÔNG ĐỦ FIELD ĐỂ TRẢ LỜI"
            )

            print(
                "→ GỌI MCP DỊCH VỤ CÔNG"
            )

            try:

                mcp_result = await search_dichvucong(
                    question
                )

                mcp_evidences = (
                    mcp_result.get(
                        "evidences",
                        []
                    )
                )

                print(
                    "MCP EVIDENCES:",
                    len(mcp_evidences)
                )

            except Exception as e:

                print(
                    "MCP ERROR:",
                    repr(e)
                )

                mcp_evidences = []

        # BUILD EVIDENCE BUNDLE AFTER COLLECTING EVIDENCE
        evidence_bundle = evidence_builder.build(
            catalog_evidences=catalog_evidences,
            rag_evidences=rag_evidences,
            mcp_evidences=mcp_evidences,
            query=question,
        )

        # -------------------------------------------------
        # FINAL LOCAL RAG RESULTS
        # -------------------------------------------------

        print(
            "\nFINAL LOCAL RAG RESULTS:",
            len(reranked_contexts)
        )

        formatted_contexts = []

        for i, item in enumerate(
            reranked_contexts
        ):

            # Ưu tiên các trường cấu trúc quan trọng trước full_text.
            # Nhờ vậy các thông tin như "Thời gian giải quyết"
            # không bị cắt mất khi MAX_CONTEXT_CHARS được áp dụng.
            context_parts = [
                f"Tên thủ tục: {item['procedure_name']}",
                f"Điểm BGE-M3: {item['score']:.4f}",
                f"Điểm Reranker: {item['rerank_score']:.4f}",
            ]

            structured_fields = [
                ("Thời gian giải quyết", item.get("processing_time", "")),
                ("Phí và lệ phí", item.get("fees", "")),
                ("Hình thức nộp", item.get("execution_methods", "")),
                ("Thành phần hồ sơ", item.get("profile_components", "")),
                ("Địa điểm tiếp nhận", item.get("location", "")),
                ("Điều kiện thực hiện", item.get("requirements_conditions", "")),
                ("Trình tự thực hiện", item.get("execution_steps", "")),
                ("Căn cứ pháp lý", item.get("legal_basis", "")),
                ("Kết quả", item.get("results", "")),
            ]

            for field_name, field_value in structured_fields:
                if field_value:
                    context_parts.append(
                        f"{field_name}: {field_value}"
                    )

            # Full text chỉ dùng bổ sung; không thay thế các field cấu trúc.
            full_text = item.get("full_text", "")
            if full_text:
                context_parts.append(
                    f"Full dữ liệu thủ tục:\n{full_text}"
                )
            elif item.get("text"):
                context_parts.append(
                    f"Nội dung:\n{item['text']}"
                )

            context = "\n".join(context_parts)

            formatted_contexts.append(
                context
            )

            print(
                f"\n--- FINAL RAG "
                f"DOCUMENT {i + 1} ---"
            )

            print(
                context[:1000]
            )

        if formatted_contexts:

            print(
                "\nLOCAL RAG + RERANKER "
                "FOUND RELEVANT DATA"
            )

        else:

            print(
                "\nLOCAL RAG + RERANKER "
                "KHÔNG TÌM THẤY "
                "DỮ LIỆU ĐỦ LIÊN QUAN"
            )

        return {
            "contexts": formatted_contexts,
            "evidence_bundle": evidence_bundle,
        }

    except Exception as e:
        import traceback
        print("LOCAL RAG ERROR TYPE:", type(e).__name__)
        print("LOCAL RAG ERROR DETAIL:", repr(e))
        traceback.print_exc()

    return {
        "contexts": [],
        "evidence_bundle": None,
    }

# =========================================================
# BOOLEAN PARSER
# =========================================================

def parse_bool(
    value
) -> bool:

    """
    Chuyển dữ liệu permission về bool thật.

    false / "false" / 0 / "0"
    -> False

    true / "true" / 1 / "1"
    -> True
    """

    if isinstance(
        value,
        bool
    ):

        return value

    if value is None:

        return False

    if isinstance(
        value,
        str
    ):

        value = (
            value
            .strip()
            .lower()
        )

        if value in {
            "true",
            "1",
            "yes",
            "y",
            "co",
            "có"
        }:

            return True

        return False

    if isinstance(
        value,
        int
    ):

        return value == 1

    return False


# =========================================================
# MCP CLIENT
# =========================================================

async def call_mcp_tool(
    tool_name: str,
    arguments: dict
):

    print(
        "\n=============================="
    )

    print(
        "MCP TOOL:",
        tool_name
    )

    server_params = (
        StdioServerParameters(

            command=
                sys.executable,

            args=[
                MCP_SERVER_PATH
            ],

            env={
                **os.environ,
                "PYTHONIOENCODING":
                    "utf-8"
            }
        )
    )

    async with stdio_client(
        server_params
    ) as (read, write):

        async with ClientSession(
            read,
            write
        ) as session:

            await session.initialize()

            print(
                "MCP CONNECTED"
            )

            result = (
                await session.call_tool(
                    tool_name,
                    arguments=arguments
                )
            )

            print(
                "MCP TOOL CALLED:",
                tool_name
            )

            mcp_raw_results = []

            for content in result.content:
                if hasattr(content, "text") and content.text:
                    mcp_raw_results.append(
                        content.text
                    )

            print(
                "MCP RAW RESULTS:",
                len(mcp_raw_results)
            )

            mcp_evidences = mcp_results_to_evidence(
                mcp_raw_results
            )

            print(
                "MCP EVIDENCES:",
                len(mcp_evidences)
            )


            print(
                "MCP TOOL CALLED:",
                tool_name
            )

            contexts = []

            for content in (
                result.content
            ):

                if hasattr(
                    content,
                    "text"
                ):

                    text = (
                        content.text
                    )

                    if text:

                        contexts.append(
                            text
                        )

            print(
                "MCP RESULTS:",
                len(contexts)
            )

            for i, text in enumerate(
                contexts
            ):

                print(
                    f"\n--- MCP RESULT "
                    f"{i + 1} ---"
                )

                print(
                    text[:1000]
                )

            return {
                "contexts": contexts,
                "evidences": mcp_evidences,
            }


# =========================================================
# MCP DỊCH VỤ CÔNG
# =========================================================

async def search_dichvucong(
    question: str
):

    return await call_mcp_tool(
        "search_dichvucong",
        {
            "keyword": question
        }
    )


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    contexts: list[str],
    source_type: str
):

    if not contexts:

        return (
            "Mình chưa tìm thấy "
            "thông tin phù hợp."
        )

    combined = "\n\n".join(
        contexts
    )

    combined = combined[
        :MAX_CONTEXT_CHARS
    ]

    if source_type == "local":

        return (
            "[Nguồn dữ liệu: "
            "LOCAL RAG / QDRANT + "
            "BGE-M3 + RERANKER]\n\n"
            + combined
        )

    if source_type == "external":

        return (
            "[Nguồn dữ liệu: "
            "MCP / CỔNG DỊCH VỤ CÔNG]\n\n"
            + combined
        )

    return combined


# =========================================================
# CHAT API
# =========================================================

@app.post(
    "/api/v1/chat/completions"
)
async def chat_completions(
    request: Request,
    data: dict
):

    # =====================================================
    # SESSION
    # =====================================================
    session_id, is_new_session = get_or_create_session_id(request)

    ensure_session(session_id)

    print(
        "SESSION_ID:",
        session_id,
        "| NEW_SESSION:",
        is_new_session
    )

    question = str(
        data.get(
            "prompt",
            ""
        )
    ).strip()
        # ============================================================
    # DIRECT CATALOG
    # ============================================================

    catalog_query = CatalogQuery(
        domain="legal",
        intent="procedure_lookup",
        procedure_name=question,
    )

    catalog_result = catalog_service.lookup(catalog_query)

    catalog_evidences = []

    if catalog_result.matched and catalog_result.record:
        catalog_evidence = catalog_record_to_evidence(
            catalog_result.record
        )
        catalog_evidences.append(catalog_evidence)

        print(
            "\nCATALOG MATCH:",
            catalog_result.record.procedure_name
        )
    else:
        print("\nCATALOG: NO EXACT MATCH")

    # =====================================================
    # PERMISSION
    # =====================================================

    raw_allow_external = data.get(
        "allow_external",
        False
    )

    allow_external = parse_bool(
        raw_allow_external
    )

    # =====================================================
    # XỬ LÝ TRẢ LỜI "CÓ" / "KHÔNG" CHO MCP
    # =====================================================
    # Nếu AI vừa hỏi xin phép tra cứu dữ liệu bên ngoài,
    # người dùng có thể chỉ cần gõ "có". Khi đó phải dùng
    # lại câu hỏi pháp lý trước đó, không được coi "có" là
    # một câu hỏi mới.

    normalized_reply = normalize_vietnamese(question)

    affirmative_replies = {
        "co",
        "co nhe",
        "co a",
        "dong y",
        "ok",
        "oke",
        "okay",
        "yes",
        "duoc",
        "cho phep",
        "tra cuu di",
        "cu tra cuu",
    }

    negative_replies = {
        "khong",
        "khong can",
        "thoi",
        "khong tra cuu",
        "no",
    }

    pending_question = None

    if not allow_external:
        pending_question = get_pending_external_question(
            session_id
        )

        if (
            pending_question
            and normalized_reply in affirmative_replies
        ):
            print(
                "PERMISSION REPLY: USER ĐỒNG Ý"
            )
            print(
                "→ KHÔI PHỤC CÂU HỎI TRƯỚC:",
                pending_question
            )

            question = pending_question
            allow_external = True

        elif (
            pending_question
            and normalized_reply in negative_replies
        ):
            print(
                "PERMISSION REPLY: USER TỪ CHỐI"
            )

    print(
        "\n=============================="
    )

    print(
        "REQUEST QUESTION:",
        question
    )

    print(
        "ALLOW_EXTERNAL:",
        repr(raw_allow_external)
    )

    print(
        "ALLOW_EXTERNAL PARSED:",
        allow_external
    )

    async def generate():

        # =================================================
        # EMPTY QUESTION
        # =================================================

        if not question:

            yield (
                "event: chunk\n"
            )

            yield (
                "data: "
                +
                json.dumps(
                    {
                        "delta":
                            "Vui lòng nhập câu hỏi.",
                        "session_id":
                            session_id
                    },
                    ensure_ascii=False
                )
                +
                "\n\n"
            )

            return

        # =================================================
        # SAVE USER MESSAGE
        # =================================================

        save_message(
            session_id,
            "user",
            question
        )

        # -------------------------------------------------
        # USER TỪ CHỐI TRA CỨU BÊN NGOÀI
        # -------------------------------------------------

        if (
            pending_question
            and normalized_reply in negative_replies
            and not allow_external
        ):
            refusal_message = (
                "Được. Mình sẽ không sử dụng dữ liệu bên ngoài "
                "và sẽ chỉ tra cứu trong dữ liệu pháp lý nội bộ."
            )

            save_message(
                session_id,
                "assistant",
                refusal_message
            )

            yield (
                "event: chunk\n"
            )

            yield (
                "data: "
                + json.dumps(
                    {
                        "delta": refusal_message,
                        "session_id": session_id
                    },
                    ensure_ascii=False
                )
                + "\n\n"
            )

            yield (
                "event: done\n"
            )

            yield (
                'data: {"status":"done"}'
                "\n\n"
            )

            return

        # =================================================
        # STEP 1
        # LOCAL RAG
        # =================================================

        yield "event: status\n"
        yield 'data: {"status":"searching_local_rag"}\n\n'

        rag_result = await search_local_rag(
            question,
            catalog_evidences=catalog_evidences,
        )

        if isinstance(rag_result, dict):
            contexts = rag_result.get("contexts", [])
            evidence_bundle = rag_result.get("evidence_bundle")
        else:
            contexts = []
            evidence_bundle = None

        mcp_evidences = []
        source_type = "local"

        # =================================================
        # STEP 2
        # LOCAL RAG CÓ DỮ LIỆU
        # =================================================

        if contexts:

            print(
                "\nLOCAL RAG + RERANKER "
                "FOUND DATA"
            )

            print(
                "→ KHÔNG GỌI MCP"
            )

            source_type = "local"

        # =================================================
        # STEP 3
        # LOCAL RAG KHÔNG CÓ
        # =================================================

        else:

            print(
                "\nLOCAL RAG + RERANKER "
                "KHÔNG ĐỦ DỮ LIỆU"
            )

            # ---------------------------------------------
            # CHƯA ĐƯỢC USER CHO PHÉP
            # ---------------------------------------------

            if not allow_external:

                print(
                    "→ CHỜ USER CHO PHÉP "
                    "TRA CỨU DỮ LIỆU NGOÀI"
                )

                yield (
                    "event: status\n"
                )

                yield (
                    'data: '
                    '{"status":'
                    '"need_external_permission"}'
                    "\n\n"
                )

                permission_message = (
                    "Mình chưa tìm thấy "
                    "thông tin phù hợp trong "
                    "dữ liệu pháp lý nội bộ.\n\n"
                    "Bạn có muốn mình sử dụng "
                    "dữ liệu bên ngoài từ "
                    "Cổng Dịch vụ công Quốc gia "
                    "để tra cứu thêm không?"
                )

                save_message(
                    session_id,
                    "assistant",
                    permission_message
                )

                yield (
                    "event: chunk\n"
                )

                yield (
                    "data: "
                    +
                    json.dumps(
                        {
                            "delta":
                                permission_message,
                            "session_id":
                                session_id
                        },
                        ensure_ascii=False
                    )
                    +
                    "\n\n"
                )

                yield (
                    "event: done\n"
                )

                yield (
                    'data: {"status":"waiting_permission"}'
                    "\n\n"
                )

                return

            # ---------------------------------------------
            # USER ĐÃ CHO PHÉP
            # ---------------------------------------------

            print(
                "→ USER ĐÃ CHO PHÉP"
            )

            print(
                "→ GỌI MCP DỊCH VỤ CÔNG"
            )

            yield (
                "event: status\n"
            )

            yield (
                'data: '
                '{"status":"searching_mcp"}'
                "\n\n"
            )



            try:
                mcp_result = await search_dichvucong(question)

                contexts = mcp_result.get("contexts", [])

                mcp_raw_results = mcp_result.get("results", [])
                mcp_evidences = mcp_results_to_evidence(
                    mcp_raw_results
                )

            except Exception as e:
                print("MCP ERROR:", repr(e))
                contexts = []
                mcp_evidences = []

            source_type = "external"


        # =================================================
        # STEP 4
        # CONTEXT
        # =================================================

        context_text = build_context(
            contexts,
            source_type
        )

        print(
            "\n=============================="
        )

        print(
            "FINAL CONTEXT SOURCE:",
            source_type
        )

        print(
            context_text[:3000]
        )

        # =================================================
        # STEP 5
        # QWEN
        # =================================================

        # PROMPT ASSEMBLY
        # Evidence Bundle → Prompt
        # =================================================

        # BUILD EVIDENCE BUNDLE



        if evidence_bundle is None:
            print("WARNING: Local RAG không trả về Evidence Bundle")


        # PROMPT ASSEMBLY
        assembled_prompt = prompt_assembler.build(
        question=question,
        evidence_bundle=evidence_bundle,
        system_instruction=(MODEL API ERROR
            str(data.get("system_instruction", "")).strip() or None
        ),
)

        system_prompt = assembled_prompt

        # =================================================
        # STEP 6
        # MODEL
        # =================================================

        yield (
            "event: status\n"
        )

        yield (
            'data: {"status":"thinking"}'
            "\n\n"
        )

        try:

            async with httpx.AsyncClient() as client:

                response = await client.post(
                    MODEL_API_URL,
                    json={
                        "prompt":
                            system_prompt
                    },
                    timeout=180.0
                )

            print(
                "MODEL STATUS:",
                response.status_code
            )

            if response.status_code != 200:

                print(
                    "MODEL ERROR:",
                    response.text
                )

                answer = (
                    "Model API trả về lỗi."
                )

            else:

                result = (
                    response.json()
                )

                print(
                    "MODEL RESULT:",
                    result
                )

                answer = (
                    result.get(
                        "answer",
                        ""
                    )
                    .strip()
                )

                if not answer:

                    answer = (
                        "Model không trả về "
                        "câu trả lời."
                    )

        except Exception as e:

            print(
                "MODEL API ERROR:",
                repr(e)
            )

            answer = (
                "Không thể kết nối "
                "Model API :8001."
            )

        # =================================================
        # SAVE ASSISTANT MESSAGE
        # =================================================

        save_message(
            session_id,
            "assistant",
            answer
        )

        # =================================================
        # RESPONSE
        # =================================================

        yield (
            "event: chunk\n"
        )

        yield (
            "data: "
            +
            json.dumps(
                {
                    "delta":
                        answer,
                    "session_id":
                        session_id
                },
                ensure_ascii=False
            )
            +
            "\n\n"
        )

        yield (
            "event: done\n"
        )

        yield (
            'data: {"status":"done"}'
            "\n\n"
        )

    response = StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":
                "no-cache",
            "Connection":
                "keep-alive",
            "X-Accel-Buffering":
                "no",
            "X-Session-ID":
                session_id
        }
    )

    if is_new_session:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            samesite=SESSION_COOKIE_SAMESITE,
            secure=SESSION_COOKIE_SECURE,
            path="/"
        )

    return response


# =========================================================
# SESSION DEBUG
# =========================================================

@app.get("/api/v1/session")
async def get_session(request: Request):
    """Kiểm tra session hiện tại và đảm bảo session tồn tại trong DB."""
    session_id, is_new_session = get_or_create_session_id(request)

    ensure_session(session_id)

    response = {
        "success": True,
        "session_id": session_id,
        "new_session": is_new_session,
        "database": CHAT_DB_PATH
    }

    if is_new_session:
        from fastapi.responses import JSONResponse

        result = JSONResponse(response)

        result.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            samesite=SESSION_COOKIE_SAMESITE,
            secure=SESSION_COOKIE_SECURE,
            path="/"
        )

        return result

    return response


@app.get("/api/v1/chat/history")
async def chat_history(
    request: Request,
    limit: int = 20
):
    """Lấy lịch sử của đúng session hiện tại."""
    session_id, is_new_session = get_or_create_session_id(request)

    ensure_session(session_id)

    history = get_chat_history(
        session_id,
        limit
    )

    return {
        "success": True,
        "session_id": session_id,
        "count": len(history),
        "messages": history
    }


@app.delete("/api/v1/chat/history")
async def delete_chat_history(request: Request):
    """Xóa lịch sử của đúng session hiện tại."""
    session_id, is_new_session = get_or_create_session_id(request)

    ensure_session(session_id)
    clear_chat_history(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "message": "Đã xóa lịch sử chat của session hiện tại."
    }


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )