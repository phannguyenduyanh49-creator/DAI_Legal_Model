from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.catalog.service import CatalogService
from backend.catalog.query_parser import catalog_query_from_text
from backend.domain_documents import (
    DomainDocumentError,
    DomainDocumentService,
    domain_document_collection,
    retrieval_collections_for_domain,
)
from backend.domain_registry import registry_service
from backend.chat_session import (
    NEGATIVE_REPLIES,
    find_pending_external_question,
    normalize_session_id,
    resolve_permission_reply,
)
from backend.mcp.adapter import (
    mcp_result_status,
    mcp_results_to_evidence
)
from backend.mcp.client import call_stdio_tool, make_server_parameters
from backend.mcp.contract import MCPToolCallError
from backend.policy.middleware import PolicyEnforcementMiddleware

from backend.prompt.assembler import PromptAssembler

from backend.evidence.normalizer import (
    procedure_to_evidence,
    catalog_record_to_evidence,
)

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from backend.rag.adapter import rag_results_to_evidence
from backend.rag.diagnostics import (
    attach_retrieval_failure_telemetry,
    classify_retrieval_result,
    retrieval_exception_details,
)
from backend.rag.relevance import procedure_relevance
from backend.rag.data_sufficiency import check_enough_data
from backend.rag.pipeline_policy import (
    build_catalog_fallback_after_retrieval_error,
    check_evidence_sufficiency,
    choose_data_source,
)
from backend.evidence.builder import EvidenceBuilder
from backend.evidence.schemas import EvidenceBundle
from backend.answer.citations import structure_answer_with_explicit_evidence
from backend.answer.evidence_gate import make_missing_prompt_evidence_result
from backend.answer.reference_policy import (
    classify_business_outcome,
    reference_answer_failure_reason,
)
from backend.answer.citation_generation import generate_with_citation_retry
from backend.answer.model_api_client import post_model_generation
from backend.answer.diagnostics import (
    classify_upstream_exception,
    citation_raw_debug_enabled,
    make_citation_attempt_logger,
    safe_raw_answer_for_debug,
    verifier_reason_codes,
)
from backend.verifier.verifier import Verifier
import httpx
import json
import asyncio
import os
import sys
import re
import time
import hashlib
import unicodedata
import uuid
import sqlite3
from datetime import datetime, timezone


catalog_service = CatalogService()
evidence_builder = EvidenceBuilder(
    min_score=0.3,
    max_evidences=10,
)

def _bounded_int_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


prompt_assembler = PromptAssembler(
    max_evidence_chars=_bounded_int_setting(
        "PROMPT_MAX_EVIDENCE_CHARS", 6000, 1000, 12000
    )
)
MODEL_MAX_NEW_TOKENS = _bounded_int_setting("MODEL_MAX_NEW_TOKENS", 256, 1, 512)
_raw_citation_debug_enabled = citation_raw_debug_enabled(os.environ)
print(
    "[CITATION DEBUG CONFIG]",
    "enabled=", _raw_citation_debug_enabled,
    flush=True,
)
verifier = Verifier()
policy_middleware = PolicyEnforcementMiddleware()

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

    return find_pending_external_question(CHAT_DB_PATH, session_id)


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

    normalized_session_id = normalize_session_id(session_id)
    if normalized_session_id:
        return normalized_session_id, False

    # Never turn arbitrary cookie contents into a session namespace.
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

domain_document_service = DomainDocumentService(
    qdrant=qdrant_client,
    embedder=embedding_model,
    registry=registry_service,
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
        print("[QUERY NORMALIZATION]", "input_chars=", len(original), "output_chars=", len(expanded))

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
    diagnostics = procedure_relevance(question, procedure_name)
    print(
        "[RETRIEVAL RELEVANCE]",
        "score=", round(score, 4),
        "match_type=", diagnostics["match_type"],
        "matched_terms=", diagnostics["matched_term_count"],
        "query_terms=", diagnostics["query_term_count"],
        "query_coverage=", round(diagnostics["query_coverage"], 4),
        "title_precision=", round(diagnostics["title_precision"], 4),
    )
    return diagnostics["relevant"]


# =========================================================
# RERANK LOCAL RESULTS
# =========================================================

async def rerank_local_results(
    question: str,
    contexts: list[dict],
    request_id: str = "unknown",
    stage_state: dict | None = None,
):

    if not contexts:
        if stage_state is not None:
            stage_state["reranking"] = "not_started"
        return []

    print(
        "[PIPELINE STAGE]",
        "request_id=", request_id,
        "stage=reranking",
        "status=started",
        "candidate_count=", len(contexts),
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

                    print("[RERANK FILTER]", "request_id=", request_id, "decision=keep_by_margin", "score_gap=", round(score_gap, 4))

                else:

                    print("[RERANK FILTER]", "request_id=", request_id, "decision=reject_by_margin", "score_gap=", round(score_gap, 4))

            filtered = adaptive_filtered[:MAX_RERANK_RESULTS]

        # -------------------------------------------------
        # TOP N
        # -------------------------------------------------

        filtered = filtered[
            :MAX_RERANK_RESULTS
        ]

        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=reranking",
            "status=completed",
            "selected_count=", len(filtered),
            "top_score=", round(filtered[0]["rerank_score"], 4) if filtered else None,
        )

        if stage_state is not None:
            stage_state["reranking"] = "completed"

        return filtered

    except Exception as e:

        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=reranking",
            "status=failed",
            "reason_code=UNKNOWN_PIPELINE_FAILURE",
            "exception_type=", type(e).__name__,
        )

        if stage_state is not None:
            stage_state["reranking"] = "failed"

        # Nếu reranker lỗi thì không làm
        # toàn bộ Local RAG chết.
        return contexts


# =========================================================
# LOCAL RAG
# =========================================================
async def search_local_rag(
    question: str,
    catalog_evidences: list | None = None,
    allow_external: bool = False,
    request_id: str = "unknown",
    domain_id: str = "legal_ai",
):
    if catalog_evidences is None:
        catalog_evidences = []
    stage_durations_ms = {}
    retrieval_collection_status = "unknown"
    retrieval_started_at = time.perf_counter()
    print(
        "[PIPELINE STAGE]",
        "request_id=", request_id,
        "stage=retrieval",
        "status=started",
        "question_chars=", len(question),
    )

    # =====================================================
    # SYNONYM MAPPING CHO RETRIEVAL
    # =====================================================
    # Chi mo rong query dung cho BGE-M3/Qdrant/Reranker.
    # Cau hoi goc van duoc giu nguyen cho Qwen va session.
    retrieval_question = expand_synonyms_for_retrieval(question)

    try:

        # -------------------------------------------------
        # BGE-M3 QUERY EMBEDDING
        # -------------------------------------------------

        embedding_started_at = time.perf_counter()
        query_vector = (
            await asyncio.to_thread(
                embedding_model.encode,
                retrieval_question,
                normalize_embeddings=True
            )
        )
        stage_durations_ms["embedding"] = round(
            (time.perf_counter() - embedding_started_at) * 1000, 2
        )

        # -------------------------------------------------
        # QDRANT TOP-K
        # -------------------------------------------------

        domain_config = registry_service.get_domain_config(domain_id)
        retrieval_collections = retrieval_collections_for_domain(
            domain_id,
            qdrant_client.collection_exists,
            registry_service,
        )
        configured_collection = domain_config.knowledge_collection
        if configured_collection and configured_collection not in retrieval_collections:
            retrieval_collection_status = (
                "partial" if retrieval_collections else "base_collection_missing"
            )
            print(
                "[RETRIEVAL COLLECTION]",
                "request_id=", request_id,
                "domain_id=", domain_id,
                "status=", retrieval_collection_status,
            )
        else:
            retrieval_collection_status = (
                "available" if retrieval_collections else "no_domain_documents"
            )

        qdrant_started_at = time.perf_counter()
        results = []
        for collection_name in retrieval_collections:
            response = await asyncio.to_thread(
                qdrant_client.query_points,
                collection_name=collection_name,
                query=query_vector.tolist(),
                limit=MAX_LOCAL_RESULTS,
                with_payload=True,
            )
            results.extend(response.points)
        stage_durations_ms["qdrant"] = round(
            (time.perf_counter() - qdrant_started_at) * 1000, 2
        )

        contexts = []
        rejected_irrelevance_count = 0

        # -------------------------------------------------
        # LOCAL PROCEDURE EXACT LOOKUP
        # -------------------------------------------------
        exact_local_items = (
            await asyncio.to_thread(find_exact_local_procedures, retrieval_question)
            if domain_config.knowledge_collection == QDRANT_COLLECTION
            else []
        )
        stage_durations_ms["local_retrieval"] = round(
            (time.perf_counter() - retrieval_started_at) * 1000, 2
        )

        if exact_local_items:
            print(
                "[PIPELINE STAGE]",
                "request_id=", request_id,
                "stage=local_exact_retrieval",
                "match_count=", len(exact_local_items),
            )
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
                    "source_type": payload.get("source_type"),
                    "domain_id": payload.get("domain_id"),
                    "source_id": payload.get("source_id"),
                    "evidence_id": payload.get("evidence_id"),
                    "title": payload.get("title"),
                    "location": payload.get("location"),
                })

        # -------------------------------------------------
        # PROCESS BGE-M3 RESULTS
        # -------------------------------------------------

        for result in results:

            payload = (
                result.payload
                or {}
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
            source_type = str(payload.get("source_type", "") or "")

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
            candidate_id = str(getattr(result, "id", "unknown"))[:100]
            safe_payload_fields = sorted(
                key for key in payload
                if key in {"thu_tuc", "procedure_name", "domain", "category", "source_type"}
            )

            if not text:
                print(
                    "[RETRIEVAL CANDIDATE]", "request_id=", request_id,
                    "candidate_id=", candidate_id,
                    "payload_fields=", ",".join(safe_payload_fields),
                    "score=", round(score, 4), "overlap=not_computed",
                    "decision=reject_empty_text",
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
                    print(
                        "[RETRIEVAL CANDIDATE]", "request_id=", request_id,
                        "candidate_id=", candidate_id,
                        "procedure_name=", procedure_name[:160],
                        "payload_fields=", ",".join(safe_payload_fields),
                        "score=", round(score, 4), "overlap=not_computed",
                        "decision=reject_low_score",
                    )
                    continue

            if source_type != "domain_upload" and not is_relevant_procedure(
                retrieval_question,
                procedure_name,
                score
            ):
                rejected_irrelevance_count += 1
                relevance = procedure_relevance(retrieval_question, procedure_name)
                print(
                    "[RETRIEVAL CANDIDATE]", "request_id=", request_id,
                    "candidate_id=", candidate_id,
                    "procedure_name=", procedure_name[:160],
                    "payload_fields=", ",".join(safe_payload_fields),
                    "score=", round(score, 4),
                    "query_coverage=", round(relevance["query_coverage"], 4),
                    "title_precision=", round(relevance["title_precision"], 4),
                    "matched_terms=", relevance["matched_term_count"],
                    "decision=reject_irrelevant",
                )

                continue

            print(
                "[RETRIEVAL CANDIDATE]",
                "request_id=", request_id,
                "decision=accepted",
                "score=", round(score, 4),
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
                    "source_type": source_type or None,
                    "domain_id": payload.get("domain_id"),
                    "source_id": payload.get("source_id"),
                    "evidence_id": payload.get("evidence_id"),
                    "title": payload.get("title"),
                    "location": payload.get("location"),
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
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=retrieval",
            "status=completed",
            "candidate_count=", len(contexts),
        )

        # -------------------------------------------------
        # EXACT PROCEDURE MATCH
        # -------------------------------------------------
        # Nếu Top-K đã chứa đúng tên thủ tục trong câu hỏi,
        # ưu tiên procedure exact match và bỏ qua reranker.
        # Điều này tránh đưa các thủ tục gần giống như:
        # "Gia hạn...", "Xóa...", "... lưu động" vào context.

        reranking_started_at = time.perf_counter()
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

            reranked_contexts = [
                exact_procedure
            ]
            reranking_status = "completed"

        else:

            # Không có exact match → dùng reranker như trước.
            rerank_state = {}
            reranked_contexts = (
                await rerank_local_results(
                    retrieval_question,
                    contexts,
                    request_id=request_id,
                    stage_state=rerank_state,
                )
            )
            reranking_status = rerank_state.get("reranking", "unknown")
        stage_durations_ms["reranking"] = round(
            (time.perf_counter() - reranking_started_at) * 1000, 2
        )
        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=reranking",
            "status=", reranking_status,
            "candidate_count=", len(contexts),
            "selected_count=", len(reranked_contexts),
        )
        # ============================================================
        # EVIDENCE LAYER
        # Direct Catalog + RAG + MCP → Evidence Builder
        # ============================================================

        evidence_started_at = time.perf_counter()
        rag_evidences = rag_results_to_evidence(
            reranked_contexts
        )







        # -------------------------------------------------
        # ENOUGH DATA CHECK
        # -------------------------------------------------

        catalog_sufficiency_candidates = [
            {
                **evidence.fields,
                "text": evidence.content,
            }
            for evidence in catalog_evidences
        ]
        enough_data, required_fields, missing_fields = check_enough_data(
            question,
            catalog_sufficiency_candidates + reranked_contexts,
        )

        mcp_evidences = []
        external_fallback_status = (
            "not_needed" if enough_data
            else "pending" if allow_external
            else "not_allowed"
        )

        # BUILD EVIDENCE BUNDLE AFTER COLLECTING EVIDENCE
        evidence_bundle = evidence_builder.build(
            catalog_evidences=catalog_evidences,
            rag_evidences=rag_evidences,
            mcp_evidences=mcp_evidences,
            query=question,
        )
        stage_durations_ms["evidence_and_sufficiency"] = round(
            (time.perf_counter() - evidence_started_at) * 1000, 2
        )
        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=data_sufficiency",
            "status=", "sufficient" if enough_data else "insufficient",
            "required_field_count=", len(required_fields),
            "missing_field_count=", len(missing_fields),
        )
        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=evidence_builder",
            "status=completed",
            "catalog_count=", len(catalog_evidences),
            "rag_count=", len(rag_evidences),
            "mcp_count=", len(mcp_evidences),
            "evidence_count=", len(evidence_bundle.evidences),
        )
        # -------------------------------------------------
        # FINAL LOCAL RAG RESULTS
        # -------------------------------------------------

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

        retrieved_count = len(results)
        reranked_count = len(reranked_contexts)
        evidence_count = len(evidence_bundle.evidences)
        retrieval_status = classify_retrieval_result(
            retrieved_count,
            len(contexts),
            rejected_irrelevance_count,
        )

        return {
            "contexts": formatted_contexts,
            "evidence_bundle": evidence_bundle,
            "retrieval_status": retrieval_status,
            "reranking_status": reranking_status,
            "evidence_builder_status": "completed",
            "enough_data": enough_data,
            "required_fields": required_fields,
            "missing_fields": missing_fields,
            "external_fallback": external_fallback_status,
            "retrieved_count": retrieved_count,
            "reranked_count": reranked_count,
            "evidence_count": evidence_count,
            "retrieval_collection_status": retrieval_collection_status,
            "stage_durations_ms": stage_durations_ms,
            "stage_status": {
                "retrieval": retrieval_status,
                "reranking": reranking_status,
                "evidence_builder": "completed",
                "data_sufficiency": "completed",
            },
        }

    except Exception as e:
        stage_durations_ms.setdefault(
            "local_retrieval",
            round((time.perf_counter() - retrieval_started_at) * 1000, 2),
        )
        failure_details = retrieval_exception_details(e)
        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=retrieval_pipeline",
            "status=failed",
            "component=", failure_details["component"],
            "reason_code=", failure_details["reason_code"],
            "exception_type=", type(e).__name__,
            "cause_exception_type=", failure_details["cause_exception_type"],
        )

        fallback = build_catalog_fallback_after_retrieval_error(
            question=question,
            catalog_evidences=catalog_evidences,
            evidence_builder=evidence_builder,
            allow_external=allow_external,
        )
        fallback["stage_durations_ms"] = stage_durations_ms
        fallback["retrieved_count"] = 0
        fallback["reranked_count"] = 0
        fallback["evidence_count"] = len(fallback["evidence_bundle"].evidences)
        fallback["retrieval_collection_status"] = retrieval_collection_status
        fallback["stage_status"] = {
            "retrieval": "failed",
            "reranking": "not_started",
            "evidence_builder": "completed",
            "data_sufficiency": "completed",
        }
        return attach_retrieval_failure_telemetry(fallback, e)

    return {
        "contexts": [],
        "evidence_bundle": None,
        "retrieval_status": "error",
        "reranking_status": "error",
        "evidence_builder_status": "not_run",
        "enough_data": False,
        "required_fields": [],
        "missing_fields": [],
        "external_fallback": "not_evaluated",
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
    arguments: dict,
    allowed_tools: list[str] | tuple[str, ...],
    request_id: str = "unknown",
):

    if tool_name not in frozenset(allowed_tools):
        raise MCPToolCallError(
            f"Tool MCP khong duoc cap quyen: {tool_name}"
        )

    print(
        "\n=============================="
    )

    print(
        "[MCP STAGE]",
        "request_id=", request_id,
        "tool=",
        tool_name
    )

    mcp_started_at = time.perf_counter()
    mcp_stage_timings: dict[str, float] = {}
    try:
        timeout_seconds = float(os.getenv("MCP_CLIENT_TIMEOUT_SECONDS", "60"))
        server_params = make_server_parameters(
            script_path=MCP_SERVER_PATH,
            cwd=BASE_DIR,
            python_executable=sys.executable,
            allowed_tools=allowed_tools,
            request_id=request_id,
            timeout_seconds=timeout_seconds,
            base_environment=os.environ,
        )

        def log_tool_stage(stage: str, elapsed_ms: float) -> None:
            mcp_stage_timings[stage] = round(elapsed_ms, 2)
            stage_status = (
                "timeout" if stage.endswith("_timeout")
                else "error" if stage.endswith("_error")
                else "completed"
            )
            print(
                "[MCP STAGE]", "request_id=", request_id,
                "stage=", stage, "status=", stage_status,
                "elapsed_ms=", round(elapsed_ms, 2),
            )

        result_text = await call_stdio_tool(
            server_params,
            tool_name,
            arguments,
            allowed_tools=allowed_tools,
            timeout_seconds=timeout_seconds,
            on_stage=log_tool_stage,
        )
    except (MCPToolCallError, ValueError) as exc:
        elapsed_ms = round((time.perf_counter() - mcp_started_at) * 1000, 2)
        print(
            "[MCP STAGE]", "request_id=", request_id,
            "status=failed", "tool=", tool_name,
            "reason_code=", getattr(exc, "reason_code", "INVALID_CONFIGURATION"),
            "exception_type=", type(exc).__name__,
            "elapsed_ms=", elapsed_ms,
        )
        raise

    print(
        "[MCP STAGE]",
        "request_id=", request_id,
        "status=called tool=",
        tool_name
    )

    if tool_name == "search_dichvucong":
        result_status = mcp_result_status(result_text)
        if result_status == "invalid_response":
            raise MCPToolCallError(
                "MCP returned a malformed DVC result.", "MCP_INVALID_RESPONSE"
            )
        mcp_raw_results = [result_text] if result_status == "completed" else []
    else:
        result_status = "completed" if result_text else "no_match"
        mcp_raw_results = []
    mapping_started_at = time.perf_counter()
    mcp_evidences = mcp_results_to_evidence(mcp_raw_results)
    mapping_elapsed_ms = round((time.perf_counter() - mapping_started_at) * 1000, 2)
    elapsed_ms = round((time.perf_counter() - mcp_started_at) * 1000, 2)
    mcp_stage_timings["result_mapping"] = mapping_elapsed_ms
    mcp_stage_timings["total"] = elapsed_ms
    print(
        "[MCP RESULTS]",
        "request_id=", request_id,
        "status=", result_status,
        "raw_count=", len(mcp_raw_results),
        "chars=", sum(map(len, mcp_raw_results)),
        "mapped_evidence_count=", len(mcp_evidences),
        "mapping_elapsed_ms=", mapping_elapsed_ms,
        "elapsed_ms=", elapsed_ms,
    )
    contexts = [result_text] if result_status == "completed" and result_text else []
    return {
        "status": result_status,
        "contexts": contexts,
        "evidences": mcp_evidences,
        "stage_durations_ms": mcp_stage_timings,
        "raw_result_count": len(mcp_raw_results),
        "mapped_evidence_count": len(mcp_evidences),
    }


# =========================================================
# MCP DỊCH VỤ CÔNG
# =========================================================

async def search_dichvucong(
    question: str,
    allowed_tools: list[str] | tuple[str, ...],
    request_id: str = "unknown",
):

    return await call_mcp_tool(
        "search_dichvucong",
        {
            "keyword": question
        },
        allowed_tools=allowed_tools,
        request_id=request_id,
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

@app.post("/api/v1/domains/{domain_id}/documents")
async def upload_domain_document(domain_id: str, file: UploadFile = File(...)):
    try:
        registry_service.get_domain_config(domain_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown domain") from exc

    content = await file.read(2 * 1024 * 1024 + 1)
    filename = file.filename or ""
    await file.close()
    try:
        return await asyncio.to_thread(
            domain_document_service.upload,
            domain_id,
            filename,
            content,
        )
    except DomainDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        print(
            "[DOMAIN DOCUMENT UPLOAD]",
            "domain_id=", domain_id,
            "status=failed",
            "exception_type=", type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Domain document indexing failed") from exc


@app.post(
    "/api/v1/chat/completions"
)
async def chat_completions(
    request: Request,
    data: dict
):
    request_id = request.headers.get("x-request-id") or f"core_{uuid.uuid4().hex}"

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
    policy_domain = str(data.get("policy_domain", "") or "").strip().lower()
    effective_domain = policy_domain or "legal_ai"
    try:
        selected_domain_config = registry_service.get_domain_config(effective_domain)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown domain") from exc
        # ============================================================
    # DIRECT CATALOG
    # ============================================================
    catalog_result = None
    catalog_lookup_duration_ms = 0.0
    catalog_lookup_status = "not_allowed"
    if selected_domain_config.knowledge_collection == QDRANT_COLLECTION:
        catalog_query = catalog_query_from_text(question)
        catalog_started_at = time.perf_counter()
        catalog_result, catalog_lookup_status = catalog_service.lookup_optional(catalog_query)
        catalog_lookup_duration_ms = round((time.perf_counter() - catalog_started_at) * 1000, 2)
    if catalog_lookup_status == "unavailable":
        print(
            "[CATALOG LOOKUP]",
            "request_id=", request_id,
            "status=unavailable",
            "reason=database_operational_error",
        )
    elif selected_domain_config.knowledge_collection == QDRANT_COLLECTION and catalog_result is None:
        catalog_lookup_status = "unknown"
        print(
            "[CATALOG LOOKUP]",
            "request_id=", request_id,
            "status=unknown",
            "reason=result_missing",
        )
    elif catalog_result is not None:
        catalog_lookup_status = "matched" if catalog_result.matched else "no_match"

    catalog_evidences = []

    if catalog_result is not None and catalog_result.matched and catalog_result.record:
        catalog_evidence = catalog_record_to_evidence(
            catalog_result.record
        )
        catalog_evidences.append(catalog_evidence)

        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=catalog_lookup",
            "status=matched",
            "evidence_count=1",
            "duration_ms=", catalog_lookup_duration_ms,
        )
    elif catalog_result is not None:
        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=catalog_lookup",
            "status=no_match",
            "evidence_count=0",
            "duration_ms=", catalog_lookup_duration_ms,
        )

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

    requested_mcp_tools = data.get("mcp_allowed_tools", [])
    if not isinstance(requested_mcp_tools, list):
        requested_mcp_tools = []
    domain_prompt_template = None
    verification_threshold = None
    try:
        if policy_domain:
            core_policy = policy_middleware.apply(policy_domain)
            domain_prompt_template = core_policy.domain_config.prompt_template
            verification_threshold = (
                core_policy.domain_config.policy.verification_threshold
            )
            effective_mcp_tools = policy_middleware.constrain_tools(
                policy_domain,
                (str(tool) for tool in requested_mcp_tools),
            )
        else:
            effective_mcp_tools = ()
    except KeyError:
        effective_mcp_tools = ()
        domain_prompt_template = None
    print(
        "[PIPELINE STAGE]", "request_id=", request_id,
        "stage=mcp_policy", "policy_domain=", policy_domain or "missing",
        "allowed_tools=", len(effective_mcp_tools),
    )

    # =====================================================
    # XỬ LÝ TRẢ LỜI "CÓ" / "KHÔNG" CHO MCP
    # =====================================================
    # Nếu AI vừa hỏi xin phép tra cứu dữ liệu bên ngoài,
    # người dùng có thể chỉ cần gõ "có". Khi đó phải dùng
    # lại câu hỏi pháp lý trước đó, không được coi "có" là
    # một câu hỏi mới.

    normalized_reply = normalize_vietnamese(question)

    pending_question = None

    if not allow_external:
        pending_question = get_pending_external_question(
            session_id
        )

        permission_resolution = resolve_permission_reply(
            question,
            pending_question,
            allow_external,
            normalized_reply,
        )
        if permission_resolution.decision == "approved":
            print(
                "PERMISSION REPLY: USER ĐỒNG Ý"
            )
            print("[SESSION]", "request_id=", request_id, "status=pending_question_restored")

            question = permission_resolution.question
            allow_external = permission_resolution.allow_external

        elif permission_resolution.decision == "denied":
            print(
                "PERMISSION REPLY: USER TỪ CHỐI"
            )

    print(
        "[PIPELINE STAGE]",
        "request_id=", request_id,
        "stage=request",
        "status=accepted",
        "question_chars=", len(question),
        "allow_external=", allow_external,
    )

    async def generate():
        pipeline_state = {
            "request_id": request_id,
            "stage_durations_ms": {
                "catalog_lookup": catalog_lookup_duration_ms,
            },
            "stage_status": {
                "catalog_lookup": catalog_lookup_status,
                "retrieval": "not_started",
                "reranking": "not_started",
                "evidence_builder": "not_started",
                "data_sufficiency": "not_started",
                "prompt_assembly": "not_started",
                "model_api": "not_started",
                "verifier": "not_started",
            },
            "catalog_lookup": catalog_lookup_status,
            "retrieval": "not_started",
            "reranking": "not_started",
            "evidence_builder": "not_started",
            "enough_data": "not_evaluated",
            "answer_data_sufficiency": "not_evaluated",
            "required_fields": [],
            "missing_fields": [],
            "answer_required_fields": [],
            "answer_missing_fields": [],
            "external_fallback": "not_evaluated",
            "prompt_assembly": "not_started",
            "model_api": "not_started",
            "citation_verification": "not_started",
            "claim_grounding_status": "not_evaluated",
            "confidence_status": "not_evaluated",
            "confidence_scorer_status": "not_configured",
            "confidence_reason": None,
            "grounding_scorer_status": "not_configured",
            "grounding_reason": "NO_GROUNDING_SCORER_CONFIGURED",
            "verification_outcome": "not_verified",
            "claim_verification_results": [],
            "verification_status": "not_started",
            "response_outcome": "pending",
            "answer_withheld": True,
            "citation_attempts": 0,
            "raw_answer_debug_enabled": _raw_citation_debug_enabled,
            "evidence_count": 0,
            "evidence_ids": [],
            "cited_evidence_ids": [],
            "model_processing_time_ms": None,
            "model_api_elapsed_ms": 0.0,
        }

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
            and normalized_reply in NEGATIVE_REPLIES
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
                    'data: {"status":"done","verifier":"not_required",'
                    '"verifier_scope":"not_applicable",'
                    '"pipeline":{"retrieval":"not_started",'
                    '"reranking":"not_started",'
                    '"citation_verification":"not_required"}}'
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
            allow_external=allow_external,
            request_id=request_id,
            domain_id=effective_domain,
        )

        if isinstance(rag_result, dict):
            contexts = rag_result.get("contexts", [])
            evidence_bundle = rag_result.get("evidence_bundle")
            pipeline_state["retrieval"] = rag_result.get(
                "retrieval_status", "unknown"
            )
            pipeline_state["reranking"] = rag_result.get(
                "reranking_status", "unknown"
            )
            pipeline_state["evidence_builder"] = rag_result.get(
                "evidence_builder_status", "unknown"
            )
            enough_data = rag_result.get("enough_data")
            pipeline_state["enough_data"] = (
                "sufficient" if enough_data is True
                else "insufficient" if enough_data is False
                else "not_evaluated"
            )
            pipeline_state["required_fields"] = rag_result.get(
                "required_fields", []
            )
            pipeline_state["missing_fields"] = rag_result.get(
                "missing_fields", []
            )
            pipeline_state["answer_data_sufficiency"] = pipeline_state[
                "enough_data"
            ]
            pipeline_state["answer_required_fields"] = list(
                pipeline_state["required_fields"]
            )
            pipeline_state["answer_missing_fields"] = list(
                pipeline_state["missing_fields"]
            )
            pipeline_state["external_fallback"] = rag_result.get(
                "external_fallback", "not_evaluated"
            )
            pipeline_state["retrieved_count"] = rag_result.get(
                "retrieved_count", 0
            )
            pipeline_state["reranked_count"] = rag_result.get(
                "reranked_count", len(contexts)
            )
            pipeline_state["evidence_count"] = rag_result.get(
                "evidence_count", 0
            )
            pipeline_state["retrieval_collection_status"] = rag_result.get(
                "retrieval_collection_status", "unknown"
            )
            pipeline_state["stage_durations_ms"].update(
                rag_result.get("stage_durations_ms", {})
            )
            pipeline_state["stage_status"].update(
                rag_result.get("stage_status", {})
            )
            pipeline_state["stage_status"]["data_sufficiency"] = (
                "completed" if enough_data is not None else "unknown"
            )
            for key in (
                "failure_stage",
                "failure_reason_code",
                "exception_type",
                "cause_exception_type",
                "failure_component",
            ):
                if rag_result.get(key):
                    pipeline_state[key] = rag_result[key]
        else:
            contexts = []
            evidence_bundle = None

        if evidence_bundle is None:
            evidence_bundle = EvidenceBundle(query=question)

        print(
            "[RETRIEVAL RESULT]",
            "request_id=", request_id,
            "contexts=", len(contexts),
            "evidence_count=", len(evidence_bundle.evidences),
        )
        pipeline_state["evidence_count"] = len(evidence_bundle.evidences)

        mcp_evidences = []
        source_type = "local"

        # =================================================
        # STEP 2
        # LOCAL RAG CÓ DỮ LIỆU
        # =================================================

        has_local_evidence = bool(contexts or evidence_bundle.evidences)
        dvc_tool_allowed = "search_dichvucong" in effective_mcp_tools
        data_source = choose_data_source(
            has_local_evidence=has_local_evidence,
            enough_data=(
                pipeline_state["enough_data"] == "sufficient"
                if pipeline_state["enough_data"] != "not_evaluated"
                else None
            ),
            allow_external=allow_external,
            mcp_tool_allowed=dvc_tool_allowed,
        )
        if not dvc_tool_allowed and pipeline_state["enough_data"] == "insufficient":
            pipeline_state["external_fallback"] = "policy_denied"

        if data_source == "local":

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

            if data_source == "permission":

                print(
                    "→ CHỜ USER CHO PHÉP "
                    "TRA CỨU DỮ LIỆU NGOÀI"
                )
                pipeline_state["citation_verification"] = "not_required"
                pipeline_state["external_fallback"] = "awaiting_permission"
                pipeline_state["response_status"] = "waiting_permission"
                pipeline_state["response_outcome"] = "needs_clarification"

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
                    "data: "
                    + json.dumps(
                        {
                            "status": "waiting_permission",
                            "outcome": "needs_clarification",
                            "verifier": "not_required",
                            "verifier_scope": "not_applicable",
                            "pipeline": pipeline_state,
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
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
                mcp_result = await search_dichvucong(
                    question,
                    allowed_tools=effective_mcp_tools,
                    request_id=request_id,
                )

                contexts = mcp_result.get("contexts", [])
                mcp_evidences = mcp_result.get("evidences", [])
                pipeline_state["external_fallback"] = mcp_result.get(
                    "status", "completed"
                )
                pipeline_state["mcp_status"] = mcp_result.get("status", "unknown")
                pipeline_state["mcp_raw_result_count"] = mcp_result.get(
                    "raw_result_count", 0
                )
                pipeline_state["mcp_mapped_evidence_count"] = mcp_result.get(
                    "mapped_evidence_count", len(mcp_evidences)
                )
                (
                    final_enough_data,
                    final_required_fields,
                    final_missing_fields,
                ) = check_evidence_sufficiency(question, mcp_evidences)
                pipeline_state["answer_data_sufficiency"] = (
                    "sufficient" if final_enough_data else "insufficient"
                )
                pipeline_state["answer_required_fields"] = final_required_fields
                pipeline_state["answer_missing_fields"] = final_missing_fields
                pipeline_state["stage_durations_ms"].update({
                    f"mcp_{stage}": duration
                    for stage, duration in mcp_result.get(
                        "stage_durations_ms", {}
                    ).items()
                    if isinstance(duration, (int, float))
                })

            except Exception as e:
                mcp_failure_code = getattr(
                    e, "reason_code", "MCP_UPSTREAM_ERROR"
                )
                print(
                    "[PIPELINE STAGE]",
                    "request_id=", request_id,
                    "stage=external_fallback",
                    "status=failed",
                    "reason_code=", mcp_failure_code,
                    "exception_type=", type(e).__name__,
                )
                contexts = []
                mcp_evidences = []
                pipeline_state["external_fallback"] = (
                    "timeout"
                    if mcp_failure_code in {
                        "MCP_TOOL_TIMEOUT",
                        "MCP_SERVER_STARTUP_TIMEOUT",
                        "MCP_INITIALIZATION_TIMEOUT",
                    }
                    else "error"
                )
                pipeline_state["external_fallback_reason_code"] = mcp_failure_code

            source_type = "external"

            evidence_bundle = evidence_builder.build(
                rag_evidences=evidence_bundle.evidences,
                mcp_evidences=mcp_evidences,
                query=question,
            )
        pipeline_state["evidence_builder"] = "completed"
        pipeline_state["evidence_count"] = len(evidence_bundle.evidences)

        # Do not spend an inference call when there is no traceable Evidence
        # for the model. The Gateway will receive an explicit fail-closed state.
        prompt_evidences = prompt_assembler.included_evidences(evidence_bundle)
        missing_evidence_result = make_missing_prompt_evidence_result(
            pipeline_state,
            source_candidate_count=(
                len(catalog_evidences) + len(contexts) + len(mcp_evidences)
            ),
            evidence_bundle_count=len(evidence_bundle.evidences),
            prompt_evidences=prompt_evidences,
        )
        if missing_evidence_result is not None:
            pipeline_state["response_outcome"] = "withheld"
            pipeline_state["answer_withheld"] = True
            save_message(
                session_id,
                "assistant",
                missing_evidence_result["answer"],
            )
            print(
                "[PIPELINE STAGE]",
                "request_id=", request_id,
                "stage=", pipeline_state["failure_stage"],
                "status=failed",
                "reason_code=EVIDENCE_IDS_MISSING",
                "source_candidate_count=", len(catalog_evidences) + len(contexts) + len(mcp_evidences),
                "evidence_bundle_count=", len(evidence_bundle.evidences),
                "prompt_evidence_count=", len(prompt_evidences),
            )
            yield "event: chunk\n"
            yield "data: " + json.dumps(
                {
                    "delta": missing_evidence_result["answer"],
                    "session_id": session_id,
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield "event: done\n"
            yield "data: " + json.dumps(
                {
                    "status": "done",
                    "outcome": "withheld",
                    "verifier": missing_evidence_result["verifier"],
                    "verifier_scope": missing_evidence_result["verifier_scope"],
                    "pipeline": missing_evidence_result["pipeline"],
                },
                ensure_ascii=False,
            ) + "\n\n"
            return


        # =================================================
        # STEP 4
        # CONTEXT
        # =================================================

        context_text = build_context(
            contexts,
            source_type
        )

        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=context_assembly",
            "status=completed",
            "source=", source_type,
            "context_chars=", len(context_text),
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
            pipeline_state["stage_status"]["prompt_assembly"] = "not_started"
            print(
                "[PIPELINE STAGE]",
                "request_id=", request_id,
                "stage=prompt_assembly",
                "status=failed",
                "reason_code=UNKNOWN_PIPELINE_FAILURE",
            )
            evidence_bundle = EvidenceBundle(query=question)


        # PROMPT ASSEMBLY
        prompt_started_at = time.perf_counter()
        assembled_prompt = prompt_assembler.build(
        question=question,
        evidence_bundle=evidence_bundle,
        system_instruction=(
        str(data.get("system_instruction", "")).strip() or None
        ),
        domain_prompt_template=domain_prompt_template,
)

        system_prompt = assembled_prompt
        verification_bundle = EvidenceBundle(
            query=evidence_bundle.query,
            evidences=prompt_evidences,
            total=len(prompt_evidences),
            metadata=dict(evidence_bundle.metadata),
        )
        pipeline_state["prompt_assembly"] = "completed"
        pipeline_state["stage_status"]["prompt_assembly"] = "completed"
        pipeline_state["stage_durations_ms"]["prompt_assembly"] = round(
            (time.perf_counter() - prompt_started_at) * 1000, 2
        )
        pipeline_state["evidence_count"] = len(prompt_evidences)
        pipeline_state["evidence_ids"] = [
            item.evidence_id for item in prompt_evidences
        ]
        # Keep source identity attached to the exact Evidence supplied to the
        # model. Omit content and internal retrieval metadata from the API.
        pipeline_state["evidence_sources"] = [
            {
                "evidence_id": item.evidence_id,
                "source_type": item.source_type,
                "source_id": item.source_id,
                "title": item.title,
                "source_url": item.source_url,
                "location": item.location,
            }
            for item in prompt_evidences
        ]
        print(
            "[PIPELINE STAGE]",
            "request_id=", request_id,
            "stage=prompt_assembly",
            "status=completed",
            "prompt_chars=", len(system_prompt),
            "has_evidence=", bool(prompt_evidences),
            "evidence_count=", len(prompt_evidences),
        )
        verification_status = "error"

        class RequestTimedVerifier:
            def verify(self, structured_answer, selected_evidence_bundle):
                verify_started_at = time.perf_counter()
                try:
                    return verifier.verify(
                        structured_answer,
                        selected_evidence_bundle,
                        minimum_confidence=verification_threshold,
                        expected_domain_id=effective_domain,
                    )
                finally:
                    pipeline_state["stage_durations_ms"]["citation_verification"] = round(
                        pipeline_state["stage_durations_ms"].get(
                            "citation_verification", 0.0
                        ) + (time.perf_counter() - verify_started_at) * 1000,
                        2,
                    )

        timed_verifier = RequestTimedVerifier()

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

            async def call_model(model_prompt: str, citation_attempt: int) -> dict:
                prompt_digest = hashlib.sha256(
                    model_prompt.encode("utf-8")
                ).hexdigest()
                print(
                    "[PIPELINE STAGE]",
                    "request_id=", request_id,
                    "stage=model_api",
                    "status=started",
                    "citation_attempt=", citation_attempt,
                    "prompt_chars=", len(model_prompt),
                    "prompt_sha256=", prompt_digest,
                )
                model_started_at = time.perf_counter()
                try:
                    async with httpx.AsyncClient() as client:
                        response = await post_model_generation(
                            client=client,
                            url=MODEL_API_URL,
                            prompt=model_prompt,
                            request_id=request_id,
                            citation_attempt=citation_attempt,
                            max_new_tokens=MODEL_MAX_NEW_TOKENS,
                            timeout_seconds=180.0,
                        )
                except Exception as exc:
                    elapsed_ms = round(
                        (time.perf_counter() - model_started_at) * 1000,
                        2,
                    )
                    pipeline_state["model_api_elapsed_ms"] += elapsed_ms
                    pipeline_state["stage_durations_ms"]["model_api"] = round(
                        pipeline_state["stage_durations_ms"].get("model_api", 0.0)
                        + elapsed_ms,
                        2,
                    )
                    reason_code = classify_upstream_exception(exc)
                    attempt_failure = {
                        "citation_attempt": citation_attempt,
                        "reason_code": reason_code,
                        "exception_type": type(exc).__name__,
                    }
                    if isinstance(exc, httpx.HTTPStatusError):
                        attempt_failure["upstream_status_code"] = exc.response.status_code
                    pipeline_state.setdefault("generation_attempt_failures", []).append(
                        attempt_failure
                    )
                    print(
                        "[PIPELINE STAGE]",
                        "request_id=", request_id,
                        "stage=model_api",
                        "status=failed",
                        "citation_attempt=", citation_attempt,
                        "reason_code=", reason_code,
                        "exception_type=", type(exc).__name__,
                        "upstream_status_code=", attempt_failure.get("upstream_status_code"),
                        "elapsed_ms=", elapsed_ms,
                        "prompt_chars=", len(model_prompt),
                        "prompt_sha256=", prompt_digest,
                    )
                    raise
                elapsed_ms = round(
                    (time.perf_counter() - model_started_at) * 1000,
                    2,
                )
                pipeline_state["model_api_elapsed_ms"] += elapsed_ms
                pipeline_state["stage_durations_ms"]["model_api"] = round(
                    pipeline_state["stage_durations_ms"].get("model_api", 0.0)
                    + elapsed_ms,
                    2,
                )
                print(
                    "[PIPELINE STAGE]",
                    "request_id=", request_id,
                    "stage=model_api",
                    "status=responded",
                    "citation_attempt=", citation_attempt,
                    "upstream_status_code=", response.status_code,
                    "elapsed_ms=", elapsed_ms,
                    "prompt_chars=", len(model_prompt),
                    "prompt_sha256=", prompt_digest,
                )
                try:
                    result = response.json()
                except ValueError:
                    return {}
                if not isinstance(result, dict):
                    return {}
                raw_answer = result.get("answer")
                print(
                    "[PIPELINE STAGE]",
                    "request_id=", request_id,
                    "stage=model_api_parse",
                    "status=completed" if isinstance(raw_answer, str) else "failed",
                    "citation_attempt=", citation_attempt,
                    "answer_chars=", len(raw_answer) if isinstance(raw_answer, str) else 0,
                )
                return result

            outcome = await generate_with_citation_retry(
                prompt=assembled_prompt,
                evidence_bundle=verification_bundle,
                generate=call_model,
                verifier=timed_verifier,
                max_attempts=2,
                raw_answer_debug_enabled=_raw_citation_debug_enabled,
                on_attempt=make_citation_attempt_logger(
                    request_id,
                    safe_raw_answer_for_debug,
                    include_raw_answer=_raw_citation_debug_enabled,
                    allowed_evidence_ids=tuple(
                        evidence.evidence_id
                        for evidence in verification_bundle.evidences
                    ),
                ),
            )
            structured_answer = outcome.structured_answer
            verification_result = outcome.verification
            generation_succeeded = (
                outcome.generation_error is None
                and outcome.last_model_response.get("success") is True
            )
            if not generation_succeeded:
                verification_status = "error"
            elif not verification_result.citation_integrity_passed:
                verification_status = "failed"
            elif verification_result.confidence_status in ("failed", "invalid_score"):
                verification_status = "failed"
            elif (
                verification_result.confidence_status == "not_verified"
                or verification_result.claim_grounding_status == "not_evaluated"
            ):
                verification_status = "not_verified"
            else:
                verification_status = "passed" if verification_result.passed else "failed"
            pipeline_state["model_api"] = (
                "completed" if generation_succeeded else "error"
            )
            pipeline_state["stage_status"]["model_api"] = (
                "completed" if generation_succeeded else "failed"
            )
            pipeline_state["model_processing_time_ms"] = (
                outcome.model_processing_time_ms
            )
            pipeline_state["citation_verification"] = (
                "passed" if verification_result.citation_integrity_passed
                else "failed" if generation_succeeded else "error"
            )
            pipeline_state["confidence_status"] = verification_result.confidence_status
            pipeline_state["confidence_score"] = verification_result.confidence_score
            pipeline_state["verification_threshold"] = verification_threshold
            pipeline_state["claim_grounding_status"] = (
                verification_result.claim_grounding_status
            )
            pipeline_state["confidence_scorer_status"] = (
                verification_result.confidence_scorer_status
            )
            pipeline_state["confidence_reason"] = verification_result.confidence_reason
            pipeline_state["confidence_reason_code"] = (
                "CONFIDENCE_EVALUATOR_UNAVAILABLE"
                if verification_result.confidence_reason
                == "NO_CALIBRATED_CONFIDENCE_SCORER_CONFIGURED"
                else "CONFIDENCE_EVALUATOR_TIMEOUT"
                if verification_result.confidence_reason == "CONFIDENCE_SCORER_TIMEOUT"
                else verification_result.confidence_reason
            )
            pipeline_state["grounding_scorer_status"] = (
                verification_result.grounding_scorer_status
            )
            pipeline_state["verification_outcome"] = verification_result.outcome
            pipeline_state["grounding_reason"] = verification_result.grounding_reason
            pipeline_state["grounding_reason_code"] = (
                "GROUNDING_EVALUATOR_UNAVAILABLE"
                if verification_result.grounding_reason
                == "NO_GROUNDING_SCORER_CONFIGURED"
                else "GROUNDING_EVALUATOR_TIMEOUT"
                if verification_result.grounding_reason == "GROUNDING_SCORER_TIMEOUT"
                else verification_result.grounding_reason
            )
            pipeline_state["claim_verification_results"] = (
                verification_result.claim_results
            )
            confidence_was_evaluated = (
                verification_result.confidence_status != "not_required"
                and verification_result.confidence_scorer_status == "completed"
            )
            if confidence_was_evaluated and verification_result.semantic_support_evaluated:
                pipeline_state["verifier_scope"] = (
                    "citation_confidence_and_semantic_grounding"
                )
            elif confidence_was_evaluated:
                pipeline_state["verifier_scope"] = "citation_integrity_and_confidence"
            elif verification_result.semantic_support_evaluated:
                pipeline_state["verifier_scope"] = (
                    "citation_integrity_and_semantic_grounding"
                )
            else:
                pipeline_state["verifier_scope"] = "citation_integrity_only"
            pipeline_state["verification_status"] = verification_status
            pipeline_state["stage_status"]["verifier"] = (
                "passed" if verification_status == "passed"
                else verification_status
                if verification_status in (
                    "failed", "not_verified", "evaluator_unavailable"
                )
                else "unknown"
            )
            answer = (
                outcome.answer
                if generation_succeeded and outcome.answer
                else outcome.reference_answer
                if generation_succeeded and outcome.reference_answer
                else "Model API request failed; no answer was returned."
                if not generation_succeeded
                else "Chua the dua ra cau tra loi phap ly du dieu kien "
                "kiem tra trich dan Evidence."
            )
            retry_attempts = outcome.attempts
            pipeline_state["citation_attempts"] = retry_attempts
            pipeline_state["citation_attempt_diagnostics"] = (
                outcome.attempt_diagnostics
            )
            pipeline_state["verifier_reason_counts"] = verifier_reason_codes(
                verification_result
            )
            if outcome.generation_error:
                last_failure = next(
                    reversed(pipeline_state.get("generation_attempt_failures", [])),
                    {},
                )
                pipeline_state["failure_stage"] = "model_api"
                pipeline_state["failure_reason_code"] = last_failure.get(
                    "reason_code"
                ) or outcome.failure_reason_code or "UNKNOWN_PIPELINE_FAILURE"
                pipeline_state["exception_type"] = last_failure.get(
                    "exception_type", outcome.generation_error
                )
                if "upstream_status_code" in last_failure:
                    pipeline_state["upstream_status_code"] = last_failure[
                        "upstream_status_code"
                    ]
            elif generation_succeeded and verification_status == "passed":
                pipeline_state.pop("failure_stage", None)
                pipeline_state.pop("failure_reason_code", None)
                pipeline_state.pop("exception_type", None)
                pipeline_state.pop("upstream_status_code", None)
            elif verification_status == "failed":
                pipeline_state["failure_stage"] = "verifier"
                pipeline_state["failure_reason_code"] = (
                    outcome.failure_reason_code
                    if not verification_result.citation_integrity_passed
                    else "VERIFICATION_CONFIDENCE_NOT_PASSED"
                    if verification_result.confidence_status in ("failed", "invalid_score")
                    else "CLAIM_GROUNDING_NOT_PASSED"
                    if verification_result.claim_grounding_status == "failed"
                    else outcome.failure_reason_code or "VERIFIER_NOT_PASSED"
                )
            elif verification_status == "not_verified":
                pipeline_state["failure_stage"] = "verifier"
                pipeline_state["failure_reason_code"] = (
                    "VERIFICATION_CONFIDENCE_NOT_VERIFIED"
                    if verification_result.confidence_status == "not_verified"
                    else outcome.failure_reason_code or "CLAIM_GROUNDING_NOT_EVALUATED"
                )
            elif verification_status == "error":
                pipeline_state["failure_stage"] = "model_api_response"
                pipeline_state["failure_reason_code"] = "UNKNOWN_PIPELINE_FAILURE"
            pipeline_state["cited_evidence_ids"] = sorted({
                evidence_id
                for claim in structured_answer.claims
                for evidence_id in claim.evidence_ids
            })
            answer_is_verified = (
                generation_succeeded
                and verification_status == "passed"
                and verification_result.passed
                and verification_result.citation_integrity_passed
            )
            reference_failure_reason = (
                None
                if answer_is_verified
                else reference_answer_failure_reason(
                    answer,
                    pipeline_state,
                    verification_status,
                )
            )
            answer_is_unverified_reference = (
                not answer_is_verified and reference_failure_reason is None
            )
            pipeline_state["reference_answer_reason_code"] = (
                None if answer_is_unverified_reference else reference_failure_reason
            )
            pipeline_state["response_outcome"] = classify_business_outcome(
                pipeline_state,
                answer_is_verified=answer_is_verified,
                answer_is_unverified_reference=answer_is_unverified_reference,
            )
            pipeline_state["answer_withheld"] = not (
                answer_is_verified or answer_is_unverified_reference
            )
            if pipeline_state["answer_withheld"]:
                answer = (
                    "Evidence hiện có chưa đủ các thông tin thiết yếu để "
                    "hiển thị câu trả lời an toàn; câu trả lời đã được giữ lại."
                    if reference_failure_reason
                    and reference_failure_reason.startswith("REFERENCE_ANSWER_ESSENTIAL_FACTS")
                    else "Câu trả lời đã được giữ lại vì chưa đạt các kiểm tra Evidence."
                )
            print(
                "[CITATION INTEGRITY]",
                "request_id=", request_id,
                "status=", verification_status,
                "citation_status=", pipeline_state["citation_verification"],
                "claim_grounding_status=", pipeline_state["claim_grounding_status"],
                "confidence_status=", pipeline_state["confidence_status"],
                "checked_claims=", verification_result.checked_claims,
                "linked_claims=", verification_result.linked_claims,
                "semantic_support_evaluated=",
                verification_result.semantic_support_evaluated,
                "attempts=", retry_attempts,
                "reason_counts=", json.dumps(
                    pipeline_state["verifier_reason_counts"],
                    sort_keys=True,
                ),
            )
            print(
                "[ANSWER RESULT]",
                "status=", pipeline_state["response_outcome"],
                "answer_chars=", len(answer),
                "attempts=", retry_attempts,
                "cited_evidence_ids=", pipeline_state["cited_evidence_ids"],
                "claims_with_evidence=", sum(
                    bool(claim.evidence_ids)
                    for claim in structured_answer.claims
                ),
            )

        except Exception as e:

            verification_status = "error"
            pipeline_state["model_api"] = "error"
            pipeline_state["citation_verification"] = "error"
            pipeline_state["stage_status"]["model_api"] = "failed"
            pipeline_state["stage_status"]["verifier"] = "unknown"
            pipeline_state["failure_stage"] = "generation_pipeline"
            pipeline_state["failure_reason_code"] = classify_upstream_exception(e)
            pipeline_state["exception_type"] = type(e).__name__

            print(
                "[PIPELINE STAGE]",
                "request_id=", request_id,
                "stage=generation_pipeline",
                "status=failed",
                "reason_code=", pipeline_state["failure_reason_code"],
                "exception_type=", type(e).__name__,
            )

            answer = (
                "Không thể kết nối "
                "Model API :8001."
            )

        if verification_status == "error":
            answer = "Model API request failed; no answer was returned."
        if pipeline_state.get("response_outcome") == "pending":
            pipeline_state["response_outcome"] = classify_business_outcome(
                pipeline_state,
                answer_is_verified=False,
                answer_is_unverified_reference=False,
            )
            pipeline_state["answer_withheld"] = True

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
            "data: "
            + json.dumps(
                {
                    "status": "done",
                    "outcome": pipeline_state["response_outcome"],
                    "verifier": verification_status,
                    "verifier_scope": pipeline_state.get(
                        "verifier_scope", "citation_integrity_only"
                    ),
                    "pipeline": pipeline_state,
                },
                ensure_ascii=False,
            )
            + "\n\n"
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
