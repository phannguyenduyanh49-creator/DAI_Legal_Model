
import hashlib
import io
import re
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from qdrant_client import QdrantClient, models

from backend.domain_registry import registry_service


router = APIRouter(prefix="/api/v2/domain-data", tags=["Domain Data Upload"])

QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
MAX_FILE_SIZE = 20 * 1024 * 1024
BATCH_SIZE = 64

_client = None
_embedding_model = None


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
    return _client


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cuda",
        )
        _embedding_model.max_seq_length = 1024
    return _embedding_model


def collection_for_domain(domain_id: str) -> str:
    # Không dùng collection pháp luật hiện có.
    safe_id = re.sub(r"[^a-z0-9_]", "_", domain_id.lower())
    return f"domain_{safe_id}_bge_m3"


@router.post("/upload")
async def upload_domain_excel(
    domain_id: str,
    file: Annotated[UploadFile, File()],
):
    domain_id = domain_id.strip().lower()

    try:
        config = registry_service.get_domain_config(domain_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain_id}' chưa được đăng ký.",
        )

    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=400,
            detail="Chỉ chấp nhận file Excel .xlsx.",
        )

    raw = await file.read(MAX_FILE_SIZE + 1)
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File vượt quá giới hạn 20 MB.",
        )

    try:
        dataframe = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Không đọc được Excel: {type(exc).__name__}.",
        )

    dataframe = dataframe.dropna(how="all")
    if dataframe.empty:
        raise HTTPException(
            status_code=400,
            detail="Excel không có dữ liệu.",
        )

    dataframe.columns = [str(c).strip() for c in dataframe.columns]
    records = dataframe.fillna("").astype(str).to_dict(orient="records")

    chunk_size = config.chunking_strategy.chunk_size
    overlap = config.chunking_strategy.chunk_overlap
    if overlap >= chunk_size:
        raise HTTPException(
            status_code=500,
            detail="Chunk overlap phải nhỏ hơn chunk size.",
        )

    chunks = []
    for row_number, record in enumerate(records, start=2):
        text = "\n".join(
            f"{key}: {value.strip()}"
            for key, value in record.items()
            if value.strip()
        )
        if not text:
            continue

        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "row": row_number,
                    "source": file.filename,
                })
            if start + chunk_size >= len(text):
                break
            start += chunk_size - overlap

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Không tìm thấy nội dung để lập chỉ mục.",
        )

    collection_name = collection_for_domain(domain_id)
    client = get_client()
    model = get_embedding_model()
    vector_size = model.get_sentence_embedding_dimension()

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    uploaded_count = 0
    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        texts = [item["text"] for item in batch]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        points = []
        for item, vector in zip(batch, vectors):
            point_id = hashlib.sha256(
                (
                    domain_id
                    + "|"
                    + file.filename
                    + "|"
                    + str(item["row"])
                    + "|"
                    + item["text"]
                ).encode("utf-8")
            ).hexdigest()[:32]

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "domain_id": domain_id,
                        "text": item["text"],
                        "source": item["source"],
                        "row": item["row"],
                    },
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        uploaded_count += len(points)

    return {
        "success": True,
        "domain_id": domain_id,
        "domain_name": config.domain_name,
        "collection": collection_name,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_size": vector_size,
        "rows_read": len(records),
        "chunks_indexed": uploaded_count,
        "message": "Excel đã được lập chỉ mục. Luồng chat hiện tại cần được tích hợp retrieval theo domain để truy vấn dữ liệu này.",
    }
