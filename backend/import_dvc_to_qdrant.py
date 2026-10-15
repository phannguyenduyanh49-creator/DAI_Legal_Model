import time
import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

QDRANT_URL = "http://localhost:6333"


DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "dichvucong_procedures"
)


EXCEL_PATH = Path(__file__).resolve().parent / "cleaned_data.xlsx"

COLLECTION_NAME = "legal_docs_bge_m3"

EMBEDDING_MODEL = "BAAI/bge-m3"

START_ID = 100000

# ------------------------------------------------------------
# EMBEDDING CONFIG
# ------------------------------------------------------------

# RTX 4070 12GB
BATCH_SIZE = 16

# Giảm độ dài text để embedding nhanh hơn
MAX_TEXT_LENGTH = 1800

# Giảm sequence length để tránh batch quá nặng
MAX_SEQ_LENGTH = 768


# ============================================================
# HELPER
# ============================================================

def clean_value(value):
    """
    Chuyển dữ liệu về string sạch.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, list):

        values = []

        for item in value:

            if item is None:
                continue

            try:
                if pd.isna(item):
                    continue
            except Exception:
                pass

            text = str(item).strip()

            if text:
                values.append(text)

        return ", ".join(values)

    return str(value).strip()


# ============================================================
# NORMALIZE PROCEDURE NAME
# ============================================================

def normalize_name(value):
    """
    Chuẩn hóa tên thủ tục để match:

        DVC dataset
        +
        cleaned_data (1).xlsx
    """

    text = clean_value(value)

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD EXCEL ENRICHMENT
# ============================================================

def load_excel():

    print()
    print("=" * 70)
    print("LOAD CLEANED DATA")
    print("=" * 70)

    if not EXCEL_PATH.exists():

        raise FileNotFoundError(
            "Khong tim thay file Excel:\n"
            f"{EXCEL_PATH}"
        )

    df = pd.read_excel(
        EXCEL_PATH
    )

    print(
        "Excel rows:",
        len(df)
    )

    required_column = (
        "Tên thủ tục hành chính"
    )

    if required_column not in df.columns:

        raise ValueError(
            "Khong tim thay cot: "
            f"{required_column}"
        )

    lookup = {}

    for _, row in df.iterrows():

        procedure_name = clean_value(
            row.get(
                "Tên thủ tục hành chính"
            )
        )

        if not procedure_name:
            continue

        key = normalize_name(
            procedure_name
        )

        if not key:
            continue

        record = {

            "excel_procedure_name":
                procedure_name,

            "processing_time":
                clean_value(
                    row.get(
                        "Thời gian giải quyết"
                    )
                ),

            "excel_fees":
                clean_value(
                    row.get(
                        "Lệ phí"
                    )
                ),

            "excel_method":
                clean_value(
                    row.get(
                        "Hình thức nộp"
                    )
                ),

            "excel_profile":
                clean_value(
                    row.get(
                        "Thành phần hồ sơ"
                    )
                ),

            "excel_location":
                clean_value(
                    row.get(
                        "Địa điểm tiếp nhận hồ sơ trực tiếp (nếu có)"
                    )
                ),

            "excel_note":
                clean_value(
                    row.get(
                        "Ghi chú"
                    )
                ),
        }

        # Nếu trùng tên thủ tục,
        # ưu tiên record có processing_time.
        if key not in lookup:

            lookup[key] = record

        else:

            old = lookup[key]

            if (
                not old.get("processing_time")
                and record.get("processing_time")
            ):
                old["processing_time"] = (
                    record["processing_time"]
                )

            if (
                not old.get("excel_fees")
                and record.get("excel_fees")
            ):
                old["excel_fees"] = (
                    record["excel_fees"]
                )

    print(
        "Excel procedures:",
        len(lookup)
    )

    return lookup


# ============================================================
# BUILD EMBEDDING TEXT
# ============================================================

def build_text(
    row,
    enrichment
):

    fields = [

        (
            "Tên thủ tục",
            row.get(
                "procedure_name"
            )
        ),

        (
            "Lĩnh vực",
            row.get(
                "category_name"
            )
        ),

        (
            "Mô tả",
            row.get(
                "description"
            )
        ),

        (
            "Trình tự thực hiện",
            row.get(
                "execution_steps"
            )
        ),

        (
            "Phương thức thực hiện",
            row.get(
                "execution_methods"
            )
        ),

        (
            "Thành phần hồ sơ",
            row.get(
                "profile_components"
            )
        ),

        (
            "Điều kiện thực hiện",
            row.get(
                "requirements_conditions"
            )
        ),

        (
            "Phí và lệ phí",
            row.get(
                "fees"
            )
        ),

        (
            "Thời gian giải quyết",
            enrichment.get(
                "processing_time"
            )
        ),

        (
            "Căn cứ pháp lý",
            row.get(
                "legal_basis"
            )
        ),

        (
            "Kết quả",
            row.get(
                "results"
            )
        ),

        (
            "Đối tượng thực hiện",
            row.get(
                "target_objects"
            )
        ),

        (
            "Cơ quan thực hiện",
            row.get(
                "executing_agencies"
            )
        ),

        (
            "Từ khóa",
            row.get(
                "keywords"
            )
        ),

        (
            "Nội dung",
            row.get(
                "content_text"
            )
        ),
    ]

    parts = []

    for title, value in fields:

        value = clean_value(
            value
        )

        if value:

            parts.append(
                f"{title}: {value}"
            )

    text = "\n".join(
        parts
    )

    return text[
        :MAX_TEXT_LENGTH
    ]


# ============================================================
# UPDATE EXISTING PAYLOADS
# ============================================================

def update_existing_payloads(
    client,
    excel_lookup
):

    print()
    print("=" * 70)
    print("UPDATE PAYLOAD EXISTING POINTS")
    print("=" * 70)

    offset = None

    total_updated = 0
    total_matched = 0
    total_processing_time = 0

    while True:

        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:

            old_payload = (
                point.payload or {}
            )

            procedure_name = clean_value(
                old_payload.get(
                    "procedure_name"
                )
            )

            key = normalize_name(
                procedure_name
            )

            excel = excel_lookup.get(
                key,
                {}
            )

            processing_time = clean_value(
                excel.get(
                    "processing_time"
                )
            )

            update_payload = {

                "processing_time":
                    processing_time,

                "excel_processing_time":
                    processing_time,

                "excel_matched":
                    bool(excel),

                "excel_procedure_name":
                    clean_value(
                        excel.get(
                            "excel_procedure_name"
                        )
                    ),

                "excel_fees":
                    clean_value(
                        excel.get(
                            "excel_fees"
                        )
                    ),

                "excel_execution_methods":
                    clean_value(
                        excel.get(
                            "excel_method"
                        )
                    ),

                "excel_profile_components":
                    clean_value(
                        excel.get(
                            "excel_profile"
                        )
                    ),

                "excel_location":
                    clean_value(
                        excel.get(
                            "excel_location"
                        )
                    ),

                "excel_note":
                    clean_value(
                        excel.get(
                            "excel_note"
                        )
                    ),
            }

            # =================================================
            # CHỈ UPDATE PAYLOAD
            #
            # KHÔNG ĐỤNG VECTOR
            # =================================================

            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload=update_payload,
                points=[point.id],
            )

            total_updated += 1

            if excel:

                total_matched += 1

            if processing_time:

                total_processing_time += 1

        print(
            f"Updated: {total_updated} | "
            f"matched: {total_matched} | "
            f"processing_time: "
            f"{total_processing_time}"
        )

        if next_offset is None:

            break

        offset = next_offset

    print()
    print(
        "PAYLOAD UPDATE HOAN TAT"
    )

    print(
        "Updated:",
        total_updated
    )

    print(
        "Excel matched:",
        total_matched
    )

    print(
        "Processing time:",
        total_processing_time
    )


# ============================================================
# CREATE PAYLOAD FOR NEW POINT
# ============================================================

def build_payload(
    row,
    text,
    enrichment
):

    return {

        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        "thu_tuc":
            clean_value(
                row.get(
                    "procedure_name"
                )
            ),

        "procedure_name":
            clean_value(
                row.get(
                    "procedure_name"
                )
            ),

        "category_name":
            clean_value(
                row.get(
                    "category_name"
                )
            ),

        "code":
            clean_value(
                row.get(
                    "code"
                )
            ),

        "formality_id":
            clean_value(
                row.get(
                    "formality_id"
                )
            ),

        # ====================================================
        # LEGAL DATA
        # ====================================================

        "description":
            clean_value(
                row.get(
                    "description"
                )
            ),

        "execution_steps":
            clean_value(
                row.get(
                    "execution_steps"
                )
            ),

        "execution_methods":
            clean_value(
                row.get(
                    "execution_methods"
                )
            ),

        "profile_components":
            clean_value(
                row.get(
                    "profile_components"
                )
            ),

        "requirements_conditions":
            clean_value(
                row.get(
                    "requirements_conditions"
                )
            ),

        "fees":
            clean_value(
                row.get(
                    "fees"
                )
            ),

        "processing_time":
            clean_value(
                enrichment.get(
                    "processing_time"
                )
            ),

        "legal_basis":
            clean_value(
                row.get(
                    "legal_basis"
                )
            ),

        "results":
            clean_value(
                row.get(
                    "results"
                )
            ),

        "target_objects":
            clean_value(
                row.get(
                    "target_objects"
                )
            ),

        "executing_agencies":
            clean_value(
                row.get(
                    "executing_agencies"
                )
            ),

        "coordinating_agencies":
            clean_value(
                row.get(
                    "coordinating_agencies"
                )
            ),

        "keywords":
            clean_value(
                row.get(
                    "keywords"
                )
            ),

        # ====================================================
        # FULL CONTENT
        # ====================================================

        "full_text":
            text,

        "full_content_text":
            clean_value(
                row.get(
                    "content_text"
                )
            ),

        "text":
            text,

        # ====================================================
        # SOURCE
        # ====================================================

        "source":
            "tmquan/dichvucong-gov-vn",

        "source_url":
            clean_value(
                row.get(
                    "source_url"
                )
            ),

        # ====================================================
        # EXCEL ENRICHMENT
        # ====================================================

        "excel_matched":
            bool(
                enrichment
            ),

        "excel_procedure_name":
            clean_value(
                enrichment.get(
                    "excel_procedure_name"
                )
            ),

        "excel_processing_time":
            clean_value(
                enrichment.get(
                    "processing_time"
                )
            ),

        "excel_fees":
            clean_value(
                enrichment.get(
                    "excel_fees"
                )
            ),

        "excel_execution_methods":
            clean_value(
                enrichment.get(
                    "excel_method"
                )
            ),

        "excel_profile_components":
            clean_value(
                enrichment.get(
                    "excel_profile"
                )
            ),

        "excel_location":
            clean_value(
                enrichment.get(
                    "excel_location"
                )
            ),

        "excel_note":
            clean_value(
                enrichment.get(
                    "excel_note"
                )
            ),
    }


# ============================================================
# RESUME EMBEDDING
# ============================================================

def resume_embedding(
    client,
    dataset,
    excel_lookup,
    resume_from
):

    total = len(
        dataset
    )

    print()
    print("=" * 70)
    print("RESUME BGE-M3 EMBEDDING")
    print("=" * 70)

    print(
        "Resume from:",
        resume_from
    )

    print(
        "Total:",
        total
    )

    remaining = (
        total
        - resume_from
    )

    print(
        "Remaining:",
        remaining
    )

    if remaining <= 0:

        print(
            "Khong con record can embed."
        )

        return

    # ========================================================
    # LOAD BGE-M3
    # ========================================================

    print()
    print(
        "Loading BGE-M3..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cuda"
    )

    model.max_seq_length = (
        MAX_SEQ_LENGTH
    )

    print(
        "BGE-M3 loaded."
    )

    print(
        "Max sequence length:",
        MAX_SEQ_LENGTH
    )

    # ========================================================
    # START
    # ========================================================

    start_time = time.time()

    for start in range(
        resume_from,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = dataset[
            start:end
        ]

        rows = []

        for i in range(
            end - start
        ):

            row = {
                key: batch[key][i]
                for key in batch.keys()
            }

            rows.append(
                row
            )

        # ====================================================
        # BUILD TEXTS
        # ====================================================

        texts = []

        enrichments = []

        for row in rows:

            key = normalize_name(
                row.get(
                    "procedure_name"
                )
            )

            enrichment = (
                excel_lookup.get(
                    key,
                    {}
                )
            )

            enrichments.append(
                enrichment
            )

            text = build_text(
                row,
                enrichment
            )

            texts.append(
                text
            )

        # ====================================================
        # BGE-M3 EMBEDDING
        # ====================================================

        encode_start = time.time()

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        encode_time = (
            time.time()
            - encode_start
        )

        # ====================================================
        # CREATE QDRANT POINTS
        # ====================================================

        points = []

        for i, (
            row,
            vector,
            text,
            enrichment
        ) in enumerate(
            zip(
                rows,
                embeddings,
                texts,
                enrichments
            )
        ):

            point_id = (
                START_ID
                + start
                + i
            )

            payload = build_payload(
                row,
                text,
                enrichment
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        # ====================================================
        # UPSERT
        # ====================================================

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        # ====================================================
        # PROGRESS
        # ====================================================

        elapsed = (
            time.time()
            - start_time
        )

        done = (
            end
            - resume_from
        )

        remaining_count = (
            total
            - end
        )

        speed = (
            done / elapsed
            if elapsed > 0
            else 0
        )

        eta = (
            remaining_count / speed
            if speed > 0
            else 0
        )

        percent = (
            end
            / total
            * 100
        )

        print(
            f"[{end:4d}/{total}] "
            f"{percent:6.2f}% | "
            f"encode={encode_time:.2f}s | "
            f"speed={speed:.1f} rows/s | "
            f"ETA={eta:.1f}s"
        )


# ============================================================
# VERIFY COLLECTION
# ============================================================

def verify_collection(
    client,
    expected_total
):

    print()
    print("=" * 70)
    print("VERIFY COLLECTION")
    print("=" * 70)

    info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        "Collection:",
        COLLECTION_NAME
    )

    print(
        "Points:",
        info.points_count
    )

    print(
        "Expected:",
        expected_total
    )

    if info.points_count == expected_total:

        print(
            "STATUS: OK"
        )

    else:

        print(
            "STATUS: CHUA DU"
        )


# ============================================================
# VERIFY BIRTH REGISTRATION
# ============================================================

def verify_birth_registration(
    client
):

    print()
    print("=" * 70)
    print("VERIFY: THU TUC DANG KY KHAI SINH")
    print("=" * 70)

    target = normalize_name(
        "Thủ tục đăng ký khai sinh"
    )

    offset = None

    while True:

        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:

            payload = (
                point.payload or {}
            )

            procedure_name = normalize_name(
                payload.get(
                    "procedure_name"
                )
            )

            if procedure_name == target:

                print()
                print(
                    "FOUND IN QDRANT"
                )

                print(
                    "Point ID:",
                    point.id
                )

                print(
                    "Procedure:",
                    payload.get(
                        "procedure_name"
                    )
                )

                print(
                    "processing_time:",
                    payload.get(
                        "processing_time"
                    )
                )

                print(
                    "excel_matched:",
                    payload.get(
                        "excel_matched"
                    )
                )

                print(
                    "excel_processing_time:",
                    payload.get(
                        "excel_processing_time"
                    )
                )

                print(
                    "excel_fees:",
                    payload.get(
                        "excel_fees"
                    )
                )

                return

        if next_offset is None:

            break

        offset = next_offset

    print(
        "Khong tim thay "
        "Thủ tục đăng ký khai sinh."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DVC + EXCEL -> BGE-M3 -> QDRANT")
    print("=" * 70)

    # ========================================================
    # CONNECT QDRANT
    # ========================================================

    client = QdrantClient(
        url=QDRANT_URL
    )

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            "Khong tim thay dataset DVC:\n"
            f"{DATASET_PATH}"
        )

    print()
    print(
        "Loading DVC dataset..."
    )

    dataset = load_from_disk(
        str(DATASET_PATH)
    )

    total = len(
        dataset
    )

    print(
        "Dataset:",
        total
    )

    # ========================================================
    # CHECK COLLECTION
    # ========================================================

    try:

        info = client.get_collection(
            COLLECTION_NAME
        )

    except Exception:

        print()
        print(
            "Collection chua ton tai."
        )

        print(
            "Tao collection moi..."
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
        )

        current_points = 0

    else:

        current_points = (
            info.points_count
        )

    print()
    print(
        "Current Qdrant points:",
        current_points
    )

    # ========================================================
    # SAFETY
    # ========================================================
    #
    # TUYỆT ĐỐI KHÔNG DELETE COLLECTION.
    #
    # ========================================================

    if current_points > total:

        raise RuntimeError(
            f"Qdrant co {current_points} points, "
            f"nhieu hon dataset {total}."
        )

    # ========================================================
    # LOAD EXCEL
    # ========================================================

    excel_lookup = load_excel()

    # ========================================================
    # UPDATE PAYLOAD EXISTING DATA
    # ========================================================

    if current_points > 0:

        update_existing_payloads(
            client,
            excel_lookup
        )

    else:

        print()
        print(
            "Khong co point cu de update."
        )

    # ========================================================
    # CHECK CURRENT POINTS AGAIN
    # ========================================================

    current_points = (
        client.get_collection(
            COLLECTION_NAME
        ).points_count
    )

    print()
    print(
        "Points after payload update:",
        current_points
    )

    # ========================================================
    # RESUME EMBEDDING
    # ========================================================

    if current_points < total:

        print()
        print(
            "Tiep tuc embedding tu index:",
            current_points
        )

        resume_embedding(
            client,
            dataset,
            excel_lookup,
            current_points
        )

    else:

        print()
        print(
            "Collection da du data."
        )

    # ========================================================
    # FINAL VERIFY
    # ========================================================

    verify_collection(
        client,
        total
    )

    verify_birth_registration(
        client
    )

    print()
    print("=" * 70)
    print("HOAN TAT")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()