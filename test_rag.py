from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "legal_docs"


print("=" * 60)
print("TEST RAG - QDRANT")
print("=" * 60)


# =========================
# CONNECT QDRANT
# =========================

client = QdrantClient(QDRANT_URL)

info = client.get_collection(COLLECTION_NAME)

print("Collection:", COLLECTION_NAME)
print("Points:", info.points_count)


# =========================
# LOAD EMBEDDING MODEL
# =========================

print("\nĐang load embedding model...")

model = SentenceTransformer(
    "bkai-foundation-models/vietnamese-bi-encoder"
)

print("Embedding model OK")


# =========================
# SEARCH LOOP
# =========================

while True:

    question = input(
        "\nNhập câu hỏi (exit để thoát): "
    ).strip()

    if question.lower() == "exit":
        print("Đã thoát.")
        break

    if not question:
        continue


    # =========================
    # QUESTION → VECTOR
    # =========================

    query_vector = model.encode(
        question
    ).tolist()


    # =========================
    # SEARCH QDRANT
    # =========================

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
    ).points


    print("\n" + "=" * 60)
    print("RAG SEARCH RESULTS")
    print("=" * 60)


    if not results:
        print("Không tìm thấy dữ liệu.")
        continue


    for i, result in enumerate(results, 1):

        payload = result.payload or {}

        print(f"\n--- RESULT {i} ---")

        print(
            "Score:",
            round(result.score, 4)
        )

        print(
            "Thủ tục:",
            payload.get(
                "thu_tuc",
                "N/A"
            )
        )

        print(
            "Source:",
            payload.get(
                "source",
                "N/A"
            )
        )

        print("\nContext:")

        print(
            payload.get(
                "text",
                "N/A"
            )[:2000]
        )

        print("-" * 60)