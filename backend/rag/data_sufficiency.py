import re
import unicodedata


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", str(text).lower().strip())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value)).strip()


def detect_required_fields(question: str) -> list[str]:
    query = _normalize(question)
    patterns = {
        "profile_components": (
            "giay to", "giay", "ho so", "thanh phan ho so", "can chuan bi",
            "chuan bi nhung gi", "can nhung gi", "bao gom nhung gi", "can nop", "nop nhung gi",
        ),
        "processing_time": (
            "bao lau", "thoi gian", "thoi han", "may ngay", "bao nhieu ngay",
            "giai quyet trong", "mat bao lau",
        ),
        "fees": (
            "le phi", "phi bao nhieu", "muc phi", "co phi", "mat bao nhieu",
            "ton bao nhieu", "bao nhieu tien",
        ),
        "execution_methods": (
            "hinh thuc nop", "nop o dau", "nop ho so o dau", "nop truc tuyen",
            "nop truc tiep", "nop online", "cach nop",
        ),
        "execution_steps": (
            "trinh tu", "cac buoc", "thuc hien nhu the nao", "lam nhu the nao", "quy trinh",
        ),
        "requirements_conditions": (
            "dieu kien", "yeu cau", "dieu kien thuc hien", "ai duoc", "doi tuong nao",
        ),
    }
    required = [field for field, phrases in patterns.items() if any(p in query for p in phrases)]
    return required or ["full_text"]


def check_enough_data(
    question: str,
    reranked_items: list[dict],
) -> tuple[bool, list[str], list[str]]:
    """Check requested fields, combining complementary chunks only for one record.

    Candidates without a stable record key are kept separate. This prevents
    unrelated procedures from collectively satisfying one request.
    """
    required_fields = detect_required_fields(question)
    if not reranked_items:
        return False, required_fields, list(required_fields)

    groups: dict[tuple[str, str], list[dict]] = {}
    for index, item in enumerate(reranked_items):
        identity = None
        for key in ("formality_id", "code"):
            value = item.get(key)
            if value is not None and str(value).strip():
                identity = (key, str(value).strip())
                break
        if identity is None and item.get("document_id") and item.get("record_id"):
            identity = (
                "document_record",
                f"{item['document_id']}:{item['record_id']}",
            )
        if identity is None and item.get("record_id"):
            identity = ("record_id", str(item["record_id"]).strip())
        group_key = identity or ("candidate", str(index))
        groups.setdefault(group_key, []).append(item)

    candidate_gaps = []
    for items in groups.values():
        has_content = any(
            str(item.get("full_text", "") or item.get("text", "")).strip()
            for item in items
        )
        if not has_content:
            candidate_gaps.append(["content"])
            continue

        missing_fields = []
        for field in required_fields:
            if field == "full_text":
                continue
            values = {
                " ".join(str(item.get(field, "")).split()).casefold()
                for item in items
                if str(item.get(field, "")).strip()
            }
            # Conflicting values do not count as complete data.
            if not values or len(values) > 1:
                missing_fields.append(field)
        if not missing_fields:
            return True, required_fields, []
        candidate_gaps.append(missing_fields)

    best_gaps = min(candidate_gaps, key=len) if candidate_gaps else list(required_fields)
    return False, required_fields, best_gaps
