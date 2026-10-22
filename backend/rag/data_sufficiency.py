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
    """Check the best retrieved procedure against fields explicitly requested."""
    required_fields = detect_required_fields(question)
    if not reranked_items:
        return False, required_fields, list(required_fields)

    candidate_gaps = []
    for item in reranked_items:
        has_content = bool(
            str(item.get("full_text", "")).strip()
            or str(item.get("text", "")).strip()
        )
        if not has_content:
            candidate_gaps.append(["content"])
            continue

        missing_fields = [
            field
            for field in required_fields
            if (
                field == "full_text"
                and not (
                    str(item.get("full_text", "")).strip()
                    or str(item.get("text", "")).strip()
                )
            )
            or (
                field != "full_text"
                and not str(item.get(field, "")).strip()
            )
        ]
        if not missing_fields:
            return True, required_fields, []
        candidate_gaps.append(missing_fields)

    best_gaps = min(candidate_gaps, key=len) if candidate_gaps else list(required_fields)
    return False, required_fields, best_gaps
