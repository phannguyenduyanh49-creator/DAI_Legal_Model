import re
from typing import Any, Optional

from backend.evidence.schemas import Evidence


DVC_SOURCE_URL = "https://dichvucong.gov.vn"


def _extract_field(
    text: str,
    field_name: str,
) -> Optional[str]:
    pattern = rf"{re.escape(field_name)}:\s*(.+)"

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    value = match.group(1).strip()

    if not value:
        return None

    return value


def _extract_detail_content(
    text: str,
) -> str:
    marker = "DỮ LIỆU LẤY TRỰC TIẾP TỪ DVC API:"

    if marker not in text:
        return text.strip()

    content = text.split(
        marker,
        1,
    )[1]

    instruction_marker = "CHỈ DÙNG DỮ LIỆU TRÊN ĐỂ TRẢ LỜI."

    if instruction_marker in content:
        content = content.split(
            instruction_marker,
            1,
        )[0]

    return content.strip()


def mcp_dvc_result_to_evidence(
    result: Any,
) -> Optional[Evidence]:

    if result is None:
        return None

    text = str(result).strip()

    if not text:
        return None

    invalid_prefixes = (
        "Câu hỏi không được xác định",
        "Không xác định được",
        "Không tìm thấy",
        "Lỗi",
    )

    if text.startswith(invalid_prefixes):
        return None

    procedure_name = _extract_field(
        text,
        "Tên thủ tục",
    )

    procedure_code = _extract_field(
        text,
        "Mã thủ tục",
    )

    procedure_id = _extract_field(
        text,
        "ID",
    )

    detail_content = _extract_detail_content(text)

    if not detail_content:
        return None

    evidence_key = (
        procedure_id
        or procedure_code
        or procedure_name
        or "unknown"
    )

    return Evidence(
        evidence_id=f"mcp:dvc:{evidence_key}",
        source_type="mcp",
        source_id=procedure_id or procedure_code,
        source_url=DVC_SOURCE_URL,
        title=procedure_name,
        content=detail_content,
        score=1.0,
        validity_from=None,
        validity_to=None,
        metadata={
            "provider": "dichvucong.gov.vn",
            "tool": "search_dichvucong",
            "procedure_id": procedure_id,
            "procedure_code": procedure_code,
            "procedure_name": procedure_name,
            "retrieval_type": "external_official_source",
        },
    )


def mcp_results_to_evidence(
    results: list[Any],
) -> list[Evidence]:

    evidences = []

    for result in results:
        evidence = mcp_dvc_result_to_evidence(
            result
        )

        if evidence is not None:
            evidences.append(evidence)

    return evidences