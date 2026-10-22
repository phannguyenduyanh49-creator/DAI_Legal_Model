import re
from typing import Any, Optional

from backend.evidence.schemas import Evidence


DVC_SOURCE_URL = "https://dichvucong.gov.vn"
DVC_NO_RESULT_PREFIX = "DVC_NO_RELEVANT_RESULT:"
_ENVELOPE_RE = re.compile(r"^DVC_OFFICIAL_PROCEDURE\s*$", re.MULTILINE)


def mcp_result_status(result: Any) -> str:
    text = str(result or "").strip()
    if text.startswith(DVC_NO_RESULT_PREFIX):
        return "no_match"
    if (
        _ENVELOPE_RE.search(text)
        and _field(text, "DVC_PROCEDURE_ID")
        and re.search(
            r"^DVC_DETAIL_BEGIN\s*\n(.+?)\nDVC_DETAIL_END\s*$",
            text,
            re.DOTALL | re.MULTILINE,
        )
    ):
        return "completed"
    return "invalid_response"


def _field(text: str, name: str) -> Optional[str]:
    match = re.search(rf"^{re.escape(name)}:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def mcp_dvc_result_to_evidence(result: Any) -> Optional[Evidence]:
    text = str(result or "").strip()
    if mcp_result_status(text) != "completed":
        return None

    procedure_id = _field(text, "DVC_PROCEDURE_ID")
    if not procedure_id:
        return None

    detail_match = re.search(
        r"^DVC_DETAIL_BEGIN\s*\n(.*?)\nDVC_DETAIL_END\s*$",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if not detail_match or not detail_match.group(1).strip():
        return None

    procedure_name = _field(text, "DVC_PROCEDURE_NAME")
    procedure_code = _field(text, "DVC_PROCEDURE_CODE")
    source_url = _field(text, "DVC_SOURCE_URL") or DVC_SOURCE_URL
    if not source_url.startswith("https://dichvucong.gov.vn/") and source_url != DVC_SOURCE_URL:
        return None

    return Evidence(
        evidence_id=f"mcp:dvc:{procedure_id}",
        source_type="mcp",
        source_id=procedure_id,
        source_url=source_url,
        title=procedure_name,
        content=detail_match.group(1).strip(),
        score=1.0,
        metadata={
            "provider": "dichvucong.gov.vn",
            "tool": "search_dichvucong",
            "procedure_id": procedure_id,
            "procedure_code": procedure_code,
            "procedure_name": procedure_name,
            "retrieval_type": "external_official_source",
        },
    )


def mcp_results_to_evidence(results: list[Any]) -> list[Evidence]:
    evidences = []
    for result in results:
        evidence = mcp_dvc_result_to_evidence(result)
        if evidence is not None:
            evidences.append(evidence)
    return evidences
