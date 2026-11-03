"""Domain-scoped structured selection for destination list/filter questions."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


MAX_TOURISM_STRUCTURED_RECORDS = 500
MAX_TOURISM_LIST_RESULTS = 10


def normalize_tourism_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFD", str(value or "").lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.replace("đ", "d").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9-]+", " ", normalized)).strip()


def is_tourism_multi_result_query(question: str) -> bool:
    query = normalize_tourism_text(question)
    asks_for_places = any(term in query for term in ("dia diem", "diem tham quan", "diem du lich", "destination"))
    asks_for_multiple = any(term in query for term in ("cac ", "liet ke", "tim cac", "so sanh", "nhung "))
    return asks_for_places and asks_for_multiple


def is_tourism_structured_query(question: str) -> bool:
    return is_tourism_multi_result_query(question) or _requested_duration(question) is not None


def _record_fields(payload: dict[str, Any]) -> dict[str, str]:
    fields = payload.get("record_fields")
    if isinstance(fields, dict):
        return {normalize_tourism_text(key).replace(" ", "_"): str(value or "").strip() for key, value in fields.items()}

    text = str(payload.get("text") or payload.get("full_text") or "")
    recovered = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() and value.strip():
            recovered[normalize_tourism_text(key).replace(" ", "_")] = value.strip()
    return recovered


def _hours_range(value: str) -> tuple[float, float] | None:
    normalized = normalize_tourism_text(value).replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|den|to)\s*(\d+(?:\.\d+)?)\s*(?:gio|h|hours?)", normalized)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        return (min(low, high), max(low, high))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:gio|h|hours?)", normalized)
    if match:
        number = float(match.group(1))
        return (number, number)
    if "nua ngay" in normalized:
        return (0.0, 4.0)
    return None


def _requested_duration(question: str) -> tuple[str, float, float] | None:
    query = normalize_tourism_text(question)
    if "nua ngay" in query:
        return ("half_day", 0.0, 4.0)
    if "duoi 4 gio" in query or "under 4 hours" in query:
        return ("under_4_hours", 0.0, 4.0)
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|den|to)\s*(\d+(?:\.\d+)?)\s*(?:gio|h|hours?)", query)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        return ("range", min(low, high), max(low, high))
    return None


def select_tourism_records(question: str, points: list[Any]) -> list[dict[str, Any]]:
    """Select supported matching point payloads; never infer missing attributes."""
    query = normalize_tourism_text(question)
    region_requested = "mien trung" if "mien trung" in query else None
    duration_requested = _requested_duration(question)
    list_query = is_tourism_multi_result_query(question)
    if not list_query and not duration_requested:
        return []

    selected = []
    seen = set()
    for point in points:
        payload = getattr(point, "payload", None) or {}
        if payload.get("domain_id") != "tourism" or payload.get("source_type") != "domain_upload":
            continue
        fields = _record_fields(payload)
        region = str(payload.get("tourism_region") or fields.get("khu_vuc") or fields.get("region") or "")
        province = str(payload.get("tourism_province_city") or fields.get("tinh_thanh_pho") or fields.get("tinh_thanh") or fields.get("province_city") or "")
        duration = str(payload.get("tourism_visit_duration") or fields.get("thoi_gian_tham_quan_goi_y") or fields.get("thoi_gian_tham_quan") or fields.get("visit_duration") or "")
        if region_requested and region:
            if "mien trung" not in normalize_tourism_text(region) and "trung bo" not in normalize_tourism_text(region):
                continue
        elif region_requested and province:
            # A city name alone does not establish a region; defer to the stored
            # region field instead of guessing geographic membership.
            continue
        elif region_requested:
            continue

        if duration_requested:
            duration_range = _hours_range(duration)
            kind, requested_low, requested_high = duration_requested
            if kind == "half_day":
                if not duration_range or duration_range[0] > requested_high:
                    continue
            elif kind == "under_4_hours":
                if not duration_range or duration_range[0] >= requested_high:
                    continue
            elif (
                not duration_range
                or duration_range[1] < requested_low
                or duration_range[0] > requested_high
            ):
                continue

        evidence_id = str(payload.get("evidence_id") or "")
        row_key = (payload.get("document_id"), payload.get("sheet_name"), payload.get("row_number"))
        key = row_key if all(value is not None for value in row_key) else evidence_id
        if not key or key in seen:
            continue
        seen.add(key)
        record_name = str(
            payload.get("record_name")
            or fields.get("ten_dia_diem")
            or fields.get("ten_diem_den")
            or fields.get("ten_dia_danh")
            or fields.get("destination_name")
            or payload.get("title")
            or "Tourism record"
        )
        selected.append({
            "procedure_name": record_name,
            "text": str(payload.get("text") or payload.get("full_text") or "").strip(),
            "full_text": str(payload.get("full_text") or payload.get("text") or "").strip(),
            "score": 0.0,
            "source_type": "domain_upload",
            "domain_id": "tourism",
            "source_id": payload.get("source_id"),
            "document_id": payload.get("document_id"),
            "record_id": payload.get("record_id") or fields.get("record_id"),
            "point_id": getattr(point, "id", None),
            "chunk_index": payload.get("chunk_index"),
            "evidence_id": evidence_id,
            "title": record_name,
            "location": payload.get("location"),
            "record_fields": fields,
            "province_city": province,
            "region": region,
            "duration": duration,
            "sheet_name": payload.get("sheet_name"),
            "row_number": payload.get("row_number"),
        })
    return selected[:MAX_TOURISM_STRUCTURED_RECORDS]
