from types import SimpleNamespace

from backend.evidence.builder import EvidenceBuilder
from backend.rag.adapter import rag_result_to_evidence
from backend.tourism_retrieval import (
    is_tourism_multi_result_query,
    is_tourism_structured_query,
    select_tourism_records,
)


def point(name, region, duration, *, domain="tourism", evidence_id=None, extra=None):
    fields = {
        "ten_dia_diem": name,
        "tinh_thanh_pho": "Quảng Nam" if region == "Miền Trung" else "Hà Nội",
        "khu_vuc": region,
        "thoi_gian_tham_quan_goi_y": duration,
        "tinh_trang_du_lieu": "Giá vé chưa xác minh; tọa độ chưa xác minh.",
    }
    if extra:
        fields.update(extra)
    return SimpleNamespace(payload={
        "domain_id": domain,
        "source_type": "domain_upload",
        "document_id": f"doc-{name}",
        "evidence_id": evidence_id or f"domain:{domain}:doc-{name}:1",
        "source_id": f"doc-{name}",
        "title": "Destinations.xlsx",
        "record_name": name,
        "record_fields": fields,
        "tourism_region": region,
        "tourism_visit_duration": duration,
        "text": "\n".join(f"{key}: {value}" for key, value in fields.items()),
        "full_text": "\n".join(f"{key}: {value}" for key, value in fields.items()),
    })


DESTINATIONS = [
    point("Bà Nà Hills", "Miền Trung", "Nửa ngày đến 1 ngày"),
    point("Hội An", "Miền Trung", "2–4 giờ"),
    point("Mỹ Khê", "Miền Trung", "3–6 giờ"),
    point("Đại Nội Huế", "Miền Trung", "3–5 giờ"),
    point("Phong Nha – Kẻ Bàng", "Miền Trung", "1 ngày"),
    point("Hồ Gươm", "Miền Bắc", "2 giờ"),
    point("Legal record", "Miền Trung", "2 giờ", domain="legal_ai"),
]


def test_central_vietnam_list_filter_returns_all_matching_records_and_excludes_other_domains():
    question = "Các địa điểm du lịch ở miền Trung?"
    assert is_tourism_multi_result_query(question)
    results = select_tourism_records(question, DESTINATIONS)
    assert [item["procedure_name"] for item in results] == [
        "Bà Nà Hills", "Hội An", "Mỹ Khê", "Đại Nội Huế", "Phong Nha – Kẻ Bàng",
    ]
    assert all(item["domain_id"] == "tourism" for item in results)
    assert all(item["evidence_id"].startswith("domain:tourism:") for item in results)


def test_half_day_duration_filter_returns_supported_ranges_without_claiming_exact_time():
    question = "Tìm các địa điểm có thể tham quan trong nửa ngày?"
    assert is_tourism_structured_query(question)
    results = select_tourism_records(question, DESTINATIONS)
    names = [item["procedure_name"] for item in results]
    assert names == ["Bà Nà Hills", "Hội An", "Mỹ Khê", "Đại Nội Huế", "Hồ Gươm"]
    assert next(item for item in results if item["procedure_name"] == "Hội An")["duration"] == "2–4 giờ"
    assert next(item for item in results if item["procedure_name"] == "Mỹ Khê")["duration"] == "3–6 giờ"
    assert "Phong Nha – Kẻ Bàng" not in names


def test_under_four_hours_matches_overlapping_ranges_but_not_unknown_or_full_day_values():
    results = select_tourism_records(
        "Tìm các địa điểm tham quan dưới 4 giờ?",
        DESTINATIONS,
    )
    names = [item["procedure_name"] for item in results]
    assert "Hội An" in names
    assert "Mỹ Khê" in names
    assert "Phong Nha – Kẻ Bàng" not in names


def test_explicit_two_to_four_hour_request_uses_interval_overlap():
    results = select_tourism_records(
        "Các địa điểm tham quan trong 2–4 giờ",
        DESTINATIONS,
    )
    names = [item["procedure_name"] for item in results]
    assert "Hội An" in names
    assert "Mỹ Khê" in names
    assert "Phong Nha – Kẻ Bàng" not in names


def test_missing_prices_and_unverified_coordinates_are_not_filled_in():
    results = select_tourism_records("Các địa điểm du lịch ở miền Trung?", DESTINATIONS)
    hoi_an = next(item for item in results if item["procedure_name"] == "Hội An")
    assert "gia_ve_nguoi_lon_vnd" not in hoi_an["record_fields"]
    assert "tọa độ" not in " ".join(hoi_an["record_fields"].keys()).lower()
    assert "chưa xác minh" in hoi_an["record_fields"]["tinh_trang_du_lieu"]


def test_region_and_duration_filters_do_not_guess_when_required_fields_are_missing():
    incomplete = point("Unknown Region", "", "")
    incomplete.payload.pop("record_fields")
    assert select_tourism_records("Các địa điểm du lịch ở miền Trung?", [incomplete]) == []
    assert select_tourism_records("Tìm các địa điểm có thể tham quan trong nửa ngày?", [incomplete]) == []


def test_old_index_text_can_supply_record_fields_without_leaking_into_legal_scope():
    legacy = SimpleNamespace(payload={
        "domain_id": "tourism",
        "source_type": "domain_upload",
        "document_id": "old-doc",
        "evidence_id": "domain:tourism:old-doc:1",
        "text": "[Sheet: Sheet1 | Row: 2]\nTên địa điểm: Hội An\nKhu vực: Miền Trung\nThời gian tham quan gợi ý: 2–4 giờ",
    })
    selected = select_tourism_records("Các địa điểm du lịch ở miền Trung?", [legacy])
    assert len(selected) == 1
    assert selected[0]["record_fields"]["ten_dia_diem"] == "Hội An"
    legacy.payload["domain_id"] = "legal_ai"
    assert select_tourism_records("Các địa điểm du lịch ở miền Trung?", [legacy]) == []


def test_selected_tourism_record_becomes_citable_evidence_with_real_source_metadata():
    selected = select_tourism_records("Các địa điểm du lịch ở miền Trung?", DESTINATIONS)[0]
    selected["rerank_score"] = 0.91
    evidence = rag_result_to_evidence(selected)
    bundle = EvidenceBuilder(min_score=0.3).build(rag_evidences=[evidence])

    assert bundle.total == 1
    assert bundle.evidences[0].evidence_id == selected["evidence_id"]
    assert bundle.evidences[0].title == "Bà Nà Hills"
    assert "Nửa ngày đến 1 ngày" in bundle.evidences[0].content
    assert bundle.evidences[0].source_id == selected["source_id"]
    assert bundle.evidences[0].metadata["document_id"] == selected["document_id"]
