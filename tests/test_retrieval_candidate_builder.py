from evals.build_retrieval_candidates import build_rows


def source(domain, document_id, record_id, record_fields, *, title="", content="", source_type="domain_upload"):
    return {
        "document_key": f"{domain}:{document_id}",
        "domain_id": domain,
        "source_id": f"source-{document_id}",
        "document_id": document_id,
        "record_id": record_id,
        "title": title,
        "content": content,
        "source_type": source_type,
        "metadata": {"record_fields": record_fields},
    }


def test_candidate_builder_never_promotes_automatic_suggestions_to_gold():
    rows = build_rows([
        source("healthcare", "hc-f", "HC-F-001", {
            "loai_ban_ghi": "facility_service_mock",
            "chuyen_khoa": "Da liễu", "quan_huyen": "Quận 1",
        }),
        source("healthcare", "hc-k", "HC-K-001", {
            "loai_ban_ghi": "health_knowledge", "chu_de": "Rửa tay",
        }),
    ])

    assert len(rows) == 100
    assert all(row["human_approved"] is False for row in rows)
    assert all(row["review_status"] == "unreviewed" for row in rows)
    assert all(row["source_provenance"]["references"] == [] for row in rows)
    assert all(row["relevant_document_ids"] == [] for row in rows)
    assert all(row["relevance_labels"] == {} for row in rows)
    hc_f = next(row for row in rows if row["expected_domain"] == "healthcare" and row["expected_record_type"] == "HC-F" and row["query"])
    hc_k = next(row for row in rows if row["expected_domain"] == "healthcare" and row["expected_record_type"] == "HC-K" and row["query"])
    assert hc_f["expected_record_type"] == "HC-F"
    assert "Da liễu" in hc_f["query"]
    assert hc_k["expected_record_type"] == "HC-K"
    assert "Rửa tay" in hc_k["query"]
    assert hc_f["auto_candidate_judgments"][0]["suggestion_status"] == "automatic_unreviewed"


def test_food_slots_are_blocked_when_no_real_source_record_exists():
    rows = build_rows([])
    food_rows = [row for row in rows if row["expected_domain"] == "food"]

    assert len(food_rows) == 5
    assert all(row["query"] is None for row in food_rows)
    assert all(row["annotation_status"] == "blocked_no_source_data" for row in food_rows)
    assert all(row["auto_candidate_judgments"] == [] for row in food_rows)
