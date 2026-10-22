from backend.rag.data_sufficiency import check_enough_data


def test_enough_data_requires_requested_fields():
    enough, required, missing = check_enough_data(
        "Hồ sơ thành lập cần giấy tờ gì?",
        [{"text": "Có nội dung thủ tục", "profile_components": "Đơn đăng ký"}],
    )

    assert enough is True
    assert required == ["profile_components"]
    assert missing == []


def test_enough_data_is_false_when_requested_field_is_missing():
    enough, required, missing = check_enough_data(
        "Hồ sơ thành lập cần giấy tờ gì?",
        [{"text": "Có nội dung thủ tục", "profile_components": " "}],
    )

    assert enough is False
    assert required == ["profile_components"]
    assert missing == ["profile_components"]


def test_enough_data_reports_all_required_fields_that_are_missing():
    enough, required, missing = check_enough_data(
        "Hồ sơ mất bao lâu và lệ phí bao nhiêu?",
        [{"full_text": "Có mô tả", "profile_components": "Đơn"}],
    )

    assert enough is False
    assert required == ["profile_components", "processing_time", "fees"]
    assert missing == ["processing_time", "fees"]


def test_enough_data_accepts_a_complete_candidate_even_if_another_is_incomplete():
    enough, _, missing = check_enough_data(
        "Hồ sơ cần giấy tờ gì?",
        [
            {"text": "Incomplete candidate"},
            {"text": "Complete candidate", "profile_components": "Đơn đăng ký"},
        ],
    )

    assert enough is True
    assert missing == []


def test_enough_data_is_false_when_no_local_candidate_exists():
    enough, required, missing = check_enough_data("Câu hỏi thủ tục", [])

    assert enough is False
    assert required == ["full_text"]
    assert missing == ["full_text"]
