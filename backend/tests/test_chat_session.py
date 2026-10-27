import sqlite3

from backend.chat_session import (
    PERMISSION_PROMPT_MARKER,
    find_pending_external_question,
    normalize_session_id,
    resolve_permission_reply,
)


def _make_messages_db(path, session_id, messages):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)"
        )
        connection.executemany(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, role, content) for role, content in messages],
        )


def test_approved_permission_resumes_original_question_for_mcp(tmp_path):
    db_path = str(tmp_path / "chat.db")
    session_id = "session-123"
    original_question = "Thủ tục đăng ký tạm trú cần giấy tờ gì?"
    _make_messages_db(
        db_path,
        session_id,
        [
            ("user", original_question),
            ("assistant", f"Chưa đủ dữ liệu. {PERMISSION_PROMPT_MARKER} tra cứu thêm không?"),
        ],
    )

    pending = find_pending_external_question(db_path, session_id)
    resolution = resolve_permission_reply("có", pending, False, "co")
    mcp_questions = []

    def fake_mcp(query):
        mcp_questions.append(query)

    if resolution.allow_external:
        fake_mcp(resolution.question)

    assert pending == original_question
    assert resolution.decision == "approved"
    assert resolution.allow_external is True
    assert mcp_questions == [original_question]


def test_pending_question_is_isolated_by_session_id(tmp_path):
    db_path = str(tmp_path / "chat.db")
    _make_messages_db(
        db_path,
        "session-with-pending",
        [
            ("user", "Original legal question"),
            ("assistant", PERMISSION_PROMPT_MARKER),
        ],
    )

    assert find_pending_external_question(db_path, "different-session") is None


def test_affirmative_reply_without_valid_pending_question_does_not_enable_mcp():
    resolution = resolve_permission_reply("có", None, False, "co")
    mcp_questions = []

    if resolution.allow_external:
        mcp_questions.append(resolution.question)

    assert resolution.question == "có"
    assert resolution.allow_external is False
    assert resolution.decision == "none"
    assert mcp_questions == []


def test_non_permission_history_does_not_restore_a_question(tmp_path):
    db_path = str(tmp_path / "chat.db")
    _make_messages_db(
        db_path,
        "session-123",
        [
            ("user", "Earlier question"),
            ("assistant", "A normal answer without a permission prompt."),
        ],
    )

    assert find_pending_external_question(db_path, "session-123") is None


def test_denied_permission_does_not_enable_external_lookup():
    resolution = resolve_permission_reply("Không", "Câu hỏi ban đầu", False, "khong")

    assert resolution.decision == "denied"
    assert resolution.allow_external is False
    assert resolution.question == "Không"


def test_affirmative_reply_without_pending_question_stays_a_normal_query():
    resolution = resolve_permission_reply("Có", None, False, "co")

    assert resolution.decision == "none"
    assert resolution.allow_external is False
    assert resolution.question == "Có"


def test_pending_question_requires_the_latest_assistant_message_to_be_permission_prompt(tmp_path):
    db_path = str(tmp_path / "chat.db")
    _make_messages_db(
        db_path,
        "session-123",
        [
            ("user", "Original legal question"),
            ("assistant", PERMISSION_PROMPT_MARKER),
            ("user", "A later unrelated message"),
        ],
    )

    assert find_pending_external_question(db_path, "session-123") is None


def test_session_id_validation_rejects_invalid_cookie_values():
    assert normalize_session_id("not-a-uuid") is None
    assert normalize_session_id(None) is None


def test_session_id_validation_canonicalizes_valid_uuid():
    assert normalize_session_id("12345678-1234-5678-1234-567812345678") == (
        "12345678-1234-5678-1234-567812345678"
    )
