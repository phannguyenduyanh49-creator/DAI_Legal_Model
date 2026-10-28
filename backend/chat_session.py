"""Small, testable helpers for the Core API's permission-confirmation session."""

import sqlite3
import uuid
from dataclasses import dataclass


PERMISSION_PROMPT_MARKER = (
    "Bạn có muốn mình sử dụng "
    "dữ liệu bên ngoài từ "
    "Cổng Dịch vụ công Quốc gia"
)

AFFIRMATIVE_REPLIES = frozenset({
    "co", "co nhe", "co a", "dong y", "ok", "oke", "okay", "yes",
    "duoc", "cho phep", "tra cuu di", "cu tra cuu",
})

NEGATIVE_REPLIES = frozenset({
    "khong", "khong can", "thoi", "khong tra cuu", "no",
})


@dataclass(frozen=True)
class PermissionResolution:
    question: str
    allow_external: bool
    decision: str


def normalize_session_id(value: str | None) -> str | None:
    """Accept only UUID session identifiers and return their canonical form."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return str(uuid.UUID(value.strip()))
    except (ValueError, AttributeError, TypeError):
        return None


def find_pending_external_question(db_path: str, session_id: str) -> str | None:
    """Read the original question only when the same session awaits permission."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE session_id = ? ORDER BY id DESC LIMIT 2
            """,
            (session_id,),
        ).fetchall()

    if len(rows) < 2:
        return None
    latest, previous = rows
    if (
        latest["role"] == "assistant"
        and PERMISSION_PROMPT_MARKER in str(latest["content"] or "")
        and previous["role"] == "user"
    ):
        return str(previous["content"] or "").strip() or None
    return None


def resolve_permission_reply(
    question: str,
    pending_question: str | None,
    allow_external: bool,
    normalized_reply: str,
) -> PermissionResolution:
    """Resume the saved legal question only for an explicit pending approval."""
    if not allow_external and pending_question:
        if normalized_reply in AFFIRMATIVE_REPLIES:
            return PermissionResolution(pending_question, True, "approved")
        if normalized_reply in NEGATIVE_REPLIES:
            return PermissionResolution(question, False, "denied")
    return PermissionResolution(question, allow_external, "none")
