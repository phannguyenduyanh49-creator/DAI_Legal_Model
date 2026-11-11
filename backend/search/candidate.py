"""Read a candidate's real display name across retrieval stores."""

from typing import Any


def candidate_name(payload: dict[str, Any]) -> str:
    """Use source metadata only; never derive a title from answer text."""
    return str(
        payload.get("thu_tuc")
        or payload.get("procedure_name")
        or payload.get("title")
        or ""
    ).strip()
