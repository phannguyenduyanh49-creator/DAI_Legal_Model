"""Resolve presentation style without changing domain policy or verification."""

from __future__ import annotations

from typing import Literal


ResponseStyle = Literal["default", "strict", "friendly"]

_DEFAULT_STYLE_BY_DOMAIN = {
    "tourism": "friendly",
}


def resolve_response_style(requested: str, domain_id: str | None) -> str:
    """Resolve an explicit override or choose the domain's safe default.

    Unknown domains use strict presentation. This affects prompt wording only;
    domain policy, Evidence requirements, and verifier behavior stay separate.
    """
    if requested in {"strict", "friendly"}:
        return requested
    return _DEFAULT_STYLE_BY_DOMAIN.get(str(domain_id or "").strip().lower(), "strict")
