"""Safe provenance fields for API diagnostics and source display."""

from __future__ import annotations

from typing import Any

from backend.evidence.schemas import Evidence


def evidence_source_summary(evidence: Evidence) -> dict[str, Any]:
    """Return existing provenance without inferring missing source metadata."""
    summary: dict[str, Any] = {
        "evidence_id": evidence.evidence_id,
        "source_type": evidence.source_type,
        "source_id": evidence.source_id,
        "title": evidence.title,
        "source_url": evidence.source_url,
        "location": evidence.location,
    }
    metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
    domain_id = metadata.get("domain_id")
    if isinstance(domain_id, str) and domain_id.strip():
        summary["domain_id"] = domain_id
    return summary
