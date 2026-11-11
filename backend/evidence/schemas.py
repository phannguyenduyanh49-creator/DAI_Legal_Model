from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    evidence_id: str
    source_type: str

    source_id: Optional[str] = None
    source_url: Optional[str] = None

    title: Optional[str] = None
    content: str = ""

    location: Optional[str] = None

    score: Optional[float] = None

    validity_from: Optional[str] = None
    validity_to: Optional[str] = None

    # Các trường dữ liệu có trong evidence
    fields: Dict[str, Any] = field(
        default_factory=dict
    )

    # Trường nào có dữ liệu / bị thiếu
    completeness: Dict[str, bool] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class EvidenceBundle:
    query: Optional[str] = None

    evidences: List[Evidence] = field(
        default_factory=list
    )

    total: int = 0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def get_evidence_by_id(self, identifier: str) -> Optional[Evidence]:
        """Resolve an evidence item by its evidence_id, short_id alias, canonical_id, or source_id."""
        if not identifier:
            return None
        clean_id = str(identifier).strip()
        # 1. Exact match on evidence_id
        for ev in self.evidences:
            if ev.evidence_id == clean_id:
                return ev
        # 2. Match on short_id metadata
        for ev in self.evidences:
            if ev.metadata.get("short_id") == clean_id:
                return ev
        # 3. Match on canonical_id metadata
        for ev in self.evidences:
            if ev.metadata.get("canonical_id") == clean_id:
                return ev
        # 4. Match on original_evidence_id
        for ev in self.evidences:
            if ev.metadata.get("original_evidence_id") == clean_id:
                return ev
        # 5. Match on source_id
        for ev in self.evidences:
            if ev.source_id and str(ev.source_id).strip() == clean_id:
                return ev
        return None

    def short_id_map(self) -> Dict[str, str]:
        """Return deterministic mapping of short_id -> canonical evidence_id."""
        mapping: Dict[str, str] = {}
        for i, ev in enumerate(self.evidences, start=1):
            short_id = ev.metadata.get("short_id") or f"E{i}"
            mapping[short_id] = ev.evidence_id
        return mapping

    def resolve_evidence_id(self, identifier: str) -> str:
        """Resolve any valid alias or short ID to the canonical evidence_id."""
        ev = self.get_evidence_by_id(identifier)
        if ev is not None:
            return ev.evidence_id
        return str(identifier).strip()
