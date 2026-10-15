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