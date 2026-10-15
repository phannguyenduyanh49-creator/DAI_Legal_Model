from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Claim:
    claim: str

    evidence_ids: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class StructuredAnswer:
    answer: str

    claims: List[Claim] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )