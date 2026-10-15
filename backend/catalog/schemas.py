from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CatalogQuery:
    domain: str
    intent: str

    formality_id: Optional[str] = None
    code: Optional[str] = None
    procedure_name: Optional[str] = None

    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CatalogSource:
    source_id: str
    source_type: str
    source_url: Optional[str] = None


@dataclass
class CatalogRecord:
    record_id: str
    code: str
    procedure_name: str

    target_type: Optional[str] = None
    category_name: Optional[str] = None

    description: Optional[str] = None
    execution_steps: Optional[str] = None
    execution_methods: Optional[str] = None
    profile_components: Optional[str] = None
    requirements_conditions: Optional[str] = None

    fees: Optional[str] = None
    legal_basis: Optional[str] = None
    results: Optional[str] = None

    target_objects: Optional[str] = None
    department_promulgate: Optional[str] = None
    executing_agencies: Optional[str] = None
    coordinating_agencies: Optional[str] = None

    source: Optional[CatalogSource] = None


@dataclass
class CatalogResult:
    matched: bool
    match_type: str
    confidence: float

    record: Optional[CatalogRecord] = None

    reason: Optional[str] = None