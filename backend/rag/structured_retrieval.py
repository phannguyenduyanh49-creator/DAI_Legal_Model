"""Strategy lookup for opt-in domain-specific structured retrieval handlers."""

from dataclasses import dataclass
from typing import Any, Callable

from backend.tourism_retrieval import (
    is_tourism_multi_result_query,
    is_tourism_structured_query,
    select_tourism_records,
)


@dataclass(frozen=True)
class StructuredRetrievalStrategy:
    is_structured_query: Callable[[str], bool]
    is_list_query: Callable[[str], bool]
    select_records: Callable[[str, list[Any]], list[dict[str, Any]]]


_STRATEGIES = {
    "tourism_records": StructuredRetrievalStrategy(
        is_structured_query=is_tourism_structured_query,
        is_list_query=is_tourism_multi_result_query,
        select_records=select_tourism_records,
    ),
}


def get_structured_retrieval_strategy(
    strategy_id: str | None,
) -> StructuredRetrievalStrategy | None:
    """Resolve a configured strategy without guessing for unknown IDs."""
    if not strategy_id:
        return None
    return _STRATEGIES.get(strategy_id)
