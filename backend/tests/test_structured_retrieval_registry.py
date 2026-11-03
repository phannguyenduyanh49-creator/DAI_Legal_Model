from backend.domain_registry import DomainRegistryService
from backend.rag.structured_retrieval import get_structured_retrieval_strategy


def test_configured_tourism_strategy_resolves_separately_from_generic_pipeline():
    registry = DomainRegistryService()
    config = registry.get_domain_config("tourism").retrieval
    handler = get_structured_retrieval_strategy(config.structured_lookup_strategy)

    assert handler is not None
    question = "Liệt kê các địa điểm du lịch ở miền Trung"
    assert handler.is_structured_query(question)
    assert handler.is_list_query(question)
    assert handler.select_records(question, []) == []


def test_unknown_or_unconfigured_domain_strategy_stays_semantic_only():
    registry = DomainRegistryService()
    for domain_id in ("education", "healthcare", "shopping", "food"):
        config = registry.get_domain_config(domain_id).retrieval
        assert get_structured_retrieval_strategy(config.structured_lookup_strategy) is None
    assert get_structured_retrieval_strategy("unreviewed_strategy") is None
