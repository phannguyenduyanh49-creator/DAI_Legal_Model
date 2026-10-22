"""Regression tests for domain routing and registry normalization."""
from types import SimpleNamespace

from backend.domain_registry import DomainRegistryService
from backend.orchestrator import DomainRouterPlanner


def analysis(query: str):
    return SimpleNamespace(
        cleaned_query=query,
        expanded_terms=[],
        extracted_citations=[],
        hypothetical_document=None,
    )


def test_legacy_hints_map_to_registered_domain_ids():
    router = DomainRouterPlanner()
    for hint in ("civil_law", "enterprise_law", "criminal_law",
                 "labor_law", "land_law", "general_legal"):
        selected, _ = router.route_domain(analysis("câu hỏi bất kỳ"), hint)
        assert selected == "legal_ai"


def test_administrative_hint_stays_separate():
    selected, confidence = DomainRouterPlanner().route_domain(
        analysis("câu hỏi"), "administrative_law"
    )
    assert selected == "administrative_law"
    assert confidence == 1.0


def test_unknown_hint_falls_back_to_content_routing():
    selected, _ = DomainRouterPlanner().route_domain(
        analysis("câu hỏi không có từ khóa chuyên ngành"), "unknown_domain"
    )
    assert selected == "legal_ai"


def test_no_keyword_match_uses_legal_ai():
    selected, confidence = DomainRouterPlanner().route_domain(
        analysis("xyz không khớp từ khóa")
    )
    assert selected == "legal_ai"
    assert confidence == 0.50


def test_administrative_keywords_route_to_administrative_law():
    selected, _ = DomainRouterPlanner().route_domain(
        analysis("Tôi cần làm thủ tục hành chính xin giấy phép")
    )
    assert selected == "administrative_law"


def test_registry_lookup_normalizes_id():
    registry = DomainRegistryService()
    assert registry.get_domain_config(" LEGAL_AI ").domain_id == "legal_ai"


def test_registry_policy_update_normalizes_id():
    registry = DomainRegistryService()
    updated = registry.update_domain_policy(" LEGAL_AI ", new_mode="FRIENDLY")
    assert updated.policy.mode == "FRIENDLY"
    assert registry.get_domain_config("legal_ai").policy.mode == "FRIENDLY"
