import pytest

from backend.domain_documents import retrieval_collections_for_domain
from backend.domain_registry import DomainRegistryService
from backend.orchestrator import DomainRouterPlanner
from backend.policy.middleware import PolicyEnforcementMiddleware
from backend.query_analysis import AnalysisResult


def test_registry_keeps_public_domain_ids_unique_and_rejects_duplicate_registration():
    registry = DomainRegistryService()
    existing = registry.get_domain_config("legal_ai")

    with pytest.raises(ValueError, match="already registered"):
        registry.register_domain(existing)

    assert registry.list_all_domains() == [
        "legal_ai",
        "general_assistant",
        "administrative_law",
        "tourism",
    ]


def test_registry_rejects_route_alias_that_conflicts_with_registered_domain_id():
    registry = DomainRegistryService()
    conflicting = registry.get_domain_config("tourism").model_copy(
        update={"domain_id": "new_domain"}
    )
    with pytest.raises(ValueError):
        registry.register_domain(conflicting)
    assert "new_domain" not in registry.list_all_domains()


def test_policy_update_normalizes_domain_id_without_creating_an_orphan_key():
    registry = DomainRegistryService()

    updated = registry.update_domain_policy(" LEGAL_AI ", new_mode="FRIENDLY")

    assert updated.domain_id == "legal_ai"
    assert registry.list_all_domains().count("legal_ai") == 1
    assert registry.get_domain_config("legal_ai").policy.mode == "FRIENDLY"


def test_general_domain_does_not_advertise_unimplemented_mcp_tools():
    registry = DomainRegistryService()

    assert registry.get_domain_config(
        "general_assistant"
    ).policy.allowed_mcp_tools == []
    assert PolicyEnforcementMiddleware(registry).apply(
        "general_assistant"
    ).allowed_mcp_tools == ()


def test_tourism_auto_route_uses_only_tourism_retrieval_targets():
    query = "Which shuttle leaves the hotel for the tourist attraction?"
    analysis = AnalysisResult(
        original_query=query,
        cleaned_query=query,
        confidence_score=0.9,
    )
    registry = DomainRegistryService()

    routing = DomainRouterPlanner().process(analysis)
    policy = PolicyEnforcementMiddleware(registry).apply(routing.selected_domain)
    available_collections = {
        "legal_docs_bge_m3",
        "domain_documents_tourism",
        "domain_documents_legal_ai",
    }

    targets = retrieval_collections_for_domain(
        policy.domain_config.domain_id,
        available_collections.__contains__,
        registry,
    )

    assert routing.selected_domain == "tourism"
    assert policy.domain_config.domain_id == "tourism"
    assert targets == ["domain_documents_tourism"]


def test_unmatched_general_query_does_not_fall_back_to_legal_sources():
    query = "What is the capital of Japan?"
    analysis = AnalysisResult(
        original_query=query,
        cleaned_query=query,
        confidence_score=0.9,
    )
    registry = DomainRegistryService()
    routing = DomainRouterPlanner().process(analysis)
    policy = PolicyEnforcementMiddleware(registry).apply(routing.selected_domain)

    targets = retrieval_collections_for_domain(
        policy.domain_config.domain_id,
        lambda name: name in {"legal_docs_bge_m3", "domain_documents_legal_ai"},
        registry,
    )

    assert routing.selected_domain == "general_assistant"
    assert policy.domain_config.domain_id == "general_assistant"
    assert targets == []


@pytest.mark.parametrize("query,expected", [
    ("Thủ tục thành lập công ty cổ phần là gì?", "enterprise_law"),
    ("Hồ sơ đăng ký thành lập doanh nghiệp cần những gì?", "enterprise_law"),
    ("Nghĩa vụ pháp luật áp dụng trong trường hợp này là gì?", "general_legal"),
    ("Thủ tục hành chính để xin giấy phép được xử lý thế nào?", "administrative_law"),
    ("Can you explain the law currently in force?", "general_legal"),
    ("What business strategy should a new cafe use?", "general_assistant"),
])
def test_registry_routes_legal_variants_and_ordinary_business(query, expected):
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)
    route = DomainRouterPlanner().process(analysis)
    assert route.selected_domain == expected
    assert route.routing_reason in {"REGISTRY_KEYWORD_RULE", "NO_DOMAIN_RULE_MATCH"}
    assert not route.clarification_required


def test_stale_tourism_hint_does_not_override_clear_legal_question():
    query = "Thủ tục thành lập công ty cổ phần là gì?"
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)
    route = DomainRouterPlanner().process(analysis, domain_hint="tourism")
    assert route.selected_domain == "enterprise_law"
    assert route.routing_reason == "CONTENT_OVERRIDES_STALE_HINT"
    assert PolicyEnforcementMiddleware(DomainRegistryService()).apply(
        route.selected_domain
    ).domain_config.domain_id == "legal_ai"
    policy = PolicyEnforcementMiddleware(DomainRegistryService()).apply(route.selected_domain)
    collections = retrieval_collections_for_domain(
        policy.domain_config.domain_id,
        {"legal_docs_bge_m3", "domain_documents_legal_ai"}.__contains__,
    )
    assert collections == ["legal_docs_bge_m3", "domain_documents_legal_ai"]


def test_stale_legal_hint_does_not_override_clear_tourism_question():
    query = "Tôi muốn đi du lịch và tìm khách sạn gần địa điểm tham quan."
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)
    route = DomainRouterPlanner().process(analysis, domain_hint="enterprise_law")
    assert route.selected_domain == "tourism"
    assert route.routing_reason == "CONTENT_OVERRIDES_STALE_HINT"


def test_mixed_legal_and_tourism_signals_request_clarification():
    query = "Thủ tục thành lập công ty cổ phần để mở khách sạn du lịch là gì?"
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)
    route = DomainRouterPlanner().process(analysis)
    assert route.selected_domain is None
    assert route.clarification_required
    assert route.routing_reason == "AMBIGUOUS_DOMAIN_SIGNALS"
    assert route.execution_plan == []


def test_disabled_domain_routes_and_retrieval_are_not_available():
    registry = DomainRegistryService()
    config = registry.get_domain_config("tourism")
    registry._registry["tourism"] = config.model_copy(update={"enabled": False})
    with pytest.raises(KeyError, match="disabled"):
        registry.route_domain_id("tourism")
    assert all(owner != "tourism" for owner, _rule in registry.iter_routing_rules())
    assert retrieval_collections_for_domain("tourism", lambda _name: True, registry) == []
    disabled_status = next(
        item for item in registry.readiness().domains if item.domain_id == "tourism"
    )
    assert not disabled_status.enabled
    assert disabled_status.retrieval_ready == "unavailable"
    assert disabled_status.data_available == "unknown"
    assert disabled_status.e2e_verified == "not_verified"


def test_readiness_distinguishes_registry_configuration_from_runtime_probes():
    readiness = DomainRegistryService().readiness()
    legal = next(item for item in readiness.domains if item.domain_id == "legal_ai")
    assert legal.registered and legal.enabled and legal.configured
    assert legal.data_available == "unknown"
    assert legal.retrieval_ready == "unknown"
    assert legal.model_ready == "unknown"
    assert legal.e2e_verified == "not_verified"
    assert legal.response_policy == "evidence_required"


def test_domain_api_adds_readiness_without_changing_legacy_domain_list():
    from fastapi.testclient import TestClient
    from backend.core_gateway import app

    with TestClient(app) as client:
        domains = client.get("/api/v2/domains/")
        readiness = client.get("/api/v2/domains/readiness")

    assert domains.status_code == 200
    assert domains.json() == ["legal_ai", "general_assistant", "administrative_law", "tourism"]
    assert readiness.status_code == 200
    legal = next(item for item in readiness.json()["domains"] if item["domain_id"] == "legal_ai")
    assert legal["data_available"] == "unknown"
    assert legal["retrieval_ready"] == "unknown"
    assert legal["model_ready"] == "unknown"
    assert legal["e2e_verified"] == "not_verified"
