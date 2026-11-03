import pytest

from backend.domain_documents import retrieval_collections_for_domain
from backend.domain_registry import DomainRegistryService, DomainRetrievalConfig
from backend.orchestrator import DomainRouterPlanner
from backend.policy.middleware import PolicyEnforcementMiddleware
from backend.query_analysis import AnalysisResult, QueryAnalysisService


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
        "healthcare",
        "education",
        "shopping",
        "food",
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


def test_explicit_tourism_domain_selection_overrides_clear_legal_question():
    query = "Thủ tục thành lập công ty cổ phần là gì?"
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)
    route = DomainRouterPlanner().process(analysis, domain_hint="tourism")
    assert route.selected_domain == "tourism"
    assert route.routing_reason == "DOMAIN_HINT"
    assert PolicyEnforcementMiddleware(DomainRegistryService()).apply(
        route.selected_domain
    ).domain_config.domain_id == "tourism"
    policy = PolicyEnforcementMiddleware(DomainRegistryService()).apply(route.selected_domain)
    collections = retrieval_collections_for_domain(
        policy.domain_config.domain_id,
        {"legal_docs_bge_m3", "domain_documents_tourism"}.__contains__,
    )
    assert collections == ["domain_documents_tourism"]


@pytest.mark.parametrize(
    "domain_id",
    [
        "legal_ai", "administrative_law", "tourism", "shopping",
        "healthcare", "education", "food", "general_assistant",
    ],
)
def test_explicit_registered_domain_id_wins_over_conflicting_router_signals(domain_id):
    registry = DomainRegistryService()
    query = "Compare company law, hotel packages, medicine and school tuition."
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)

    routing = DomainRouterPlanner(registry).process(analysis, domain_hint=domain_id)
    policy = PolicyEnforcementMiddleware(registry).apply(routing.selected_domain)

    assert routing.selected_domain == domain_id
    assert routing.routing_reason == "DOMAIN_HINT"
    assert policy.domain_config.domain_id == domain_id


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
    assert legal.display_name_vi == "Pháp lý AI"
    assert legal.upload_enabled


def test_domain_api_adds_readiness_without_changing_legacy_domain_list():
    from fastapi.testclient import TestClient
    from backend.core_gateway import app

    with TestClient(app) as client:
        domains = client.get("/api/v2/domains/")
        readiness = client.get("/api/v2/domains/readiness")
        upload_targets = client.get("/api/v2/domains/upload-targets")

    assert domains.status_code == 200
    assert domains.json() == [
        "legal_ai", "general_assistant", "administrative_law", "tourism",
        "healthcare", "education", "shopping", "food",
    ]
    assert readiness.status_code == 200
    legal = next(item for item in readiness.json()["domains"] if item["domain_id"] == "legal_ai")
    assert legal["data_available"] == "unknown"
    assert legal["retrieval_ready"] == "unknown"
    assert legal["model_ready"] == "unknown"
    assert legal["e2e_verified"] == "not_verified"
    assert legal["display_name_vi"] == "Pháp lý AI"
    assert legal["upload_enabled"] is True
    assert upload_targets.status_code == 200
    upload_payload = upload_targets.json()
    assert [item["domain_id"] for item in upload_payload["domains"]] == [
        "administrative_law", "tourism", "healthcare", "education", "shopping", "food",
    ]
    assert "legal_ai" not in [item["domain_id"] for item in upload_payload["domains"]]
    assert "general_assistant" not in [item["domain_id"] for item in upload_payload["domains"]]
    for item in upload_payload["domains"]:
        assert item["enabled"] is True
        assert item["upload_enabled"] is True
        assert item["configured"] is True
        assert item["display_name"]
        assert item["allowed_extensions"] == [".txt", ".md", ".xlsx"]
        assert item["data_available"] == "unknown"
        assert item["e2e_verified"] == "not_verified"


def test_gateway_upload_rejects_auto_and_domains_not_enabled_for_upload():
    from fastapi.testclient import TestClient
    from backend.core_gateway import app

    with TestClient(app) as client:
        automatic = client.post(
            "/api/v2/domains/auto/documents",
            files={"file": ("note.txt", b"content", "text/plain")},
        )
        general = client.post(
            "/api/v2/domains/general_assistant/documents",
            files={"file": ("note.txt", b"content", "text/plain")},
        )

    assert automatic.status_code == 404
    assert general.status_code == 409


def test_each_requested_upload_domain_has_explicit_vietnamese_name_and_formats():
    registry = DomainRegistryService()
    expected = {
        "administrative_law": "Pháp luật hành chính",
        "tourism": "Du lịch",
        "healthcare": "Y tế",
        "education": "Giáo dục",
        "shopping": "Mua sắm",
        "food": "Ẩm thực",
    }
    assert [item.domain_id for item in registry.list_upload_domains()] == list(expected)
    for domain_id, display_name_vi in expected.items():
        config = registry.get_domain_config(domain_id)
        item = next(item for item in registry.list_upload_domains() if item.domain_id == domain_id)
        assert config.enabled
        assert config.upload_enabled
        assert config.display_name_vi == display_name_vi
        assert config.supported_extensions == [".txt", ".md", ".xlsx"]
        assert item.display_name_vi == display_name_vi
        assert item.allowed_extensions == config.supported_extensions
        assert item.data_available == "unknown"
        assert item.e2e_verified == "not_verified"


@pytest.mark.parametrize("query,expected", [
    ("Tôi tìm thông tin chăm sóc sức khỏe tại bệnh viện.", "healthcare"),
    ("Học phí và điều kiện tuyển sinh của trường đại học?", "education"),
    ("So sánh sản phẩm và chính sách đổi trả.", "shopping"),
    ("Tìm công thức nấu ăn và món ăn truyền thống.", "food"),
])
def test_new_domain_queries_route_to_their_registered_domain(query, expected):
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)
    routing = DomainRouterPlanner().process(analysis)
    assert routing.selected_domain == expected
    config = DomainRegistryService().get_domain_config(expected)
    assert config.evidence_required is True
    assert config.catalog_enabled is False
    assert config.policy.allowed_mcp_tools == []
    assert config.policy.verification_threshold is None


def test_hotel_weather_query_does_not_expand_hoi_an_into_company_law_or_override_hint():
    query = (
        "According to Vietnam_Tourism.xlsx, what weather is described for "
        "Hoi An from September through January?"
    )
    analysis = QueryAnalysisService().analyze(query, enable_hyde=False)

    assert "an" in analysis.expanded_terms
    assert "công ty cổ phần" not in analysis.expanded_terms
    assert "doanh nghiệp tư nhân" not in analysis.expanded_terms
    routing = DomainRouterPlanner().process(analysis, domain_hint="tourism")
    assert routing.selected_domain == "tourism"
    assert routing.routing_reason == "DOMAIN_HINT"


def test_complete_legal_thesaurus_phrase_still_expands():
    analysis = QueryAnalysisService().analyze(
        "Thủ tục đăng ký doanh nghiệp là gì?",
        enable_hyde=False,
    )

    assert "công ty cổ phần" in analysis.expanded_terms


def test_health_and_food_evidence_ambiguity_asks_before_retrieval():
    query = "Thông tin dinh dưỡng cho người cần chăm sóc sức khỏe."
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)
    routing = DomainRouterPlanner().process(analysis)
    assert routing.selected_domain is None
    assert routing.clarification_required
    assert routing.execution_plan == []


@pytest.mark.parametrize(
    "query,expected_risk,reason_code",
    [
        ("Where is the nearest hospital?", "high", "DOMAIN_DEFAULT_RISK_LEVEL"),
        ("What is the correct medication dosage?", "high", "PERSONALIZED_MEDICAL_ADVICE"),
        ("Tôi đang khó thở, nên làm gì?", "critical", "POTENTIAL_MEDICAL_EMERGENCY"),
    ],
)
def test_healthcare_risk_rules_are_reported_without_claiming_verification(
    query, expected_risk, reason_code
):
    analysis = AnalysisResult(original_query=query, cleaned_query=query, confidence_score=0.9)
    routing = DomainRouterPlanner().process(analysis)
    assert routing.selected_domain == "healthcare"
    assert routing.risk_level == expected_risk
    assert routing.risk_reason_code == reason_code


def test_new_domain_collections_are_isolated_from_each_other_and_legal_sources():
    registry = DomainRegistryService()
    all_collections = {
        "legal_docs_bge_m3",
        *(f"domain_documents_{domain}" for domain in registry.list_enabled_domains()),
    }
    for domain in ("healthcare", "education", "shopping", "food", "tourism"):
        targets = retrieval_collections_for_domain(
            domain, all_collections.__contains__, registry
        )
        assert targets == [f"domain_documents_{domain}"]


def test_new_domains_are_configured_but_not_claimed_data_or_runtime_ready():
    registry = DomainRegistryService()
    for domain_id in ("healthcare", "education", "shopping", "food"):
        config = registry.get_domain_config(domain_id)
        readiness = next(
            item for item in registry.readiness().domains
            if item.domain_id == domain_id
        )
        assert config.supported_extensions == [".txt", ".md", ".xlsx"]
        assert config.knowledge_collection is None
        assert readiness.configured
        assert readiness.data_available == "unknown"
        assert readiness.retrieval_ready == "unknown"
        assert readiness.model_ready == "unknown"
        assert readiness.e2e_verified == "not_verified"


def test_registry_owns_domain_retrieval_fields_and_strategy_defaults():
    registry = DomainRegistryService()
    tourism = registry.get_domain_config("tourism").retrieval
    education = registry.get_domain_config("education").retrieval

    assert tourism.structured_lookup_strategy == "tourism_records"
    assert tourism.retrieval_strategy == "hybrid"
    assert {"region", "province_city", "visit_duration"} <= set(tourism.filterable_fields)
    assert "record_fields" in tourism.searchable_fields
    assert education.structured_lookup_strategy is None
    assert education.retrieval_strategy == "semantic"
    assert "domain_id" in education.metadata_fields


def test_new_semantic_domain_can_use_generic_registry_retrieval_contract():
    config = DomainRetrievalConfig(
        metadata_fields=["domain_id", "document_id", "record_fields"],
        filterable_fields=["grade"],
        searchable_fields=["text", "record_fields"],
    )

    assert config.retrieval_strategy == "semantic"
    assert config.ranking_strategy == "rerank"
    assert config.structured_lookup_strategy is None
