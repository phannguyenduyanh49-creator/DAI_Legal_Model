import pytest

from backend.mcp.adapter import mcp_dvc_result_to_evidence, mcp_result_status
from backend.evidence.builder import EvidenceBuilder
from backend.evidence.schemas import Evidence
from backend.domain_documents import retrieval_collections_for_domain
from backend.domain_registry import DomainRegistryService
from backend.orchestrator import DomainRouterPlanner
from backend.policy.middleware import PolicyEnforcementMiddleware
from backend.query_analysis import AnalysisResult
from backend.prompt.assembler import PromptAssembler


def test_strict_and_friendly_mcp_permissions_never_expand_beyond_registry():
    middleware = PolicyEnforcementMiddleware()

    strict = middleware.apply("administrative_law")
    assert strict.allowed_mcp_tools == ("search_dichvucong", "search_procedure")

    friendly = middleware.apply("general_assistant")
    assert friendly.domain_config.policy.mode == "FRIENDLY"
    assert friendly.allowed_mcp_tools == ()


def test_policy_rechecks_forwarded_tools_against_registry():
    middleware = PolicyEnforcementMiddleware()
    assert middleware.constrain_tools(
        "administrative_law",
        ["search_dichvucong", "search_legal_documents", "unknown_tool"],
    ) == ("search_dichvucong",)


def test_friendly_mode_is_reachable_through_domain_router_and_stays_allowlisted():
    analysis = AnalysisResult(
        original_query="general question",
        cleaned_query="general question",
        confidence_score=0.9,
    )
    routing = DomainRouterPlanner().process(
        analysis,
        domain_hint="general_assistant",
    )
    policy = PolicyEnforcementMiddleware().apply(routing.selected_domain)

    assert policy.domain_config.policy.mode == "FRIENDLY"


def test_tourism_domain_is_registered_and_unknown_domains_do_not_fall_back_to_legal():
    routing = DomainRouterPlanner().process(
        query_analysis=AnalysisResult(
            original_query="Which shuttle leaves the pier?",
            cleaned_query="Which shuttle leaves the pier?",
            confidence_score=0.9,
        ),
        domain_hint="tourism",
    )
    assert routing.selected_domain == "tourism"
    policy = PolicyEnforcementMiddleware().apply("tourism")
    assert policy.domain_config.domain_id == "tourism"
    assert policy.allowed_mcp_tools == ()
    with pytest.raises(KeyError):
        PolicyEnforcementMiddleware().apply("unknown_domain")


def test_frontend_legal_ai_hint_routes_to_the_same_registered_legal_domain():
    routing = DomainRouterPlanner().process(
        query_analysis=AnalysisResult(
            original_query="What does the legal source say?",
            cleaned_query="What does the legal source say?",
            confidence_score=0.9,
        ),
        domain_hint="legal_ai",
    )

    assert routing.selected_domain == "legal_ai"
    assert PolicyEnforcementMiddleware().apply(
        routing.selected_domain
    ).domain_config.domain_id == "legal_ai"


def test_company_question_auto_routes_to_legal_and_stays_legal_with_stale_nonlegal_hint():
    query = "Thủ tục thành lập công ty cổ phần là gì?"
    analysis = AnalysisResult(
        original_query=query,
        cleaned_query=query,
        confidence_score=0.9,
    )
    router = DomainRouterPlanner()
    registry = DomainRegistryService()
    middleware = PolicyEnforcementMiddleware(registry)

    automatic = router.process(analysis)
    assert automatic.selected_domain == "enterprise_law"
    assert middleware.apply(automatic.selected_domain).domain_config.domain_id == "legal_ai"
    assert middleware.apply(automatic.selected_domain).domain_config.knowledge_collection == "legal_docs_bge_m3"

    stale_tourism = router.process(analysis, domain_hint="tourism")
    assert stale_tourism.selected_domain == "enterprise_law"
    assert middleware.apply(stale_tourism.selected_domain).domain_config.domain_id == "legal_ai"
    assert middleware.apply(stale_tourism.selected_domain).domain_config.knowledge_collection == "legal_docs_bge_m3"

    generic_analysis = AnalysisResult(
        original_query="What time does the shuttle leave?",
        cleaned_query="What time does the shuttle leave?",
        confidence_score=0.9,
    )
    explicit_general = router.process(generic_analysis, domain_hint="general_assistant")
    assert explicit_general.selected_domain == "general_assistant"
    general_config = middleware.apply(explicit_general.selected_domain).domain_config
    assert general_config.knowledge_collection is None
    assert retrieval_collections_for_domain(
        "general_assistant", lambda _name: False, registry
    ) == []


@pytest.mark.parametrize(
    "query",
    [
        "Thủ tục thành lập công ty cổ phần là gì?",
        "Hồ sơ đăng ký thành lập công ty cổ phần gồm những giấy tờ gì?",
        "Công ty cổ phần khác công ty trách nhiệm hữu hạn ở điểm nào?",
        "Thời hạn giải quyết hồ sơ đăng ký doanh nghiệp là bao lâu?",
    ],
)
def test_reported_legal_queries_override_stale_tourism_hint_and_enable_legal_sources(query):
    analysis = AnalysisResult(
        original_query=query,
        cleaned_query=query,
        confidence_score=0.9,
    )

    routing = DomainRouterPlanner().process(analysis, domain_hint="tourism")
    policy = PolicyEnforcementMiddleware().apply(routing.selected_domain)

    assert routing.selected_domain in {"enterprise_law", "administrative_law"}
    assert policy.domain_config.domain_id == "legal_ai"
    assert policy.domain_config.knowledge_collection == "legal_docs_bge_m3"
    assert "search_procedure" in policy.allowed_mcp_tools


def test_mcp_result_becomes_structured_evidence():
    evidence = mcp_dvc_result_to_evidence(
        "DVC_OFFICIAL_PROCEDURE\nDVC_PROCEDURE_NAME: Cap ho chieu\n"
        "DVC_PROCEDURE_CODE: 123\nDVC_PROCEDURE_ID: 456\n"
        "DVC_SOURCE_URL: https://dichvucong.gov.vn\n"
        "DVC_DETAIL_BEGIN\nHo so gom giay to hop le.\nDVC_DETAIL_END"
    )

    assert evidence is not None
    assert evidence.evidence_id == "mcp:dvc:456"
    assert evidence.source_type == "mcp"
    assert evidence.content == "Ho so gom giay to hop le."
    assert evidence.metadata["tool"] == "search_dichvucong"


def test_mcp_evidence_id_survives_builder_and_prompt_assembly():
    evidence = Evidence(
        evidence_id="mcp:dvc:456",
        source_type="mcp",
        content="DVC detail. " * 450,
        score=1.0,
        metadata={"tool": "search_dichvucong"},
    )
    bundle = EvidenceBuilder().build(
        rag_evidences=[
            Evidence(
                evidence_id="rag:long-local",
                source_type="rag",
                content="Local detail. " * 400,
                score=1.0,
            )
        ],
        mcp_evidences=[evidence],
        query="procedure documents",
    )
    prompt_assembler = PromptAssembler(max_evidence_chars=6000)
    included = prompt_assembler.included_evidences(bundle)
    prompt = prompt_assembler.build(bundle.query, bundle)

    assert bundle.evidences[0].evidence_id == "rag:long-local"
    assert included[0].evidence_id == "mcp:dvc:456"
    assert included[0].metadata["tool"] == "search_dichvucong"
    assert "Allowed Evidence IDs: mcp:dvc:456" in prompt
    assert "rag:long-local" not in prompt


def test_oversized_mcp_evidence_is_truncated_with_id_and_metadata_preserved():
    evidence = Evidence(
        evidence_id="mcp:dvc:large-result",
        source_type="mcp",
        title="Official procedure",
        content="Official detail. " * 1000,
        score=1.0,
        source_url="https://dichvucong.gov.vn",
        metadata={"tool": "search_dichvucong", "procedure_id": "large-result"},
    )
    assembler = PromptAssembler(max_evidence_chars=6000)
    bundle = EvidenceBuilder().build(mcp_evidences=[evidence], query="procedure")

    included = assembler.included_evidences(bundle)
    prompt = assembler.build(bundle.query, bundle)

    assert len(included) == 1
    assert included[0].evidence_id == evidence.evidence_id
    assert included[0].source_url == evidence.source_url
    assert included[0].metadata == evidence.metadata
    assert "[Evidence content truncated to fit prompt budget.]" in included[0].content
    assert f"[Evidence ID: {evidence.evidence_id}]" in prompt
    assert "Allowed Evidence IDs: mcp:dvc:large-result" in prompt
    assert "rag:" not in prompt


def test_oversized_local_evidence_keeps_its_id_in_prompt():
    evidence = Evidence(
        evidence_id="rag:large-local-result",
        source_type="rag",
        title="Local procedure",
        content="Local detail. " * 1000,
        score=0.8,
        metadata={"formality_id": "local-result"},
    )
    assembler = PromptAssembler(max_evidence_chars=1200)
    bundle = EvidenceBuilder().build(rag_evidences=[evidence], query="procedure")

    included = assembler.included_evidences(bundle)
    prompt = assembler.build(bundle.query, bundle)

    assert len(included) == 1
    assert included[0].evidence_id == evidence.evidence_id
    assert included[0].metadata == evidence.metadata
    assert "[Evidence content truncated to fit prompt budget.]" in included[0].content
    assert f"[Evidence ID: {evidence.evidence_id}]" in prompt
    assert f"Allowed Evidence IDs: {evidence.evidence_id}" in prompt


def test_mcp_result_errors_are_not_converted_to_evidence():
    assert mcp_dvc_result_to_evidence("Lỗi timeout MCP: quá hạn") is None


def test_no_match_contract_is_distinct_from_invalid_and_completed_results():
    assert mcp_result_status(
        "DVC_NO_RELEVANT_RESULT: no sufficiently relevant procedure matched."
    ) == "no_match"
    assert mcp_result_status("unexpected text") == "invalid_response"
    assert mcp_result_status(
        "DVC_OFFICIAL_PROCEDURE\nDVC_PROCEDURE_ID: 7\n"
        "DVC_DETAIL_BEGIN\nDetail\nDVC_DETAIL_END"
    ) == "completed"
