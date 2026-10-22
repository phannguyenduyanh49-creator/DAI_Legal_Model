from backend.mcp.adapter import mcp_dvc_result_to_evidence, mcp_result_status
from backend.evidence.builder import EvidenceBuilder
from backend.evidence.schemas import Evidence
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
    assert policy.allowed_mcp_tools == ()


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
