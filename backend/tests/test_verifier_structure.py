import asyncio
import json
from unittest.mock import patch

import httpx

from backend.answer.citations import structure_answer_with_explicit_evidence
from backend.answer.citation_generation import generate_with_citation_retry
from backend.answer.diagnostics import make_citation_attempt_logger
from backend.answer.model_api_client import post_model_generation
from backend.answer.schemas import Claim, StructuredAnswer
from backend.evidence.schemas import Evidence, EvidenceBundle
from backend.prompt.assembler import PromptAssembler
from backend.verifier.verifier import Verifier


def make_bundle(content="Procedure details"):
    return EvidenceBundle(
        query="question",
        evidences=[
            Evidence(
                evidence_id="rag:example-001",
                source_type="rag",
                content=content,
            )
        ],
        total=1,
    )


def test_extracts_explicit_evidence_id_at_sentence_end():
    structured = structure_answer_with_explicit_evidence(
        "Submit the application. [Evidence ID: rag:example-001]"
    )

    assert len(structured.claims) == 1
    assert structured.claims[0].evidence_ids == ["rag:example-001"]


def test_citation_on_standalone_paragraph_end_binds_only_previous_claim():
    structured = structure_answer_with_explicit_evidence(
        "First legal statement.\nSecond legal statement.\n"
        "[Evidence ID: rag:example-001]"
    )

    assert len(structured.claims) == 2
    assert structured.claims[0].evidence_ids == []
    assert structured.claims[1].evidence_ids == ["rag:example-001"]


def test_standalone_citation_after_blank_line_does_not_cross_paragraph_boundary():
    structured = structure_answer_with_explicit_evidence(
        "Uncited claim.\n\n[Evidence ID: rag:example-001]"
    )

    assert len(structured.claims) == 1
    assert structured.claims[0].evidence_ids == []


def test_heading_citation_does_not_attach_to_following_claim():
    structured = structure_answer_with_explicit_evidence(
        "## Heading [Evidence ID: rag:example-001]\nUncited claim."
    )

    assert len(structured.claims) == 1
    assert structured.claims[0].claim == "Uncited claim."
    assert structured.claims[0].evidence_ids == []


def test_malformed_citation_is_not_accepted():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:example-001"
    )

    result = Verifier().verify(structured, make_bundle())
    assert result.passed is False
    assert result.linked_claims == 0


def test_citation_followed_by_prose_is_not_accepted():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:example-001] unsupported trailing text"
    )

    result = Verifier().verify(structured, make_bundle())
    assert result.passed is False
    assert result.linked_claims == 0


def test_verifier_rejects_missing_citation():
    answer = StructuredAnswer(answer="No citation", claims=[Claim("No citation")])

    result = Verifier().verify(answer, make_bundle())

    assert result.passed is False
    assert result.checked_claims == 1
    assert result.linked_claims == 0
    assert result.claim_results == [{
        "claim": "No citation",
        "evidence_ids": [],
        "passed": False,
        "reasons": ["Claim has no evidence: No citation"],
        "citation_integrity_status": "failed",
        "claim_grounding_status": "not_evaluated",
        "grounding_reason": "NO_GROUNDING_SCORER_CONFIGURED",
    }]


def test_verifier_rejects_unknown_evidence_id():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:does-not-exist]"
    )

    result = Verifier().verify(structured, make_bundle())

    assert result.passed is False
    assert any("Evidence not found" in error for error in result.errors)


def test_verifier_rejects_empty_evidence_content():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:example-001]"
    )

    result = Verifier().verify(structured, make_bundle(content="  "))

    assert result.passed is False
    assert any("empty content" in error for error in result.errors)


def test_one_claim_can_reference_multiple_allowed_evidence_ids():
    bundle = EvidenceBundle(
        query="question",
        evidences=[
            Evidence(evidence_id="rag:first", source_type="rag", content="First source"),
            Evidence(evidence_id="mcp:dvc:second", source_type="mcp", content="Second source"),
        ],
    )
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:first] [Evidence ID: mcp:dvc:second]"
    )

    result = Verifier().verify(structured, bundle)

    assert structured.claims[0].evidence_ids == ["rag:first", "mcp:dvc:second"]
    assert result.citation_integrity_passed is True
    assert result.claim_results[0]["claim_grounding_status"] == "not_evaluated"


def test_unknown_id_still_fails_when_the_same_claim_has_a_valid_id():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:example-001] [Evidence ID: rag:missing]"
    )

    result = Verifier().verify(structured, make_bundle())

    assert result.citation_integrity_passed is False
    assert any("Evidence not found: rag:missing" in error for error in result.errors)


def test_sentence_citation_does_not_attach_to_other_claims_on_same_line():
    structured = structure_answer_with_explicit_evidence(
        "First claim. Second claim. [Evidence ID: rag:example-001]"
    )

    assert [claim.evidence_ids for claim in structured.claims] == [
        [], ["rag:example-001"]
    ]
    result = Verifier().verify(structured, make_bundle())
    assert result.citation_integrity_passed is False
    assert result.linked_claims == 1


def test_citation_at_line_start_or_middle_does_not_ground_trailing_claim():
    for answer in (
        "[Evidence ID: rag:example-001] Claim.",
        "Claim [Evidence ID: rag:example-001] remains unsupported.",
    ):
        structured = structure_answer_with_explicit_evidence(answer)
        result = Verifier().verify(structured, make_bundle())
        assert result.citation_integrity_passed is False


def test_empty_model_output_has_no_claims_and_fails_closed():
    structured = structure_answer_with_explicit_evidence("  \n")
    result = Verifier().verify(structured, make_bundle())

    assert result.checked_claims == 0
    assert result.citation_integrity_passed is False
    assert "Answer contains no claims to validate." in result.errors


def test_invalid_confidence_threshold_is_not_treated_as_not_required():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:example-001]"
    )

    result = Verifier().verify(
        structured, make_bundle(), minimum_confidence=float("nan")
    )

    assert result.confidence_status == "invalid_threshold"
    assert result.confidence_reason == "THRESHOLD_MUST_BE_FINITE_AND_BETWEEN_0_AND_1"
    assert result.passed is False


def test_valid_citation_does_not_claim_grounding_without_a_grounding_scorer():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:example-001]"
    )

    result = Verifier().verify(structured, make_bundle())

    assert result.citation_integrity_passed is True
    assert result.claim_grounding_status == "not_evaluated"
    assert result.passed is False
    assert result.linked_claims == 1
    assert result.semantic_support_evaluated is False


def test_configured_confidence_threshold_is_not_fabricated_without_scorer():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:example-001]"
    )

    result = Verifier().verify(
        structured, make_bundle(), minimum_confidence=0.85
    )

    assert result.citation_integrity_passed is True
    assert result.confidence_score is None
    assert result.confidence_status == "not_verified"
    assert result.passed is False


def test_configured_confidence_threshold_accepts_only_real_injected_score():
    structured = structure_answer_with_explicit_evidence(
        "Claim. [Evidence ID: rag:example-001]"
    )

    below = Verifier().verify(
        structured, make_bundle(), minimum_confidence=0.85, confidence_score=0.84
    )
    meets = Verifier().verify(
        structured, make_bundle(), minimum_confidence=0.85, confidence_score=0.85
    )

    assert below.confidence_status == "failed"
    assert below.passed is False
    assert meets.confidence_status == "passed"
    assert meets.claim_grounding_status == "not_evaluated"
    assert meets.passed is False


def test_prompt_contains_exact_allowed_ids_and_citation_rule():
    prompt = PromptAssembler().build("question", make_bundle())

    assert "[Evidence ID: rag:example-001]" in prompt
    assert "Allowed Evidence IDs: rag:example-001" in prompt
    assert "REQUIRED CITATION FORMAT" in prompt


def test_prompt_applies_domain_template_using_query_and_bounded_evidence():
    prompt = PromptAssembler().build(
        "question from user",
        make_bundle(),
        domain_prompt_template="Domain query={query}; sources={context}",
    )

    assert "=== DOMAIN-SPECIFIC INSTRUCTIONS ===" in prompt
    assert "Domain query=question from user" in prompt
    assert "sources=the shared EVIDENCE section above" in prompt
    assert prompt.count("[Evidence ID: rag:example-001]") == 1


def test_malformed_domain_template_does_not_break_prompt_assembly():
    prompt = PromptAssembler().build(
        "question",
        make_bundle(),
        domain_prompt_template="{missing_field}",
    )

    assert "REQUIRED CITATION FORMAT" in prompt
    assert "DOMAIN-SPECIFIC INSTRUCTIONS" not in prompt


def test_model_api_request_preserves_assembled_prompt_and_trace_headers():
    prompt = (
        "=== EVIDENCE ===\n[Evidence ID: rag:example-001]\n"
        "Nội dung có dấu tiếng Việt.\n"
        "Allowed Evidence IDs: rag:example-001"
    )
    captured = {}

    async def send():
        def handler(request):
            captured["body"] = request.read().decode("utf-8")
            captured["request_id"] = request.headers["X-Request-ID"]
            captured["citation_attempt"] = request.headers["X-Citation-Attempt"]
            return httpx.Response(200, json={"success": True, "answer": "ok"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await post_model_generation(
                client=client,
                url="http://model.test/generate",
                prompt=prompt,
                request_id="trace_test_123",
                citation_attempt=2,
            )
            return response.json()

    result = asyncio.run(send())

    assert result["success"] is True
    assert json.loads(captured["body"]) == {
        "prompt": prompt,
        "max_new_tokens": 256,
    }
    assert captured["request_id"] == "trace_test_123"
    assert captured["citation_attempt"] == "2"


def test_generation_retries_for_citation_then_withholds_without_grounding():
    calls = []
    attempts = []

    async def generate(prompt, attempt):
        calls.append(prompt)
        attempts.append(attempt)
        answer = (
            "Claim without citation."
            if len(calls) == 1
            else "Claim. [Evidence ID: rag:example-001]"
        )
        return {
            "success": True,
            "answer": answer,
            "processing_time_ms": 25 if len(calls) == 1 else 30,
        }

    result = asyncio.run(
        generate_with_citation_retry(
            prompt="initial prompt",
            evidence_bundle=make_bundle(),
            generate=generate,
            verifier=Verifier(),
            max_attempts=99,
        )
    )

    assert len(calls) == 2
    assert attempts == [1, 2]
    assert "CITATION REPAIR" in calls[1]
    assert "rag:example-001" in calls[1]
    assert result.attempts == 2
    assert result.answer is None
    assert result.verification.citation_integrity_passed is True
    assert result.verification.claim_grounding_status == "not_evaluated"
    assert result.verification.passed is False
    assert result.model_processing_time_ms == 55


def test_generation_withholds_once_when_confidence_threshold_cannot_be_checked():
    calls = []

    class ThresholdVerifier(Verifier):
        def verify(self, answer, evidence_bundle):
            return super().verify(
                answer, evidence_bundle, minimum_confidence=0.85
            )

    async def generate(_prompt, attempt):
        calls.append(attempt)
        return {
            "success": True,
            "answer": "Claim. [Evidence ID: rag:example-001]",
        }

    result = asyncio.run(
        generate_with_citation_retry(
            prompt="prompt",
            evidence_bundle=make_bundle(),
            generate=generate,
            verifier=ThresholdVerifier(),
        )
    )

    assert calls == [1]
    assert result.answer is None
    assert result.verification.citation_integrity_passed is True
    assert result.verification.confidence_status == "not_verified"


def test_citation_attempt_logger_matches_retry_callback_signature(capsys):
    calls = []

    async def generate(prompt, attempt):
        calls.append(attempt)
        answer = (
            "Uncited statement."
            if attempt == 1
            else "Supported statement. [Evidence ID: rag:example-001]"
        )
        return {"success": True, "answer": answer}

    result = asyncio.run(
        generate_with_citation_retry(
            prompt="prompt",
            evidence_bundle=make_bundle(),
            generate=generate,
            verifier=Verifier(),
            on_attempt=make_citation_attempt_logger(
                "trace_test",
                lambda value: value[:200],
            ),
        )
    )

    assert calls == [1, 2]
    assert result.answer is None
    assert result.verification.citation_integrity_passed is True
    assert result.verification.passed is False
    output = capsys.readouterr().out
    assert "citation_attempt= 1" in output
    assert "citation_attempt= 2" in output


def test_generation_stops_after_two_invalid_or_unexpected_responses():
    calls = []
    diagnostics = []

    async def generate(prompt, attempt):
        calls.append(prompt)
        return {"success": True, "unexpected": "shape"}

    result = asyncio.run(
        generate_with_citation_retry(
            prompt="initial prompt",
            evidence_bundle=make_bundle(),
            generate=generate,
            verifier=Verifier(),
            max_attempts=10,
            on_attempt=lambda attempt, answer, structured, verification: diagnostics.append(
                (attempt, answer, structured, verification)
            ),
        )
    )

    assert len(calls) == 2
    assert result.attempts == 2
    assert result.answer is None
    assert result.verification.passed is False
    assert [item[0] for item in diagnostics] == [1, 2]
    assert all(item[3].claim_results == [] for item in diagnostics)


def test_generation_retries_transport_failure_once_and_preserves_prompt():
    calls = []
    prompt = "assembled prompt with Evidence ID: rag:example-001"

    async def generate(current_prompt, attempt):
        calls.append((current_prompt, attempt))
        if attempt == 1:
            raise httpx.ConnectError("temporary connection failure")
        return {
            "success": True,
            "answer": "Supported claim. [Evidence ID: rag:example-001]",
        }

    result = asyncio.run(
        generate_with_citation_retry(
            prompt=prompt,
            evidence_bundle=make_bundle(),
            generate=generate,
            verifier=Verifier(),
        )
    )

    assert calls == [(prompt, 1), (prompt, 2)]
    assert result.attempts == 2
    assert result.generation_error is None
    assert result.verification.citation_integrity_passed is True
    assert result.verification.passed is False
    assert result.answer is None


def test_generation_failure_stops_at_configured_attempt_limit():
    calls = []

    async def generate(_prompt, attempt):
        calls.append(attempt)
        raise httpx.ConnectError("model unavailable")

    result = asyncio.run(
        generate_with_citation_retry(
            prompt="assembled prompt",
            evidence_bundle=make_bundle(),
            generate=generate,
            verifier=Verifier(),
            max_attempts=20,
        )
    )

    assert calls == [1, 2]
    assert result.attempts == 2
    assert result.generation_error == "ConnectError"
    assert result.answer is None
    assert result.verification.passed is False


def test_gateway_withholds_upstream_answer_when_verifier_failed():
    from backend.core_gateway import LegacyPipelineResult, app, gateway_instance

    async def failed_pipeline(**_kwargs):
        return LegacyPipelineResult(
            answer="Uncited legal answer that must not be returned.",
            pipeline={"citation_verification": "failed"},
            verifier_status="failed",
            verifier_scope="citation_integrity_only",
        )

    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
        ) as client:
            return await client.post(
                "/api/v2/legal-query",
                json={
                    "query": "Thủ tục thành lập công ty cổ phần là gì?",
                    "enable_hyde": False,
                },
            )

    with patch.object(gateway_instance, "_call_legacy_pipeline", failed_pipeline):
        response = asyncio.run(send_request())

    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "FAILED"
    assert data["pipeline_execution"]["answer_withheld"] is True
    assert "Uncited legal answer" not in data["answer"]


def test_gateway_withholds_uncited_answer_when_verifier_status_is_missing():
    from backend.core_gateway import LegacyPipelineResult, app, gateway_instance

    async def pipeline_without_verifier(**_kwargs):
        return LegacyPipelineResult(
            answer="Uncited answer with no verifier result.",
            pipeline={"citation_verification": "not_reported"},
        )

    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
        ) as client:
            return await client.post(
                "/api/v2/legal-query",
                json={"query": "Question without a citation", "enable_hyde": False},
            )

    with patch.object(
        gateway_instance,
        "_call_legacy_pipeline",
        pipeline_without_verifier,
    ):
        response = asyncio.run(send_request())

    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "FAILED"
    assert data["pipeline_execution"]["verifier"] == "not_reported"
    assert data["pipeline_execution"]["answer_withheld"] is True
    assert data["answer"] == (
        "Chua the dua ra cau tra loi phap ly du dieu kien "
        "kiem tra trich dan Evidence."
    )
    assert "Uncited answer" not in data["answer"]


def test_gateway_withholds_answer_when_verifier_errors():
    from backend.core_gateway import LegacyPipelineResult, app, gateway_instance

    async def failed_verification(**_kwargs):
        return LegacyPipelineResult(
            answer="Answer produced before verifier error.",
            pipeline={"citation_verification": "error"},
            verifier_status="error",
            verifier_scope="citation_integrity_only",
        )

    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
        ) as client:
            return await client.post(
                "/api/v2/legal-query",
                json={"query": "Question that failed verification", "enable_hyde": False},
            )

    with patch.object(
        gateway_instance,
        "_call_legacy_pipeline",
        failed_verification,
    ):
        response = asyncio.run(send_request())

    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "FAILED"
    assert data["pipeline_execution"]["verifier"] == "error"
    assert data["pipeline_execution"]["answer_withheld"] is True
    assert "Answer produced before verifier error" not in data["answer"]


def test_gateway_withholds_answer_when_passed_status_has_no_evidence_telemetry():
    from backend.core_gateway import LegacyPipelineResult, app, gateway_instance

    async def inconsistent_pipeline(**_kwargs):
        return LegacyPipelineResult(
            answer="Claim. [Evidence ID: rag:missing-telemetry]",
            pipeline={"citation_verification": "passed"},
            verifier_status="passed",
            verifier_scope="citation_integrity_only",
        )

    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
        ) as client:
            return await client.post(
                "/api/v2/legal-query",
                json={"query": "Question with inconsistent telemetry", "enable_hyde": False},
            )

    with patch.object(gateway_instance, "_call_legacy_pipeline", inconsistent_pipeline):
        response = asyncio.run(send_request())

    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "FAILED"
    assert data["pipeline_execution"]["answer_withheld"] is True
    assert "missing-telemetry" not in data["answer"]


def test_gateway_withholds_answer_when_cited_id_is_not_in_allowed_evidence_ids():
    from backend.core_gateway import LegacyPipelineResult, app, gateway_instance

    async def invalid_citation_pipeline(**_kwargs):
        return LegacyPipelineResult(
            answer="Claim. [Evidence ID: rag:unknown]",
            pipeline={
                "citation_verification": "passed",
                "evidence_ids": ["rag:known"],
                "cited_evidence_ids": ["rag:unknown"],
            },
            verifier_status="passed",
            verifier_scope="citation_integrity_only",
        )

    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
        ) as client:
            return await client.post(
                "/api/v2/legal-query",
                json={"query": "Question with invalid Evidence ID", "enable_hyde": False},
            )

    with patch.object(gateway_instance, "_call_legacy_pipeline", invalid_citation_pipeline):
        response = asyncio.run(send_request())

    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "FAILED"
    assert data["pipeline_execution"]["answer_withheld"] is True
    assert "rag:unknown" not in data["answer"]
