from backend.answer.diagnostics import (
    citation_raw_debug_enabled,
    make_citation_attempt_logger,
    safe_raw_answer_for_debug,
    verifier_reason_codes,
)
from backend.answer.schemas import Claim, StructuredAnswer
from backend.evidence.schemas import EvidenceBundle
from backend.verifier.verifier import Verifier


def test_citation_logs_keep_trace_and_reason_codes_without_legal_content(capsys):
    private_question = "PRIVATE_USER_QUESTION_391"
    private_claim = "PRIVATE_LEGAL_CLAIM_284"
    private_context = "PRIVATE_EVIDENCE_CONTEXT_175"
    private_answer = "PRIVATE_RAW_ANSWER_982"
    structured = StructuredAnswer(
        answer=private_answer,
        claims=[Claim(claim=private_claim, evidence_ids=[])],
    )
    result = Verifier().verify(structured, EvidenceBundle(query=private_question))
    callback = make_citation_attempt_logger(
        "trace_legal_0123456789abcdef0123456789abcdef",
        lambda value: value,
    )

    callback(2, private_answer, structured, result)
    output = capsys.readouterr().out

    assert "trace_legal_0123456789abcdef0123456789abcdef" in output
    assert "citation_attempt= 2" in output
    assert "CLAIM_WITHOUT_EVIDENCE" in output
    for private_text in (private_question, private_claim, private_context, private_answer):
        assert private_text not in output


def test_verifier_reason_aggregation_does_not_expose_claim_text():
    structured = StructuredAnswer(
        answer="private",
        claims=[Claim(claim="claim A", evidence_ids=["missing-id"])],
    )
    result = Verifier().verify(structured, EvidenceBundle())

    assert verifier_reason_codes(result) == {"EVIDENCE_ID_NOT_FOUND": 1}


def test_raw_answer_debug_is_opt_in_and_keeps_request_id(capsys):
    private_answer = "PRIVATE_RAW_ANSWER_982"
    structured = StructuredAnswer(
        answer=private_answer,
        claims=[Claim(claim="Claim", evidence_ids=[])],
    )
    result = Verifier().verify(structured, EvidenceBundle())
    make_citation_attempt_logger("trace_debug", lambda value: value)(
        1, private_answer, structured, result
    )
    assert "CITATION RAW DEBUG" not in capsys.readouterr().out

    make_citation_attempt_logger(
        "trace_debug", lambda value: value, include_raw_answer=True
    )(1, private_answer, structured, result)
    output = capsys.readouterr().out
    assert "CITATION RAW DEBUG" in output
    assert "trace_debug" in output
    assert private_answer in output
    assert "redacted= False" in output


def test_opt_in_raw_answer_debug_marks_redaction_and_truncation(capsys):
    raw_answer = "Contact a@b.com; token=secret-value " + ("x" * 2100)
    structured = StructuredAnswer(answer=raw_answer, claims=[])
    result = Verifier().verify(structured, EvidenceBundle())

    make_citation_attempt_logger(
        "trace_legal_0123456789abcdef0123456789abcdef",
        safe_raw_answer_for_debug,
        include_raw_answer=True,
    )(1, raw_answer, structured, result)

    output = capsys.readouterr().out
    assert "redacted= True" in output
    assert "truncated= True" in output
    assert "a@b.com" not in output
    assert "secret-value" not in output
    assert len(output) < 2800


def test_attempt_logging_reports_unknown_ids_and_verification_scope(capsys):
    structured = StructuredAnswer(
        answer="Claim [Evidence ID: rag:wrong]",
        claims=[Claim(claim="Claim", evidence_ids=["rag:wrong"])],
    )
    result = Verifier().verify(structured, EvidenceBundle())
    make_citation_attempt_logger(
        "trace_legal_0123456789abcdef0123456789abcdef",
        safe_raw_answer_for_debug,
        allowed_evidence_ids=("rag:allowed",),
    )(1, structured.answer, structured, result)

    output = capsys.readouterr().out
    assert 'unknown_evidence_ids= ["rag:wrong"]' in output
    assert "allowed_evidence_count= 1" in output
    assert "claim_grounding_status= not_evaluated" in output
    assert "confidence_status= not_required" in output


def test_debug_gate_requires_development_and_explicit_opt_in():
    assert citation_raw_debug_enabled({}) is False
    assert citation_raw_debug_enabled({
        "DAI_ENVIRONMENT": "production",
        "DAI_DEBUG_CITATION_RAW_ANSWER": "true",
    }) is False
    assert citation_raw_debug_enabled({
        "DAI_ENVIRONMENT": "development",
        "DAI_DEBUG_CITATION_RAW_ANSWER": "yes",
    }) is True
