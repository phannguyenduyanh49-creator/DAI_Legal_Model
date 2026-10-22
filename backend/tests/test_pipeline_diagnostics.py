from backend.answer.diagnostics import (
    make_citation_attempt_logger,
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
