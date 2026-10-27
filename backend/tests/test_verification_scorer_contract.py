import json
from pathlib import Path

import pytest

from backend.answer.citations import structure_answer_with_explicit_evidence
from backend.evidence.schemas import Evidence, EvidenceBundle
from backend.verifier.scoring import (
    ConfidenceAssessment,
    GroundingAssessment,
    ScorerUnavailableError,
)
from backend.verifier.verifier import Verifier


EVIDENCE_ID = "rag:legal-source-1"


def _bundle(content="Người nộp hồ sơ phải sử dụng Mẫu CT01."):
    return EvidenceBundle(
        query="q",
        evidences=[Evidence(
            evidence_id=EVIDENCE_ID,
            source_type="rag",
            content=content,
        )],
        total=1,
    )


def _answer(text="Người nộp hồ sơ phải sử dụng Mẫu CT01.", evidence_id=EVIDENCE_ID):
    return structure_answer_with_explicit_evidence(
        f"{text} [Evidence ID: {evidence_id}]"
    )


class FixedConfidenceScorer:
    def __init__(self, score=0.9):
        self.value = score
        self.seen_answer = None

    def score(self, answer, evidence_bundle, *, timeout_seconds):
        self.seen_answer = answer
        return ConfidenceAssessment(self.value, "synthetic-test-confidence")


class FixedGroundingScorer:
    def __init__(self, status="supported"):
        self.status = status
        self.seen_evidence = None

    def evaluate(self, claim, cited_evidence, *, timeout_seconds):
        self.seen_evidence = tuple(cited_evidence)
        return GroundingAssessment(
            self.status,
            f"SYNTHETIC_{self.status.upper()}",
            "synthetic-test-grounding",
        )


def test_verifier_integrates_labeled_synthetic_support_and_contradiction_cases():
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "verifier"
        / "evaluation"
        / "synthetic_grounding_cases.jsonl"
    )
    cases = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
    ]
    evaluable_cases = [
        case for case in cases
        if case["label"] in {"supported", "unsupported", "contradicted"}
    ]

    class LabeledFixtureOracle:
        """Test oracle for verifier wiring; it does not evaluate semantics."""

        def __init__(self, expected_case):
            self.expected_case = expected_case
            self.seen_pair = None

        def evaluate(self, claim, cited_evidence, *, timeout_seconds):
            self.seen_pair = (claim.claim, tuple(cited_evidence))
            assert claim.claim == self.expected_case["claim"]
            assert [item.evidence_id for item in cited_evidence] == [
                self.expected_case["evidence_id"]
            ]
            assert [item.content for item in cited_evidence] == [
                self.expected_case["evidence_text"]
            ]
            return GroundingAssessment(
                self.expected_case["label"],
                f"SYNTHETIC_{self.expected_case['label'].upper()}",
                "synthetic-fixture-oracle",
            )

    assert {case["label"] for case in evaluable_cases} == {
        "supported", "unsupported", "contradicted"
    }
    for case in evaluable_cases:
        evidence = Evidence(
            evidence_id=case["evidence_id"],
            source_type="fixture",
            content=case["evidence_text"],
        )
        answer = structure_answer_with_explicit_evidence(
            f"{case['claim']} [Evidence ID: {case['evidence_id']}]"
        )
        scorer = LabeledFixtureOracle(case)
        result = Verifier(
            confidence_scorer=FixedConfidenceScorer(),
            grounding_scorer=scorer,
        ).verify(
            answer,
            EvidenceBundle(query="synthetic fixture", evidences=[evidence]),
            minimum_confidence=0.85,
        )

        assert result.citation_integrity_passed is True
        assert result.claim_results[0]["citation_integrity_status"] == "passed"
        assert result.claim_results[0]["claim_grounding_status"] == case["label"]
        assert result.claim_grounding_status == (
            "passed" if case["label"] == "supported" else "failed"
        )
        assert result.passed is (case["label"] == "supported")
        assert result.outcome == (
            "verified" if case["label"] == "supported" else "failed"
        )
        assert scorer.seen_pair is not None


@pytest.mark.parametrize(
    ("status", "aggregate"),
    [
        ("supported", "passed"),
        ("unsupported", "failed"),
        ("contradicted", "failed"),
        ("partially_supported", "failed"),
    ],
)
def test_grounding_scorer_status_controls_aggregate_without_inference_from_id(
    status, aggregate
):
    grounding = FixedGroundingScorer(status)
    result = Verifier(
        confidence_scorer=FixedConfidenceScorer(),
        grounding_scorer=grounding,
    ).verify(_answer(), _bundle(), minimum_confidence=0.85)

    assert result.claim_grounding_status == aggregate
    assert result.semantic_support_evaluated is True
    assert result.claim_results[0]["claim_grounding_status"] == status
    assert result.passed is (status == "supported")
    # The scorer receives the exact Evidence referenced by the citation.
    assert [e.evidence_id for e in grounding.seen_evidence] == [EVIDENCE_ID]
    assert [e.content for e in grounding.seen_evidence] == [
        "Người nộp hồ sơ phải sử dụng Mẫu CT01."
    ]


@pytest.mark.parametrize(
    ("status", "aggregate", "outcome"),
    [
        ("insufficient_evidence", "insufficient_evidence", "failed"),
        ("inconclusive", "inconclusive", "inconclusive"),
    ],
)
def test_grounding_contract_preserves_insufficient_and_inconclusive_results(
    status, aggregate, outcome
):
    result = Verifier(
        confidence_scorer=FixedConfidenceScorer(),
        grounding_scorer=FixedGroundingScorer(status),
    ).verify(_answer(), _bundle(), minimum_confidence=0.85)

    assert result.claim_results[0]["claim_grounding_status"] == status
    assert result.claim_results[0]["claim_verified"] is False
    assert result.claim_grounding_status == aggregate
    assert result.semantic_support_evaluated is True
    assert result.outcome == outcome
    assert result.passed is False


def test_evaluator_unavailable_assessment_is_not_treated_as_inconclusive_or_support():
    result = Verifier(grounding_scorer=FixedGroundingScorer("evaluator_unavailable")).verify(
        _answer(), _bundle()
    )

    assert result.claim_results[0]["claim_grounding_status"] == "evaluator_unavailable"
    assert result.grounding_scorer_status == "unavailable"
    assert result.semantic_support_evaluated is False
    assert result.outcome == "evaluator_unavailable"
    assert result.passed is False


def test_missing_evidence_does_not_invoke_semantic_scorer_or_pass():
    grounding = FixedGroundingScorer()
    result = Verifier(grounding_scorer=grounding).verify(
        _answer(evidence_id="rag:missing"), _bundle()
    )

    assert result.citation_integrity_passed is False
    assert result.claim_grounding_status == "not_evaluated"
    assert result.grounding_scorer_status == "not_run"
    assert result.passed is False
    assert grounding.seen_evidence is None


def test_domain_metadata_mismatch_fails_citation_integrity_before_grounding():
    evidence = Evidence(
        evidence_id=EVIDENCE_ID,
        source_type="domain_upload",
        content="Procedure text.",
        metadata={"domain_id": "tourism"},
    )
    grounding = FixedGroundingScorer()
    result = Verifier(grounding_scorer=grounding).verify(
        _answer(),
        EvidenceBundle(query="q", evidences=[evidence]),
        expected_domain_id="legal_ai",
    )

    assert result.citation_integrity_passed is False
    assert result.grounding_scorer_status == "not_run"
    assert result.claim_results[0]["citation_integrity_status"] == "failed"
    assert any("domain mismatch" in error for error in result.errors)
    assert grounding.seen_evidence is None


def test_duplicate_evidence_ids_are_ambiguous_and_cannot_be_verified():
    first = Evidence(
        evidence_id=EVIDENCE_ID,
        source_type="rag",
        content="First source text.",
    )
    second = Evidence(
        evidence_id=EVIDENCE_ID,
        source_type="catalog",
        content="Different source text.",
    )
    grounding = FixedGroundingScorer()
    result = Verifier(grounding_scorer=grounding).verify(
        _answer(),
        EvidenceBundle(query="q", evidences=[first, second]),
    )

    assert result.citation_integrity_passed is False
    assert result.claim_results[0]["citation_integrity_status"] == "failed"
    assert any("Evidence ID is ambiguous" in error for error in result.errors)
    assert result.outcome == "failed"
    assert result.passed is False
    assert grounding.seen_evidence is None


def test_invalid_citation_fails_before_scorer_can_claim_support():
    result = Verifier(
        confidence_scorer=FixedConfidenceScorer(),
        grounding_scorer=FixedGroundingScorer(),
    ).verify(_answer(evidence_id="rag:unknown"), _bundle(), minimum_confidence=0.1)

    assert result.citation_integrity_passed is False
    assert result.confidence_scorer_status == "not_run"
    assert result.grounding_scorer_status == "not_run"
    assert result.outcome == "failed"
    assert result.passed is False


@pytest.mark.parametrize("failure", ["timeout", "unavailable", "error", "malformed"])
def test_grounding_scorer_failure_preserves_not_evaluated_and_fails_closed(failure):
    class FailingGrounding:
        def evaluate(self, claim, cited_evidence, *, timeout_seconds):
            if failure == "timeout":
                raise TimeoutError()
            if failure == "unavailable":
                raise ScorerUnavailableError()
            if failure == "error":
                raise RuntimeError("provider detail must not escape result")
            return {"status": "supported"}

    result = Verifier(
        confidence_scorer=FixedConfidenceScorer(),
        grounding_scorer=FailingGrounding(),
    ).verify(_answer(), _bundle(), minimum_confidence=0.85)

    assert result.claim_grounding_status == "not_evaluated"
    assert result.semantic_support_evaluated is False
    assert result.grounding_scorer_status in (
        "timeout", "unavailable", "error", "invalid_output"
    )
    assert result.claim_results[0]["claim_grounding_status"] == "evaluator_unavailable"
    assert result.claim_results[0]["claim_verified"] is False
    assert result.passed is False


@pytest.mark.parametrize("failure", ["timeout", "unavailable", "error", "malformed", "invalid_score"])
def test_confidence_scorer_failure_never_fabricates_a_score(failure):
    class FailingConfidence:
        def score(self, answer, evidence_bundle, *, timeout_seconds):
            if failure == "timeout":
                raise TimeoutError()
            if failure == "unavailable":
                raise ScorerUnavailableError()
            if failure == "error":
                raise RuntimeError("provider detail")
            if failure == "malformed":
                return {"score": 0.99}
            return ConfidenceAssessment(float("nan"), "bad-scorer")

    result = Verifier(
        confidence_scorer=FailingConfidence(),
        grounding_scorer=FixedGroundingScorer(),
    ).verify(_answer(), _bundle(), minimum_confidence=0.85)

    assert result.confidence_score is None
    assert result.confidence_status in ("not_verified", "invalid_score")
    assert result.outcome == (
        "evaluator_unavailable"
        if result.confidence_scorer_status in {"timeout", "unavailable", "error", "invalid_output"}
        else "not_verified"
    )
    assert result.passed is False


def test_missing_scorers_produce_explicit_unavailable_states_and_no_false_success():
    result = Verifier().verify(
        _answer(), _bundle(), minimum_confidence=0.85
    )

    assert result.citation_integrity_passed is True
    assert result.confidence_status == "not_verified"
    assert result.confidence_scorer_status == "unavailable"
    assert result.confidence_reason == "NO_CALIBRATED_CONFIDENCE_SCORER_CONFIGURED"
    assert result.claim_grounding_status == "not_evaluated"
    assert result.grounding_scorer_status == "unavailable"
    assert result.grounding_reason == "NO_GROUNDING_SCORER_CONFIGURED"
    assert result.outcome == "evaluator_unavailable"
    assert result.passed is False


def test_insufficient_evidence_status_is_preserved_and_fails_closed():
    class InsufficientGrounding:
        def evaluate(self, claim, cited_evidence, *, timeout_seconds):
            return GroundingAssessment(
                "insufficient_evidence",
                "SYNTHETIC_INSUFFICIENT_EVIDENCE",
                "synthetic-fixture-oracle",
            )

    result = Verifier(
        confidence_scorer=FixedConfidenceScorer(),
        grounding_scorer=InsufficientGrounding(),
    ).verify(_answer(), _bundle(), minimum_confidence=0.85)

    assert result.citation_integrity_passed is True
    assert result.claim_grounding_status == "insufficient_evidence"
    assert result.grounding_scorer_status == "completed"
    assert result.claim_results[0]["claim_grounding_status"] == "insufficient_evidence"
    assert result.outcome == "failed"
    assert result.passed is False


def test_grounding_results_are_per_claim_and_every_claim_must_be_supported():
    answer = structure_answer_with_explicit_evidence(
        f"First fact. [Evidence ID: {EVIDENCE_ID}]\n"
        f"Second fact. [Evidence ID: {EVIDENCE_ID}]"
    )

    class MixedGrounding:
        def __init__(self):
            self.calls = 0

        def evaluate(self, claim, cited_evidence, *, timeout_seconds):
            self.calls += 1
            label = "supported" if self.calls == 1 else "partially_supported"
            return GroundingAssessment(label, "LABELED_FIXTURE", "synthetic")

    result = Verifier(
        confidence_scorer=FixedConfidenceScorer(),
        grounding_scorer=MixedGrounding(),
    ).verify(answer, _bundle(), minimum_confidence=0.5)

    assert [r["claim_grounding_status"] for r in result.claim_results] == [
        "supported", "partially_supported"
    ]
    assert result.claim_grounding_status == "failed"
    assert result.passed is False


def test_later_evaluator_failure_preserves_prior_claim_assessment_but_not_overall_pass():
    answer = structure_answer_with_explicit_evidence(
        f"First fact. [Evidence ID: {EVIDENCE_ID}]\n"
        f"Second fact. [Evidence ID: {EVIDENCE_ID}]"
    )

    class FailsOnSecondClaim:
        def __init__(self):
            self.calls = 0

        def evaluate(self, claim, cited_evidence, *, timeout_seconds):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError()
            return GroundingAssessment("supported", "SUPPORTED", "synthetic")

    result = Verifier(
        confidence_scorer=FixedConfidenceScorer(),
        grounding_scorer=FailsOnSecondClaim(),
    ).verify(answer, _bundle(), minimum_confidence=0.5)

    assert [claim["claim_grounding_status"] for claim in result.claim_results] == [
        "supported", "evaluator_unavailable"
    ]
    assert [claim["claim_verified"] for claim in result.claim_results] == [True, False]
    assert result.claim_grounding_status == "not_evaluated"
    assert result.grounding_scorer_status == "timeout"
    assert result.outcome == "evaluator_unavailable"
    assert result.passed is False
