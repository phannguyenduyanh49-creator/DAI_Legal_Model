"""Offline experiment contract tests; none establishes legal grounding quality."""

import json

from benchmark.citation_experiments import assess, prompts
from backend.evidence.schemas import Evidence, EvidenceBundle


def _bundle():
    return EvidenceBundle(evidences=[
        Evidence(evidence_id="rag:source-a", source_type="rag", content="Source A."),
        Evidence(evidence_id="rag:source-b", source_type="rag", content="Source B."),
    ])


def test_variants_use_the_same_evidence_ids_and_content():
    variants = prompts(_bundle())
    assert set(variants) == {
        "E0_current", "E1_concise", "E2_example", "E3_structured"
    }
    for prompt in variants.values():
        assert "rag:source-a" in prompt and "rag:source-b" in prompt
        assert '"Source A."' in prompt and '"Source B."' in prompt


def test_structured_experiment_requires_every_claim_to_cite_a_known_id():
    valid = json.dumps({"claims": [
        {"text": "A", "evidence_ids": ["rag:source-a"]},
        {"text": "B", "evidence_ids": ["rag:source-b"]},
    ]})
    assert assess(valid, _bundle(), "E3_structured")["citation_integrity"] == "passed"
    for bad in (
        '{"claims":[{"text":"A","evidence_ids":["rag:source-a","rag:source-b"]},'
        '{"text":"B","evidence_ids":[]}]}',
        '{"claims":[{"text":"A","evidence_ids":["rag:unknown"]}]}',
        '{"claims":[{"text":"","evidence_ids":["rag:source-a"]}]}',
        '{"claims":[{"text":"A","evidence_ids":"rag:source-a"}]}',
    ):
        assert assess(bad, _bundle(), "E3_structured")["citation_integrity"] == "failed"


def test_valid_structured_citation_does_not_claim_semantic_grounding():
    answer = '{"claims":[{"text":"An unsupported number","evidence_ids":["rag:source-a"]}]}'
    result = assess(answer, _bundle(), "E3_structured")
    assert result["citation_integrity"] == "passed"
    assert result["grounding_status"] == "not_evaluated"
