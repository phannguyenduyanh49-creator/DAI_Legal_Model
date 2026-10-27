from types import SimpleNamespace
import sys

import pytest
import torch

from backend.answer.schemas import Claim
from backend.evidence.schemas import Evidence
from backend.verifier.scoring import ScorerUnavailableError
from backend.verifier.transformers_nli import (
    EXPERIMENTAL_MODEL_ID,
    TransformersNliGroundingScorer,
)


class FakeTokenizer:
    def __init__(self, length=8):
        self.length = length
        self.pairs = None
        self.kwargs = None

    def __call__(self, premises, hypotheses, **kwargs):
        self.pairs = list(zip(premises, hypotheses))
        self.kwargs = kwargs
        return {"input_ids": torch.ones((len(premises), self.length), dtype=torch.long)}


class FakeModel:
    def __init__(self, relations):
        self.config = SimpleNamespace(
            id2label={
                0: "entailment",
                1: "neutral",
                2: "contradiction",
                **({3: "partially_supported"} if len(relations[0]) == 4 else {}),
            },
            max_position_embeddings=512,
        )
        self.relations = relations
        self.device = None
        self.evaluating = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True

    def __call__(self, input_ids):
        return SimpleNamespace(logits=torch.tensor(self.relations, dtype=torch.float32))


def make_scorer(relations, *, tokenizer=None, model=None):
    tokenizer = tokenizer or FakeTokenizer()
    model = model or FakeModel(relations)
    return TransformersNliGroundingScorer(tokenizer, model), tokenizer, model


@pytest.mark.parametrize(
    ("relation_index", "expected"),
    [(0, "supported"), (1, "unsupported"), (2, "contradicted")],
)
def test_experimental_nli_adapter_maps_multilingual_nli_labels(relation_index, expected):
    logits = [[0.0, 0.0, 0.0]]
    logits[0][relation_index] = 2.0
    scorer, tokenizer, model = make_scorer(logits)
    claim = Claim("Người nộp đơn phải gửi Mẫu A.", ["rag:allowed"])
    evidence = Evidence(
        evidence_id="rag:allowed",
        source_type="rag",
        content="Nội dung chính xác của Evidence được trích dẫn.",
    )

    result = scorer.evaluate(claim, (evidence,), timeout_seconds=5.0)

    assert result.status == expected
    assert result.scorer_id == EXPERIMENTAL_MODEL_ID
    assert tokenizer.pairs == [
        (evidence.content, claim.claim)
    ]
    assert tokenizer.kwargs["truncation"] is False
    assert model.device == "cpu"
    assert model.evaluating is True


def test_nli_adapter_can_normalize_partial_label_from_a_matching_classifier():
    scorer, _, _ = make_scorer([[0.0, 0.0, 0.0, 3.0]])
    result = scorer.evaluate(
        Claim("Composite claim", ["rag:a"]),
        (Evidence("rag:a", "rag", content="Only one part."),),
        timeout_seconds=5.0,
    )
    assert result.status == "partially_supported"


def test_neutral_is_not_misrepresented_as_partial_support():
    scorer, _, _ = make_scorer([[0.0, 3.0, 0.0]])
    result = scorer.evaluate(
        Claim("Claim", ["rag:a"]),
        (Evidence("rag:a", "rag", content="Unrelated Evidence."),),
        timeout_seconds=5.0,
    )
    assert result.status == "unsupported"
    assert result.reason_code == "NLI_NEUTRAL_NOT_ENTAILED"


def test_adapter_rejects_uncited_or_missing_evidence_objects():
    scorer, _, _ = make_scorer([[3.0, 0.0, 0.0]])
    with pytest.raises(ScorerUnavailableError) as error:
        scorer.evaluate(
            Claim("Claim", ["rag:claimed"]),
            (Evidence("rag:other", "rag", content="Uncited text."),),
            timeout_seconds=5.0,
        )
    assert error.value.reason_code == "NLI_CITATION_EVIDENCE_MISMATCH"


def test_one_contradicting_cited_evidence_fails_even_if_another_entails():
    # One row per cited Evidence item: entailment and contradiction coexist.
    scorer, _, _ = make_scorer([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    result = scorer.evaluate(
        Claim("Claim", ["rag:a", "rag:b"]),
        (
            Evidence("rag:a", "rag", content="Supporting passage."),
            Evidence("rag:b", "rag", content="Conflicting passage."),
        ),
        timeout_seconds=5.0,
    )
    assert result.status == "contradicted"


def test_adapter_refuses_to_truncate_long_legal_evidence():
    scorer, _, _ = make_scorer([[3.0, 0.0, 0.0]], tokenizer=FakeTokenizer(length=513))
    with pytest.raises(ScorerUnavailableError) as error:
        scorer.evaluate(
            Claim("Claim", ["rag:a"]),
            (Evidence("rag:a", "rag", content="Long legal text."),),
            timeout_seconds=5.0,
        )
    assert error.value.reason_code == "NLI_INPUT_EXCEEDS_MODEL_LIMIT"


def test_adapter_discards_results_that_finish_after_timeout():
    import time

    class SlowModel(FakeModel):
        def __call__(self, input_ids):
            time.sleep(0.02)
            return super().__call__(input_ids)

    model = SlowModel([[3.0, 0.0, 0.0]])
    scorer, _, _ = make_scorer([[3.0, 0.0, 0.0]], model=model)
    with pytest.raises(TimeoutError):
        scorer.evaluate(
            Claim("Claim", ["rag:a"]),
            (Evidence("rag:a", "rag", content="Text."),),
            timeout_seconds=0.001,
        )


def test_adapter_rejects_checkpoint_with_unrecognized_label_schema():
    model = FakeModel([[1.0, 0.0]])
    model.config.id2label = {0: "yes", 1: "no"}
    with pytest.raises(ScorerUnavailableError) as error:
        TransformersNliGroundingScorer(FakeTokenizer(), model)
    assert error.value.reason_code == "NLI_LABEL_SCHEMA_UNSUPPORTED"


def test_pretrained_factory_is_cache_only_by_default(monkeypatch):
    calls = []
    tokenizer = FakeTokenizer()
    model = FakeModel([[3.0, 0.0, 0.0]])

    class FakeAuto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return tokenizer if len(calls) == 1 else model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeAuto,
            AutoModelForSequenceClassification=FakeAuto,
        ),
    )
    scorer = TransformersNliGroundingScorer.from_pretrained()

    assert scorer.device == "cpu"
    assert len(calls) == 2
    assert all(call_args["local_files_only"] is True for _, call_args in calls)
    assert all(path == EXPERIMENTAL_MODEL_ID for path, _ in calls)


def test_adapter_rejects_malformed_model_logits():
    class MalformedModel(FakeModel):
        def __call__(self, input_ids):
            return SimpleNamespace(logits=torch.tensor([1.0, 0.0, -1.0]))

    scorer, _, _ = make_scorer(
        [[3.0, 0.0, 0.0]], model=MalformedModel([[3.0, 0.0, 0.0]])
    )
    with pytest.raises(ScorerUnavailableError) as error:
        scorer.evaluate(
            Claim("Claim", ["rag:a"]),
            (Evidence("rag:a", "rag", content="Text."),),
            timeout_seconds=5.0,
        )
    assert error.value.reason_code == "NLI_MODEL_INVALID_OUTPUT"
