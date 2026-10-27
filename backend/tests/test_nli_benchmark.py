import json
from types import SimpleNamespace

import pytest
import torch

from backend.verifier.evaluation.extract_catalog_candidates import _candidate_rows
from backend.verifier.evaluation.run_nli_benchmark import (
    DatasetError,
    _label_mapping,
    _metrics,
    load_jsonl,
    run_benchmark,
)


def _row(**overrides):
    row = {
        "sample_id": "sample-1",
        "domain": "legal_ai",
        "claim": "Hồ sơ gồm mẫu đơn.",
        "evidence_id": "catalog:record-1",
        "evidence_text": "Hồ sơ gồm mẫu đơn.",
        "gold_label": "",
        "review_status": "unreviewed",
        "source_record_id": "record-1",
        "notes": "candidate",
    }
    row.update(overrides)
    return row


def test_load_jsonl_accepts_unlabeled_candidate(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = load_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0]["gold_label"] == ""
    assert loaded[0]["review_status"] == "unreviewed"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gold_label": "entailment"}, "unsupported gold_label"),
        ({"review_status": "reviewed"}, "invalid review_status"),
        ({"evidence_text": ""}, "must be non-empty"),
    ],
)
def test_load_jsonl_rejects_invalid_dataset_records(tmp_path, overrides, message):
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(_row(**overrides)) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_jsonl(path)


def test_candidate_extractor_leaves_gold_labels_empty_and_uses_catalog_source():
    records = [
        SimpleNamespace(
            category_name="residence",
            target_type="",
            formality_id="F-1",
            source_url="https://example.test/source",
            description="Nộp hồ sơ trực tiếp.",
            execution_steps="Bước một.",
            requirements_conditions=None,
            profile_components=None,
            fees=None,
            results=None,
            legal_basis=None,
        ),
        SimpleNamespace(
            category_name="business",
            target_type="",
            formality_id="F-2",
            source_url=None,
            description="Thực hiện trực tuyến.",
            execution_steps=None,
            requirements_conditions=None,
            profile_components=None,
            fees=None,
            results=None,
            legal_basis=None,
        ),
    ]
    candidates = _candidate_rows(records, limit=80)
    assert len(candidates) == 2
    assert {item["evidence_id"] for item in candidates} == {"catalog:F-1", "catalog:F-2"}
    assert all(item["gold_label"] == "" for item in candidates)
    assert all(item["review_status"] == "unreviewed" for item in candidates)
    first = next(item for item in candidates if item["source_record_id"] == "F-1")
    assert first["source_url"] == "https://example.test/source"


def test_catalog_extraction_limit_is_bounded():
    from backend.verifier.evaluation.extract_catalog_candidates import extract_candidates

    with pytest.raises(ValueError, match="between 50 and 100"):
        extract_candidates(10)


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "entailment", 1: "neutral", 2: "contradiction"},
        {"0": "contradiction", "1": "neutral", "2": "entailment"},
    ],
)
def test_label_mapping_uses_config_label_names_not_assumed_indices(id2label):
    mapped = _label_mapping(id2label)
    assert set(mapped.values()) == {"entailment", "neutral", "contradiction"}


def test_label_mapping_refuses_ambiguous_generic_classifier_labels():
    with pytest.raises(DatasetError, match="does not identify"):
        _label_mapping({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_unreviewed_data_never_produces_metrics():
    report = _metrics([_row(predicted_label="supported")])
    assert report["status"] == "metrics unavailable"
    assert report["unlabeled_sample_count"] == 1


def test_metrics_only_include_expert_confirmed_labels_and_count_errors_as_wrong():
    report = _metrics(
        [
            _row(gold_label="supported", review_status="expert_confirmed", predicted_label="supported"),
            _row(sample_id="sample-2", gold_label="contradicted", review_status="expert_confirmed", predicted_label=None),
            _row(sample_id="sample-3", gold_label="unsupported", review_status="in_review", predicted_label="unsupported"),
        ]
    )
    assert report["reviewed_sample_count"] == 2
    assert report["labeled_not_expert_confirmed_count"] == 1
    assert report["accuracy"] == 0.5
    assert report["confusion_matrix"]["contradicted"]["ERROR_OR_UNMAPPED"] == 1


def test_metrics_report_one_vs_rest_false_positive_rate():
    report = _metrics(
        [
            _row(gold_label="supported", review_status="expert_confirmed", predicted_label="supported"),
            _row(sample_id="sample-2", gold_label="unsupported", review_status="expert_confirmed", predicted_label="supported"),
            _row(sample_id="sample-3", gold_label="contradicted", review_status="expert_confirmed", predicted_label="contradicted"),
            _row(sample_id="sample-4", gold_label="partially_supported", review_status="expert_confirmed", predicted_label="partially_supported"),
        ]
    )

    assert report["per_label"]["supported"]["precision"] == 0.5
    assert report["per_label"]["supported"]["recall"] == 1.0
    assert report["per_label"]["supported"]["false_positive_rate"] == 1 / 3


def test_insufficient_evidence_is_not_collapsed_into_unsupported_metric_class():
    report = _metrics(
        [
            _row(
                gold_label="insufficient_evidence",
                review_status="expert_confirmed",
                predicted_label="unsupported",
            )
        ]
    )

    assert report["confusion_matrix"]["insufficient_evidence"]["unsupported"] == 1
    assert report["per_label"]["insufficient_evidence"]["recall"] == 0.0
    assert report["per_label"]["unsupported"]["support"] == 0


def test_benchmark_saves_probabilities_latency_and_long_input_status(monkeypatch):
    class FakeTokenizer:
        def __call__(self, premise, hypothesis, **kwargs):
            token_count = 513 if "long" in premise else 8
            return {"input_ids": torch.ones((1, token_count), dtype=torch.long)}

    class FakeModel:
        config = SimpleNamespace(
            id2label={0: "contradiction", 1: "neutral", 2: "entailment"},
            max_position_embeddings=512,
            _commit_hash="test-revision",
        )

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[5.0, 0.0, 1.0]]))

    tokenizer, model = FakeTokenizer(), FakeModel()

    class Factory:
        calls = []

        @classmethod
        def from_pretrained(cls, model_id, *, local_files_only):
            cls.calls.append((model_id, local_files_only))
            return tokenizer if len(cls.calls) == 1 else model

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", Factory)
    monkeypatch.setattr(transformers, "AutoModelForSequenceClassification", Factory)
    results, report = run_benchmark(
        [_row(), _row(sample_id="long", evidence_text="long evidence")]
    )
    assert Factory.calls and all(local_only for _, local_only in Factory.calls)
    assert results[0]["nli_label"] == "contradiction"
    assert results[0]["predicted_label"] == "contradicted"
    assert set(results[0]["probabilities"]) == {"entailment", "neutral", "contradiction"}
    assert results[0]["latency_ms"] is not None
    assert results[1]["status"] == "input_too_long"
    assert results[1]["latency_ms"] is None
    assert report["model_revision"] == "test-revision"
    assert report["device"] == "cpu"
    assert report["claim_count"] == 2
    assert report["cited_evidence_item_count"] == 2
    assert report["claim_evidence_pair_count"] == 2
    assert report["completed_inference_pair_count"] == 1
    assert report["model_load_latency_ms"] >= 0
    assert report["evaluation_latency_ms"] >= 0
    assert report["host_memory"]["process_rss_after_model_load_bytes"] > 0


def test_benchmark_rejects_unsupported_device_before_loading_model():
    with pytest.raises(DatasetError, match="device must be cpu or cuda"):
        run_benchmark([_row()], device="tpu")


def test_benchmark_records_per_sample_model_errors(monkeypatch):
    class FakeTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.ones((1, 8), dtype=torch.long)}

    class BrokenModel:
        config = SimpleNamespace(
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
            max_position_embeddings=512,
        )

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            raise RuntimeError("internal details not exported")

    class Factory:
        calls = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls += 1
            return FakeTokenizer() if cls.calls == 1 else BrokenModel()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", Factory)
    monkeypatch.setattr(transformers, "AutoModelForSequenceClassification", Factory)
    results, report = run_benchmark([_row()])
    assert results[0]["status"] == "error"
    assert results[0]["error_type"] == "RuntimeError"
    assert "internal details" not in json.dumps(results)
    assert report["prediction_status_counts"] == {"error": 1}
