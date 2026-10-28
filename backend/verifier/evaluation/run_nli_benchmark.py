"""Run the experimental Vietnamese NLI checkpoint against reviewed JSONL.

The model is loaded from the local Hugging Face cache only and runs on CPU by
default. This is an offline evaluation tool; it is not a production verifier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from backend.verifier.transformers_nli import EXPERIMENTAL_MODEL_ID


GOLD_LABELS = (
    "supported",
    "unsupported",
    "contradicted",
    "partially_supported",
    "insufficient_evidence",
)
REVIEWED_STATUS = "expert_confirmed"
NLI_TO_GROUNDING = {
    "entailment": "supported",
    "neutral": "unsupported",
    "contradiction": "contradicted",
}


class DatasetError(ValueError):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"line {line_number}: record must be an object")
            required = (
                "sample_id", "domain", "claim", "evidence_id", "evidence_text",
                "gold_label", "review_status", "source_record_id", "notes",
            )
            missing = [field for field in required if field not in row]
            if missing:
                raise DatasetError(f"line {line_number}: missing fields {missing}")
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_ids:
                raise DatasetError(f"line {line_number}: sample_id must be unique and non-empty")
            seen_ids.add(sample_id)
            if not all(isinstance(row[key], str) for key in ("claim", "evidence_id", "evidence_text")):
                raise DatasetError(f"line {line_number}: claim and Evidence fields must be strings")
            label = row["gold_label"]
            if label not in ("", *GOLD_LABELS):
                raise DatasetError(f"line {line_number}: unsupported gold_label {label!r}")
            if row["review_status"] not in ("unreviewed", "in_review", "expert_confirmed", "needs_adjudication"):
                raise DatasetError(f"line {line_number}: invalid review_status")
            if not row["claim"].strip() or not row["evidence_id"].strip() or not row["evidence_text"].strip():
                raise DatasetError(f"line {line_number}: claim, evidence_id and evidence_text must be non-empty")
            rows.append(row)
    return rows


def _label_name(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "entailment": "entailment", "label_0": "label_0",
        "neutral": "neutral", "label_1": "label_1",
        "contradiction": "contradiction", "label_2": "label_2",
    }
    return aliases.get(normalized)


def _label_mapping(id2label: dict[Any, Any]) -> dict[int, str]:
    mapped: dict[int, str] = {}
    for raw_id, raw_label in id2label.items():
        try:
            label_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        name = _label_name(raw_label)
        if name:
            mapped[label_id] = name
    canonical = {name for name in mapped.values()}
    # Generic LABEL_0 names carry no semantic ordering. Refuse to guess which
    # index means entailment, neutral, or contradiction.
    if canonical != {"entailment", "neutral", "contradiction"}:
        raise DatasetError("checkpoint id2label does not identify all three NLI relations")
    return mapped


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = [
        row for row in rows
        if row.get("gold_label") in GOLD_LABELS and row.get("review_status") == REVIEWED_STATUS
    ]
    unreviewed = [row for row in rows if not row.get("gold_label")]
    nonconfirmed = [
        row for row in rows
        if row.get("gold_label") in GOLD_LABELS and row.get("review_status") != REVIEWED_STATUS
    ]
    if not reviewed:
        return {
            "status": "metrics unavailable",
            "reason": "No valid gold labels with review_status=expert_confirmed",
            "reviewed_sample_count": 0,
            "unlabeled_sample_count": len(unreviewed),
            "labeled_not_expert_confirmed_count": len(nonconfirmed),
        }

    confusion_labels = [*GOLD_LABELS, "ERROR_OR_UNMAPPED"]
    matrix = {gold: {pred: 0 for pred in confusion_labels} for gold in GOLD_LABELS}
    for row in reviewed:
        predicted = row.get("predicted_label")
        if predicted not in GOLD_LABELS:
            predicted = "ERROR_OR_UNMAPPED"
        matrix[row["gold_label"]][predicted] += 1
    per_label = {}
    f1s = []
    for label in GOLD_LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in GOLD_LABELS if other != label)
        fn = sum(matrix[label][other] for other in confusion_labels if other != label)
        tn = len(reviewed) - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        false_positive_rate = fp / (fp + tn) if fp + tn else None
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "false_positive_rate": false_positive_rate,
            "f1": f1,
            "support": sum(matrix[label].values()),
        }
        f1s.append(f1)
    correct = sum(matrix[label][label] for label in GOLD_LABELS)
    return {
        "status": "computed_on_expert_confirmed_only",
        "reviewed_sample_count": len(reviewed),
        "unlabeled_sample_count": len(unreviewed),
        "labeled_not_expert_confirmed_count": len(nonconfirmed),
        "confusion_matrix": matrix,
        "per_label": per_label,
        "macro_f1": sum(f1s) / len(f1s),
        "accuracy": correct / len(reviewed),
    }


def _latency_summary(results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = sorted(
        float(row["latency_ms"]) for row in results
        if isinstance(row.get("latency_ms"), (int, float)) and math.isfinite(row["latency_ms"])
    )
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[p95_index],
    }


def run_benchmark(
    rows: list[dict[str, Any]],
    *,
    model_id: str = EXPERIMENTAL_MODEL_ID,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Explicit cache-only loading. Device selection must be requested.
    import torch
    import psutil
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device not in {"cpu", "cuda"}:
        raise DatasetError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise DatasetError("CUDA was requested but is unavailable")

    process = psutil.Process()
    rss_before_model_load_bytes = process.memory_info().rss
    host_memory_total_bytes = psutil.virtual_memory().total
    host_memory_available_before_load_bytes = psutil.virtual_memory().available
    model_load_started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
    model.to(device)
    model.eval()
    model_load_latency_ms = (time.perf_counter() - model_load_started) * 1000
    rss_after_model_load_bytes = process.memory_info().rss
    cuda_model_allocated_bytes = (
        torch.cuda.memory_allocated() if device == "cuda" else None
    )
    relation_by_index = _label_mapping(model.config.id2label)
    max_tokens = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        max_tokens = getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_tokens, int) or max_tokens > 100_000:
            max_tokens = None

    evaluation_started = time.perf_counter()
    results = []
    for row in rows:
        result = {
            "sample_id": row["sample_id"],
            "domain": row["domain"],
            "evidence_id": row["evidence_id"],
            "gold_label": row["gold_label"],
            "review_status": row["review_status"],
            "predicted_label": None,
            "nli_label": None,
            "probabilities": None,
            "input_tokens": None,
            "latency_ms": None,
            "status": "pending",
            "error_type": None,
        }
        try:
            encoded = tokenizer(
                row["evidence_text"],
                row["claim"],
                return_tensors="pt",
                truncation=False,
            )
            input_tokens = int(encoded["input_ids"].shape[-1])
            result["input_tokens"] = input_tokens
            if max_tokens is not None and input_tokens > max_tokens:
                result["status"] = "input_too_long"
                result["error_type"] = "InputExceedsModelLimit"
                results.append(result)
                continue
            if device == "cuda":
                encoded = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)[0]
            if device == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000
            if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(relation_by_index):
                raise ValueError("model logits do not match id2label schema")
            pred_index = int(torch.argmax(probabilities).item())
            nli_label = relation_by_index.get(pred_index)
            if nli_label not in NLI_TO_GROUNDING:
                raise ValueError("model predicted an unmapped NLI label")
            result.update(
                {
                    "nli_label": nli_label,
                    "predicted_label": NLI_TO_GROUNDING[nli_label],
                    "probabilities": {
                        relation_by_index[index]: float(probabilities[index].item())
                        for index in sorted(relation_by_index)
                    },
                    "latency_ms": latency_ms,
                    "status": "completed",
                }
            )
        except Exception as exc:  # per-item fault isolation; retain sample result
            result["status"] = "error"
            result["error_type"] = type(exc).__name__
        results.append(result)

    metrics_input = [
        {**source, **prediction}
        for source, prediction in zip(rows, results)
    ]
    evaluation_latency_ms = (time.perf_counter() - evaluation_started) * 1000
    report = {
        "model_id": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": device,
        "local_files_only": True,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count_logical": os.cpu_count(),
            "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None,
        },
        "model_load_latency_ms": model_load_latency_ms,
        "evaluation_latency_ms": evaluation_latency_ms,
        "host_memory": {
            "total_bytes": host_memory_total_bytes,
            "available_before_model_load_bytes": host_memory_available_before_load_bytes,
            "process_rss_before_model_load_bytes": rss_before_model_load_bytes,
            "process_rss_after_model_load_bytes": rss_after_model_load_bytes,
            "process_rss_after_evaluation_bytes": process.memory_info().rss,
        },
        "cuda_model_allocated_bytes": cuda_model_allocated_bytes,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else None
        ),
        "sample_count": len(rows),
        "claim_count": len(rows),
        "cited_evidence_item_count": len(rows),
        "claim_evidence_pair_count": len(rows),
        "completed_inference_pair_count": sum(
            row["status"] == "completed" for row in results
        ),
        "prediction_status_counts": dict(Counter(row["status"] for row in results)),
        "latency": _latency_summary(results),
        "metrics": _metrics(metrics_input),
        "limitations": [
            "NLI softmax values are not legal correctness probabilities or calibrated confidence.",
            "Neutral maps to unsupported; the checkpoint cannot distinguish partial support.",
            "Only the cited Evidence text in each row is evaluated.",
            "Results require human-reviewed labels and do not imply production readiness.",
        ],
    }
    return results, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--predictions", type=Path, default=Path("nli_predictions.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("nli_report.json"))
    parser.add_argument("--model-id", default=EXPERIMENTAL_MODEL_ID)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    rows = load_jsonl(args.dataset)
    results, report = run_benchmark(rows, model_id=args.model_id, device=args.device)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8", newline="\n") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"predictions={args.predictions}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
