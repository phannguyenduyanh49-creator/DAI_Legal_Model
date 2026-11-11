from types import SimpleNamespace
from copy import deepcopy

import pytest

from backend.model_runtime import (
    DeadlineStoppingCriteria,
    generation_finish_reason,
    monotonic_request_deadline,
    configured_model_device,
    pytorch_cuda_memory_snapshot,
    pytorch_cuda_peak_snapshot,
)


def test_generation_finish_reason_distinguishes_eos_from_output_cap():
    assert generation_finish_reason(2, 37, 256, 2) == "eos"
    assert generation_finish_reason(7, 256, 256, [2, 3]) == "max_new_tokens"
    assert generation_finish_reason(None, 0, 256, 2) == "empty"
    assert generation_finish_reason(7, 4, 256, 2) == "other"


def test_deadline_stopping_is_request_scoped_and_checks_each_generation_step():
    now = [10.0]
    first = DeadlineStoppingCriteria(12.0, clock=lambda: now[0])
    second = DeadlineStoppingCriteria(20.0, clock=lambda: now[0])
    assert first(None, None) is False
    now[0] = 12.0
    assert first(None, None) is True
    assert first.expired is True
    assert second(None, None) is False
    assert deepcopy(first) is first


def test_request_deadline_conversion_rejects_invalid_values():
    assert monotonic_request_deadline(110.0, now_epoch=100.0, now_monotonic=50.0) == 60.0
    assert monotonic_request_deadline(90.0, now_epoch=100.0, now_monotonic=50.0) == 50.0
    for invalid in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            monotonic_request_deadline(invalid)


def test_device_defaults_preserve_existing_cuda_placement():
    assert configured_model_device(
        "EMBEDDING_DEVICE", "cuda", environ={}, cuda_available=True
    ) == "cuda"
    assert configured_model_device(
        "RERANKER_DEVICE", "cpu", environ={}, cuda_available=False
    ) == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda", " CPU ", "CUDA"])
def test_device_setting_accepts_only_cpu_or_cuda(device):
    assert configured_model_device(
        "MODEL_DEVICE", "cpu", environ={"MODEL_DEVICE": device}, cuda_available=True
    ) == device.strip().lower()


def test_device_setting_rejects_invalid_value_without_fallback():
    with pytest.raises(RuntimeError, match="must be one of: cpu, cuda"):
        configured_model_device(
            "EMBEDDING_DEVICE",
            "cuda",
            environ={"EMBEDDING_DEVICE": "auto"},
            cuda_available=True,
        )


def test_cuda_setting_fails_clearly_when_cuda_is_unavailable():
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        configured_model_device(
            "RERANKER_DEVICE",
            "cpu",
            environ={"RERANKER_DEVICE": "cuda"},
            cuda_available=False,
        )


def test_allocator_metrics_are_explicitly_unavailable_without_cuda():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert pytorch_cuda_memory_snapshot(torch) == {
        "allocated_bytes": None,
        "reserved_bytes": None,
    }
    assert pytorch_cuda_peak_snapshot(torch) == {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }


def test_allocator_metrics_report_process_wide_pytorch_counters():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=lambda: 10,
            memory_reserved=lambda: 20,
            max_memory_allocated=lambda: 30,
            max_memory_reserved=lambda: 40,
        )
    )
    assert pytorch_cuda_memory_snapshot(torch) == {
        "allocated_bytes": 10,
        "reserved_bytes": 20,
    }
    assert pytorch_cuda_peak_snapshot(torch) == {
        "peak_allocated_bytes": 30,
        "peak_reserved_bytes": 40,
    }
