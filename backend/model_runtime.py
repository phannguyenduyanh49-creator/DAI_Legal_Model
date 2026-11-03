"""Small, dependency-light helpers for model device configuration and telemetry."""

from __future__ import annotations

import os
from typing import Any, Mapping


SUPPORTED_MODEL_DEVICES = {"cpu", "cuda"}


def configured_model_device(
    env_name: str,
    default: str,
    *,
    environ: Mapping[str, str] | None = None,
    cuda_available: bool,
) -> str:
    """Read and validate a model device without silently falling back."""
    values = os.environ if environ is None else environ
    configured = values.get(env_name)
    device = str(configured if configured is not None and configured.strip() else default)
    device = device.strip().lower()
    if device not in SUPPORTED_MODEL_DEVICES:
        raise RuntimeError(
            f"{env_name} must be one of: cpu, cuda (received {device!r})"
        )
    if device == "cuda" and not cuda_available:
        raise RuntimeError(
            f"{env_name}=cuda was requested, but CUDA is unavailable; "
            "set the variable to cpu explicitly to use CPU execution"
        )
    return device


def pytorch_cuda_memory_snapshot(torch_module: Any) -> dict[str, int | None]:
    """Return process-wide PyTorch CUDA allocator counters, not model-only usage."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": int(cuda.memory_allocated()),
        "reserved_bytes": int(cuda.memory_reserved()),
    }


def pytorch_cuda_peak_snapshot(torch_module: Any) -> dict[str, int | None]:
    """Return process-wide PyTorch CUDA peak allocator counters."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(cuda.max_memory_reserved()),
    }
