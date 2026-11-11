"""Small, dependency-light helpers for model device configuration and telemetry."""

from __future__ import annotations

import os
import time
from typing import Any, Mapping


SUPPORTED_MODEL_DEVICES = {"cpu", "cuda"}


class DeadlineStoppingCriteria:
    """Cooperatively stop generation at the next token after a monotonic deadline."""

    def __init__(self, deadline_monotonic: float, *, clock=None) -> None:
        self.deadline_monotonic = deadline_monotonic
        self.clock = clock or time.monotonic
        self.expired = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self.expired = self.clock() >= self.deadline_monotonic
        return self.expired

    def __deepcopy__(self, memo):
        # Transformers clones StoppingCriteriaList before generate(). Keep the
        # request-scoped stop flag observable by the API handler.
        return self


def monotonic_request_deadline(deadline_epoch: float | None, *, now_epoch=None, now_monotonic=None) -> float | None:
    """Convert a caller's wall-clock deadline once, avoiding later clock changes."""
    if deadline_epoch is None:
        return None
    import math
    if isinstance(deadline_epoch, bool) or not isinstance(deadline_epoch, (int, float)) or not math.isfinite(deadline_epoch):
        raise ValueError("request deadline must be a finite Unix timestamp")
    epoch = time.time() if now_epoch is None else now_epoch
    monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    return monotonic + max(0.0, deadline_epoch - epoch)


def generation_finish_reason(
    last_token_id: int | None,
    output_token_count: int,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | tuple[int, ...] | None,
) -> str:
    """Classify an observed generation stop without assuming every cap is EOS."""
    if output_token_count <= 0:
        return "empty"
    ids = (
        (eos_token_ids,)
        if isinstance(eos_token_ids, int)
        else tuple(eos_token_ids or ())
    )
    if last_token_id is not None and last_token_id in ids:
        return "eos"
    if output_token_count >= max_new_tokens:
        return "max_new_tokens"
    return "other"


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
