"""Bounded, process-local admission control for expensive model inference."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path
from threading import BoundedSemaphore
import tempfile
from time import monotonic
import time
from typing import Iterator

from fastapi import HTTPException


class InferenceBusy(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ModelInstanceAlreadyRunning(RuntimeError):
    pass


class ModelInstanceLock:
    """Hold an OS file lock for the lifetime of one Model API process."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        configured_path = path or os.getenv("MODEL_API_INSTANCE_LOCK_PATH")
        self.path = Path(configured_path) if configured_path else (
            Path(tempfile.gettempdir()) / "dai-legal-model-api-qwen-lora-v4.lock"
        )
        self._handle = None

    def acquire(self) -> "ModelInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise ModelInstanceAlreadyRunning(
                f"Another Model API process holds the model lock at {self.path}"
            ) from exc
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "ModelInstanceLock":
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.release()


class InferenceAdmission:
    """Limit active generations and the number of requests allowed to wait."""

    def __init__(
        self,
        *,
        concurrency: int = 1,
        queue_capacity: int = 2,
        queue_timeout_seconds: float = 120.0,
    ):
        if concurrency < 1 or queue_capacity < 0 or queue_timeout_seconds <= 0:
            raise ValueError("Invalid inference admission configuration")
        self.concurrency = concurrency
        self.queue_capacity = queue_capacity
        self.queue_timeout_seconds = queue_timeout_seconds
        self._admitted = BoundedSemaphore(concurrency + queue_capacity)
        self._active = BoundedSemaphore(concurrency)

    @contextmanager
    def enter(self) -> Iterator[float]:
        if not self._admitted.acquire(blocking=False):
            raise InferenceBusy("queue_full")
        active_acquired = False
        started = monotonic()
        try:
            active_acquired = self._active.acquire(
                timeout=self.queue_timeout_seconds
            )
            if not active_acquired:
                raise InferenceBusy("queue_timeout")
            yield (monotonic() - started) * 1000
        finally:
            if active_acquired:
                self._active.release()
            self._admitted.release()


def with_inference_admission(
    gate: InferenceAdmission,
    *,
    concurrency: int,
    queue_capacity: int,
):
    """Apply admission control and translate saturation into HTTP 503."""

    def decorate(endpoint):
        @wraps(endpoint)
        def wrapped(*args, **kwargs):
            request_id = kwargs.get("request_id") or "not_provided"
            started = time.perf_counter()
            try:
                with gate.enter() as queue_wait_ms:
                    print(
                        "[MODEL QUEUE]",
                        f"request_id={request_id}",
                        "status=admitted",
                        f"queue_wait_ms={round(queue_wait_ms, 2)}",
                        f"concurrency={concurrency}",
                        f"queue_capacity={queue_capacity}",
                        flush=True,
                    )
                    return endpoint(*args, **kwargs)
            except InferenceBusy as exc:
                print(
                    "[MODEL QUEUE]",
                    f"request_id={request_id}",
                    "status=busy",
                    f"reason={exc.reason}",
                    f"waited_ms={round((time.perf_counter() - started) * 1000, 2)}",
                    flush=True,
                )
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "MODEL_INFERENCE_BUSY",
                        "reason": exc.reason,
                        "message": "Model inference is busy. Retry the request shortly.",
                    },
                    headers={"Retry-After": "5"},
                ) from exc

        return wrapped

    return decorate

