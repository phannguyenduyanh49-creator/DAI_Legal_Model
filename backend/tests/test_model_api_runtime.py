from threading import Event, Thread

import pytest

from fastapi import HTTPException

from backend.model_api_runtime import (
    InferenceAdmission,
    InferenceBusy,
    ModelInstanceAlreadyRunning,
    ModelInstanceLock,
    with_inference_admission,
)


def test_admission_rejects_requests_above_active_and_queue_capacity():
    gate = InferenceAdmission(concurrency=1, queue_capacity=0, queue_timeout_seconds=1)
    entered = Event()
    release = Event()

    def active_request():
        with gate.enter():
            entered.set()
            release.wait(timeout=2)

    thread = Thread(target=active_request)
    thread.start()
    assert entered.wait(timeout=1)

    with pytest.raises(InferenceBusy, match="queue_full"):
        with gate.enter():
            pass

    release.set()
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_admission_times_out_waiting_request_and_releases_capacity():
    gate = InferenceAdmission(concurrency=1, queue_capacity=1, queue_timeout_seconds=0.02)
    entered = Event()
    release = Event()

    def active_request():
        with gate.enter():
            entered.set()
            release.wait(timeout=2)

    thread = Thread(target=active_request)
    thread.start()
    assert entered.wait(timeout=1)
    with pytest.raises(InferenceBusy, match="queue_timeout"):
        with gate.enter():
            pass
    release.set()
    thread.join(timeout=1)
    assert not thread.is_alive()

    with gate.enter() as waited_ms:
        assert waited_ms >= 0


def test_admission_releases_slots_when_inference_raises():
    gate = InferenceAdmission(concurrency=1, queue_capacity=0, queue_timeout_seconds=1)
    with pytest.raises(RuntimeError, match="generation failed"):
        with gate.enter():
            raise RuntimeError("generation failed")
    with gate.enter():
        pass


def test_admission_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        InferenceAdmission(concurrency=0)


def test_http_wrapper_returns_retryable_busy_response_when_queue_is_full():
    gate = InferenceAdmission(concurrency=1, queue_capacity=0, queue_timeout_seconds=1)

    @with_inference_admission(gate, concurrency=1, queue_capacity=0)
    def endpoint(request_id=None):
        return {"request_id": request_id}

    with gate.enter():
        with pytest.raises(HTTPException) as raised:
            endpoint(request_id="trace-test")

    assert raised.value.status_code == 503
    assert raised.value.headers["Retry-After"] == "5"
    assert raised.value.detail["code"] == "MODEL_INFERENCE_BUSY"


def test_model_instance_lock_rejects_a_second_lock_owner(tmp_path):
    lock_path = tmp_path / "model-api.lock"
    first = ModelInstanceLock(lock_path).acquire()
    try:
        with pytest.raises(ModelInstanceAlreadyRunning):
            ModelInstanceLock(lock_path).acquire()
    finally:
        first.release()

    second = ModelInstanceLock(lock_path).acquire()
    second.release()
