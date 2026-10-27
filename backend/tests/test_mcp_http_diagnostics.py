import urllib.error
import sys
from io import BytesIO
from types import SimpleNamespace

import mcp_server
from mcp_server import DVC_DETAIL_API, DVC_SEARCH_API, post_json


class FakeResponse:
    status = 200
    headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"code":"OK","data":{"rows":[]}}'


def test_dvc_http_log_has_trace_step_status_elapsed_and_retry_count(monkeypatch, capsys):
    monkeypatch.setenv("DAI_REQUEST_ID", "trace_legal_http_test")
    monkeypatch.setattr("mcp_server.urllib.request.urlopen", lambda *_a, **_k: FakeResponse())

    result = post_json(DVC_SEARCH_API, {"q": "safe query"})

    assert result["code"] == "OK"
    output = capsys.readouterr().err
    assert "request_id= trace_legal_http_test" in output
    assert "step= search" in output
    assert "status_code= 200" in output
    assert "attempt=1 retry_count=0" in output
    assert "elapsed_ms=" in output


def test_dvc_http_error_is_logged_without_retry(monkeypatch, capsys):
    monkeypatch.setenv("DAI_REQUEST_ID", "trace_legal_http_error")

    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            DVC_DETAIL_API, 503, "unavailable", {}, BytesIO(b"details")
        )

    monkeypatch.setattr("mcp_server.urllib.request.urlopen", fail)
    try:
        post_json(DVC_DETAIL_API, {"id": "safe-id"})
    except urllib.error.HTTPError:
        pass
    else:
        raise AssertionError("Expected upstream HTTPError")

    output = capsys.readouterr().err
    assert "request_id= trace_legal_http_error" in output
    assert "step= detail" in output
    assert "status_code= 503" in output
    assert "attempt=1 retry_count=0" in output
    assert "details" not in output


def test_embedding_model_load_is_deferred_until_local_embedding_tool(monkeypatch):
    calls = []

    class FakeEmbeddingModel:
        def __init__(self, name):
            calls.append(name)

    monkeypatch.setattr(mcp_server, "embedding_model", None)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeEmbeddingModel),
    )

    assert calls == []
    assert isinstance(mcp_server.get_embedding_model(), FakeEmbeddingModel)
    assert calls == [mcp_server.EMBEDDING_MODEL]
