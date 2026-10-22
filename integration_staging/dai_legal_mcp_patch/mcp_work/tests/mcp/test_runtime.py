import os
import time

import pytest

from backend.mcp.runtime import (
    ToolNotAllowedError,
    ToolTimeoutError,
    execute_tool,
    get_allowed_tools,
    guarded_tool,
    is_tool_allowed,
    parse_allowed_tools,
)


def test_default_allowlist_contains_only_registered_tools(monkeypatch):
    monkeypatch.delenv("MCP_ALLOWED_TOOLS", raising=False)
    assert get_allowed_tools() == frozenset({
        "search_legal_documents",
        "search_procedure",
        "search_dichvucong",
    })
    assert is_tool_allowed("search_dichvucong")
    assert not is_tool_allowed("delete_everything")


def test_allowlist_environment_is_exact_and_empty_means_deny_all(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_TOOLS", "search_dichvucong, search_procedure")
    assert get_allowed_tools() == frozenset({"search_dichvucong", "search_procedure"})
    assert is_tool_allowed("search_dichvucong")
    assert not is_tool_allowed("search_legal_documents")
    monkeypatch.setenv("MCP_ALLOWED_TOOLS", "")
    assert get_allowed_tools() == frozenset()


def test_denied_tool_is_not_executed():
    called = []
    def tool():
        called.append(True)
        return "should not run"

    with pytest.raises(ToolNotAllowedError):
        execute_tool("search_dichvucong", tool, allowed_tools=set())
    assert called == []


def test_timeout_is_reported():
    def slow_tool():
        time.sleep(0.15)
        return "late"

    with pytest.raises(ToolTimeoutError, match="timeout"):
        execute_tool(
            "search_dichvucong",
            slow_tool,
            allowed_tools={"search_dichvucong"},
            timeout_seconds=0.01,
        )


def test_tool_exception_propagates_to_caller():
    def broken_tool():
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError, match="upstream failed"):
        execute_tool(
            "search_procedure",
            broken_tool,
            allowed_tools={"search_procedure"},
            timeout_seconds=1,
        )


def test_successful_result_is_returned_unchanged():
    result = {"items": [1, 2], "source": "test"}
    assert execute_tool(
        "search_legal_documents",
        lambda query: result,
        "luật",
        allowed_tools={"search_legal_documents"},
        timeout_seconds=1,
    ) == result


def test_guarded_tool_returns_stable_text_for_permission_timeout_and_error(monkeypatch):
    monkeypatch.setenv("MCP_ALLOWED_TOOLS", "")

    @guarded_tool("search_dichvucong")
    def denied(keyword):
        return "ok"

    assert denied("hộ chiếu").startswith("Lỗi phân quyền MCP:")

    monkeypatch.setenv("MCP_ALLOWED_TOOLS", "search_dichvucong")
    monkeypatch.setenv("MCP_TOOL_TIMEOUT_SECONDS", "0.01")

    @guarded_tool("search_dichvucong")
    def timed_out(keyword):
        time.sleep(0.15)
        return "late"

    assert timed_out("hộ chiếu").startswith("Lỗi timeout MCP:")

    monkeypatch.setenv("MCP_TOOL_TIMEOUT_SECONDS", "1")

    @guarded_tool("search_dichvucong")
    def failed(keyword):
        raise ValueError("bad input")

    assert failed("x") == "Lỗi MCP tool 'search_dichvucong': ValueError: bad input"

    @guarded_tool("search_dichvucong")
    def successful(keyword):
        return {"answer": "ok"}

    assert successful("x") == "{'answer': 'ok'}"


def test_call_contract_checks_permission_and_normalizes_text():
    import asyncio
    from types import SimpleNamespace
    from backend.mcp.contract import MCPToolCallError, call_mcp_tool

    class Session:
        async def call_tool(self, name, arguments):
            assert name == "search_dichvucong"
            assert arguments == {"keyword": "hộ chiếu"}
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text="Tên thủ tục: Cấp hộ chiếu"), SimpleNamespace(text="ID: 123")],
            )

    with pytest.raises(MCPToolCallError, match="không được cấp quyền"):
        asyncio.run(call_mcp_tool(
            Session(), "search_dichvucong", {"keyword": "hộ chiếu"},
            allowed_tools=set(), timeout_seconds=1,
        ))

    result = asyncio.run(call_mcp_tool(
        Session(), "search_dichvucong", {"keyword": "hộ chiếu"},
        allowed_tools={"search_dichvucong"}, timeout_seconds=1,
    ))
    assert result == "Tên thủ tục: Cấp hộ chiếu\nID: 123"


def test_call_contract_timeout_and_mcp_error_result():
    import asyncio
    from types import SimpleNamespace
    from backend.mcp.contract import MCPToolCallError, call_mcp_tool

    class SlowSession:
        async def call_tool(self, name, arguments):
            await __import__("asyncio").sleep(0.1)
            return SimpleNamespace(isError=False, content=[])

    with pytest.raises(MCPToolCallError, match="quá thời gian"):
        asyncio.run(call_mcp_tool(
            SlowSession(), "search_procedure", {"keyword": "hộ chiếu"},
            allowed_tools={"search_procedure"}, timeout_seconds=0.01,
        ))

    class ErrorSession:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text="upstream error")])

    with pytest.raises(MCPToolCallError, match="upstream error"):
        asyncio.run(call_mcp_tool(
            ErrorSession(), "search_procedure", {"keyword": "hộ chiếu"},
            allowed_tools={"search_procedure"}, timeout_seconds=1,
        ))
