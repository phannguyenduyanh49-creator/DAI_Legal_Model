"""Stable caller-side tool contract for an initialized MCP session.

The caller owns session/process lifecycle. It may reuse a session only when its
transport supports safe concurrency; the current stdio integration creates an
isolated process and initialized session for each request.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Iterable, Protocol

MCP_TOOL_CONTRACT: dict[str, dict[str, Any]] = {
    "search_legal_documents": {
        "arguments": {"query": "str", "limit": "int (optional, default 5)"},
        "description": "Tìm kiếm tài liệu pháp luật trong Qdrant local.",
        "result": "text",
    },
    "search_procedure": {
        "arguments": {"keyword": "str", "limit": "int (optional, default 5)"},
        "description": "Tìm kiếm thủ tục trong kho pháp luật local.",
        "result": "text",
    },
    "search_dichvucong": {
        "arguments": {"keyword": "str"},
        "description": "Tra cứu thủ tục từ Cổng Dịch vụ công Quốc gia.",
        "result": "text",
    },
}


class MCPToolCallError(RuntimeError):
    """Raised for caller-side permission, timeout, or MCP tool errors."""

    def __init__(self, message: str, reason_code: str = "MCP_TOOL_ERROR") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class SupportsCallTool(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


def _content_to_text(result: Any) -> str:
    """Convert standard MCP text content blocks to one text result."""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if not content:
        return ""

    chunks: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text")
        if text is not None:
            chunks.append(str(text))
    return "\n".join(chunks).strip()


async def call_mcp_tool(
    session: SupportsCallTool,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    allowed_tools: Iterable[str],
    timeout_seconds: float = 60.0,
    on_stage: Callable[[str, float], None] | None = None,
) -> str:
    """Call an initialized MCP session with explicit caller-side authorization.

    Args:
        session: Initialized MCP ``ClientSession`` (lifecycle managed elsewhere).
        tool_name: One of the registered tool names in ``MCP_TOOL_CONTRACT``.
        arguments: Tool arguments, following the contract below.
        allowed_tools: Effective permissions supplied by the calling domain policy.
        timeout_seconds: Caller-side deadline for the remote MCP request.

    Returns:
        Joined text content from the MCP result. This intentionally preserves the
        current server's string-return contract so ``adapter.py`` can be reused.

    Raises:
        MCPToolCallError: Unknown/unauthorized tool, timeout, MCP error result, or
            an exception raised by the session.
    """
    if tool_name not in MCP_TOOL_CONTRACT:
        raise MCPToolCallError(f"Tool MCP không tồn tại: {tool_name}")
    if tool_name not in frozenset(allowed_tools):
        raise MCPToolCallError(f"Tool MCP không được cấp quyền: {tool_name}")
    if not isinstance(arguments, dict):
        raise MCPToolCallError("arguments phải là dict.")
    if timeout_seconds <= 0:
        raise MCPToolCallError("timeout_seconds phải là số dương.")

    try:
        tool_started_at = time.perf_counter()
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=float(timeout_seconds),
        )
        if on_stage is not None:
            on_stage("tool_call", (time.perf_counter() - tool_started_at) * 1000)
    except asyncio.TimeoutError as exc:
        if on_stage is not None:
            on_stage("tool_call_timeout", float(timeout_seconds) * 1000)
        error = MCPToolCallError(
            f"G\u1ecdi MCP tool '{tool_name}' qu\u00e1 th\u1eddi gian {timeout_seconds:g}s."
        )
        error.reason_code = "MCP_TOOL_TIMEOUT"
        raise error from exc
    except Exception as exc:
        if on_stage is not None:
            on_stage("tool_call_error", (time.perf_counter() - tool_started_at) * 1000)
        error = MCPToolCallError(
            f"G\u1ecdi MCP tool '{tool_name}' th\u1ea5t b\u1ea1i: {type(exc).__name__}: {exc}"
        )
        error.reason_code = "MCP_UPSTREAM_ERROR"
        raise error from exc

    is_error = getattr(result, "isError", None)
    if is_error is None and isinstance(result, dict):
        is_error = result.get("isError", False)
    text = _content_to_text(result)
    server_error_prefixes = (
        "Lỗi phân quyền MCP:",
        "Lỗi timeout MCP:",
        "Lỗi MCP tool '",
    )
    if is_error or text.startswith(server_error_prefixes):
        if not is_error and text.startswith(server_error_prefixes[1]):
            code = "MCP_TOOL_TIMEOUT"
        elif not is_error and text.startswith(server_error_prefixes[0]):
            code = "MCP_POLICY_DENIED"
        else:
            code = "MCP_UPSTREAM_ERROR"
        raise MCPToolCallError(text or "MCP tool returned an error.", code)
    return text
