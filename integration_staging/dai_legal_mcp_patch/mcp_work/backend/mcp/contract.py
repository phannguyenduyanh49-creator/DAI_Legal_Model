"""Stable caller-side contract for invoking the persistent MCP client session.

The integration layer (for example the orchestrator/gateway) should own the MCP
ClientSession lifecycle and pass that already-initialized session to
``call_mcp_tool``. This avoids starting a new model/Qdrant-backed server per call.
"""
from __future__ import annotations

import asyncio
from typing import Any, Iterable, Protocol

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
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments),
            timeout=float(timeout_seconds),
        )
    except asyncio.TimeoutError as exc:
        raise MCPToolCallError(
            f"Gọi MCP tool '{tool_name}' quá thời gian {timeout_seconds:g}s."
        ) from exc
    except Exception as exc:
        raise MCPToolCallError(
            f"Gọi MCP tool '{tool_name}' thất bại: {type(exc).__name__}: {exc}"
        ) from exc

    is_error = getattr(result, "isError", None)
    if is_error is None and isinstance(result, dict):
        is_error = result.get("isError", False)
    text = _content_to_text(result)
    if is_error:
        raise MCPToolCallError(text or f"MCP tool '{tool_name}' trả về lỗi.")
    return text
