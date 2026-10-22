"""Shared policy and bounded execution helpers for DAI Legal MCP tools.

This module deliberately has no dependency on Qdrant, Transformers, DVC, or the
MCP SDK so its security and timeout behavior can be tested in isolation.
"""
from __future__ import annotations

import functools
import os
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0
DEFAULT_TOOL_NAMES = frozenset({
    "search_legal_documents",
    "search_procedure",
    "search_dichvucong",
})

# Shared pool avoids waiting for a timed-out call during context-manager shutdown.
# A timed-out synchronous function may continue in its worker thread; see notes in
# the integration contract before using this for untrusted/non-cooperative code.
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dai-mcp-tool")


class ToolNotAllowedError(PermissionError):
    """Raised when a tool is not present in the effective allowlist."""


class ToolTimeoutError(TimeoutError):
    """Raised when a tool does not return within its configured deadline."""


def parse_allowed_tools(value: str | None) -> frozenset[str]:
    """Parse comma-separated tool names; None means backward-compatible defaults.

    An explicitly empty string denies every tool. Tool names are exact and
    case-sensitive to avoid accidentally broadening permissions.
    """
    if value is None:
        return DEFAULT_TOOL_NAMES
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def get_allowed_tools() -> frozenset[str]:
    """Read the allowlist at call time so environment/config changes take effect."""
    return parse_allowed_tools(os.getenv("MCP_ALLOWED_TOOLS"))


def is_tool_allowed(tool_name: str, allowed_tools: Iterable[str] | None = None) -> bool:
    """Return whether a named tool is permitted by the supplied/effective policy."""
    allowed = get_allowed_tools() if allowed_tools is None else frozenset(allowed_tools)
    return tool_name in allowed


def get_tool_timeout_seconds() -> float:
    """Return a validated timeout from MCP_TOOL_TIMEOUT_SECONDS (default 60s)."""
    raw = os.getenv("MCP_TOOL_TIMEOUT_SECONDS", str(DEFAULT_TOOL_TIMEOUT_SECONDS))
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("MCP_TOOL_TIMEOUT_SECONDS phải là số dương.") from exc
    if timeout <= 0:
        raise ValueError("MCP_TOOL_TIMEOUT_SECONDS phải là số dương.")
    return timeout


def execute_tool(
    tool_name: str,
    func: Callable[..., T],
    *args: Any,
    allowed_tools: Iterable[str] | None = None,
    timeout_seconds: float | None = None,
    **kwargs: Any,
) -> T:
    """Authorize and execute a synchronous tool with a hard caller-side deadline.

    The allowlist is checked before submitting any work. Timeout stops waiting for
    the result but cannot forcibly kill a running Python thread; network operations
    should therefore also use their own I/O timeouts.
    """
    if not is_tool_allowed(tool_name, allowed_tools):
        raise ToolNotAllowedError(f"Tool '{tool_name}' không được cấp quyền.")

    timeout = get_tool_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
    if timeout <= 0:
        raise ValueError("timeout_seconds phải là số dương.")

    future: Future[T] = _EXECUTOR.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()  # Cancels only if the worker has not started.
        raise ToolTimeoutError(
            f"Tool '{tool_name}' vượt quá timeout {timeout:g}s."
        ) from exc


def guarded_tool(tool_name: str) -> Callable[[Callable[..., Any]], Callable[..., str]]:
    """Decorator for MCP tool functions, preserving their discoverable signature.

    MCP's current tool return contract in this project is text. Errors are returned
    as stable Vietnamese text rather than crashing the stdio server. The detailed
    exception type is retained, but stack traces and secrets are not returned.
    """
    def decorate(func: Callable[..., Any]) -> Callable[..., str]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> str:
            try:
                result = execute_tool(tool_name, func, *args, **kwargs)
                if isinstance(result, str):
                    return result
                if result is None:
                    return ""
                return str(result)
            except ToolNotAllowedError as exc:
                return f"Lỗi phân quyền MCP: {exc}"
            except ToolTimeoutError as exc:
                return f"Lỗi timeout MCP: {exc}"
            except Exception as exc:
                return f"Lỗi MCP tool '{tool_name}': {type(exc).__name__}: {exc}"
        return wrapped
    return decorate
