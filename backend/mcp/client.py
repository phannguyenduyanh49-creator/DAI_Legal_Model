"""Per-request MCP stdio process lifecycle and caller-side authorization."""
from __future__ import annotations

import asyncio
import math
import os
import time
from pathlib import Path
from typing import Callable, Iterable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from backend.mcp.contract import MCP_TOOL_CONTRACT, MCPToolCallError, call_mcp_tool
from backend.mcp.runtime import scoped_tool_environment


def make_server_parameters(
    *,
    script_path: str | Path,
    cwd: str | Path,
    python_executable: str,
    allowed_tools: Iterable[str],
    request_id: str,
    timeout_seconds: float,
    base_environment: dict[str, str] | None = None,
) -> StdioServerParameters:
    """Build an explicitly scoped child environment and executable command."""
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    base = dict(os.environ if base_environment is None else base_environment)
    environment = scoped_tool_environment(allowed_tools, base)
    configured_server_timeout = base.get("MCP_TOOL_TIMEOUT_SECONDS")
    try:
        server_timeout = float(configured_server_timeout or timeout)
    except (TypeError, ValueError):
        server_timeout = timeout
    if not math.isfinite(server_timeout) or server_timeout <= 0:
        server_timeout = timeout
    environment["MCP_TOOL_TIMEOUT_SECONDS"] = str(min(server_timeout, timeout))
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["DAI_REQUEST_ID"] = request_id
    return StdioServerParameters(
        command=python_executable,
        args=[str(Path(script_path).resolve())],
        cwd=str(Path(cwd).resolve()),
        env=environment,
        encoding="utf-8",
        encoding_error_handler="replace",
    )


async def call_stdio_tool(
    server_params: StdioServerParameters,
    tool_name: str,
    arguments: dict,
    *,
    allowed_tools: Iterable[str],
    timeout_seconds: float,
    on_stage: Callable[[str, float], None] | None = None,
) -> str:
    """Start, initialize, call, and close one isolated stdio session.

    Every invocation owns its process and ``ClientSession``. A dead process is
    discarded by the async context managers; the next invocation starts fresh.
    """
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if tool_name not in MCP_TOOL_CONTRACT:
        raise MCPToolCallError(f"Unknown MCP tool: {tool_name}", "MCP_UNKNOWN_TOOL")
    if tool_name not in frozenset(allowed_tools):
        raise MCPToolCallError(
            f"MCP tool is denied by the effective domain policy: {tool_name}",
            "MCP_POLICY_DENIED",
        )

    phase = "process_start"
    started_at = time.perf_counter()

    def report(name: str, began: float) -> None:
        if on_stage is not None:
            on_stage(name, (time.perf_counter() - began) * 1000)

    async def run_session() -> str:
        nonlocal phase
        process_started = time.perf_counter()
        async with stdio_client(server_params) as (read, write):
            report("process_start_and_stdio", process_started)
            phase = "initialize"
            initialize_started = time.perf_counter()
            async with ClientSession(read, write) as session:
                await session.initialize()
                report("initialize", initialize_started)
                phase = "tool_call"
                return await call_mcp_tool(
                    session,
                    tool_name,
                    arguments,
                    allowed_tools=allowed_tools,
                    timeout_seconds=timeout,
                    on_stage=on_stage,
                )

    try:
        return await asyncio.wait_for(run_session(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        report(f"{phase}_timeout", started_at)
        reason = (
            "MCP_SERVER_STARTUP_TIMEOUT"
            if phase == "process_start"
            else "MCP_INITIALIZATION_TIMEOUT"
            if phase == "initialize"
            else "MCP_TOOL_TIMEOUT"
        )
        raise MCPToolCallError(
            f"MCP {phase} exceeded its {timeout:g}s deadline.", reason
        ) from exc
    except MCPToolCallError:
        raise
    except Exception as exc:
        report(f"{phase}_error", started_at)
        reason = (
            "MCP_SERVER_STARTUP_FAILED"
            if phase == "process_start"
            else "MCP_INITIALIZATION_FAILED"
            if phase == "initialize"
            else "MCP_UPSTREAM_ERROR"
        )
        raise MCPToolCallError(
            f"MCP {phase} failed: {type(exc).__name__}", reason
        ) from exc
