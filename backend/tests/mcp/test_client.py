from __future__ import annotations

import asyncio
import ast
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.mcp import client
from backend.mcp.contract import MCP_TOOL_CONTRACT, MCPToolCallError


VALID_DVC = """DVC_OFFICIAL_PROCEDURE
DVC_PROCEDURE_NAME: Cấp Phiếu lý lịch tư pháp
DVC_PROCEDURE_CODE: 2.000488
DVC_PROCEDURE_ID: 102076
DVC_SOURCE_URL: https://dichvucong.gov.vn/p/home/dvc-tthc-thu-tuc-hanh-chinh.html
DVC_DETAIL_BEGIN
Hồ sơ và thời hạn giải quyết.
DVC_DETAIL_END"""


def run_async(test_function):
    @wraps(test_function)
    def wrapper(*args, **kwargs):
        return asyncio.run(test_function(*args, **kwargs))
    return wrapper


def test_parameters_pin_executable_cwd_policy_and_caller_server_deadline(tmp_path):
    params = client.make_server_parameters(
        script_path=tmp_path / "mcp_server.py",
        cwd=tmp_path,
        python_executable="python-test",
        allowed_tools={"search_dichvucong"},
        request_id="trace-1",
        timeout_seconds=12,
        base_environment={
            "MCP_ALLOWED_TOOLS": "search_procedure,search_dichvucong",
            "MCP_TOOL_TIMEOUT_SECONDS": "30",
            "KEEP": "yes",
        },
    )
    assert params.command == "python-test"
    assert params.args == [str((tmp_path / "mcp_server.py").resolve())]
    assert params.cwd == str(tmp_path.resolve())
    assert params.env["MCP_ALLOWED_TOOLS"] == "search_dichvucong"
    assert params.env["MCP_TOOL_TIMEOUT_SECONDS"] == "12.0"
    assert params.env["DAI_REQUEST_ID"] == "trace-1"
    assert params.env["KEEP"] == "yes"


def test_registered_server_tool_names_and_arguments_match_caller_contract():
    source_path = Path(__file__).resolve().parents[3] / "mcp_server.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    server_tools = {}
    for node in module.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorator_names = {
            decorator.func.id if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            else decorator.func.attr if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            else decorator.id if isinstance(decorator, ast.Name)
            else decorator.attr if isinstance(decorator, ast.Attribute)
            else None
            for decorator in node.decorator_list
        }
        if "tool" in decorator_names and "guarded_tool" in decorator_names:
            server_tools[node.name] = [arg.arg for arg in node.args.args]

    assert set(server_tools) == set(MCP_TOOL_CONTRACT)
    for name, contract in MCP_TOOL_CONTRACT.items():
        expected_arguments = list(contract["arguments"])
        assert server_tools[name] == expected_arguments


@run_async
async def test_stdio_call_matches_contract_and_denies_before_start(monkeypatch):
    starts = []

    @asynccontextmanager
    async def fake_stdio(_params):
        starts.append(True)
        yield "read", "write"

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, name, arguments):
            assert name == "search_dichvucong"
            assert arguments == {"keyword": "lý lịch tư pháp"}
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text=VALID_DVC)])

    monkeypatch.setattr(client, "stdio_client", fake_stdio)
    monkeypatch.setattr(client, "ClientSession", lambda *_args: Session())
    params = client.make_server_parameters(
        script_path="mcp_server.py", cwd=".", python_executable="python",
        allowed_tools={"search_dichvucong"}, request_id="t", timeout_seconds=2,
        base_environment={},
    )
    with pytest.raises(MCPToolCallError) as denied:
        await client.call_stdio_tool(
            params, "search_dichvucong", {"keyword": "lý lịch tư pháp"},
            allowed_tools=set(), timeout_seconds=1,
        )
    assert denied.value.reason_code == "MCP_POLICY_DENIED"
    assert not starts

    result = await client.call_stdio_tool(
        params, "search_dichvucong", {"keyword": "lý lịch tư pháp"},
        allowed_tools={"search_dichvucong"}, timeout_seconds=1,
    )
    assert result == VALID_DVC
    assert len(starts) == 1


@run_async
async def test_startup_initialize_timeout_and_process_death_are_classified_and_restarted(monkeypatch):
    behavior = {"start_error": False, "init_error": False, "init_slow": False, "call_error": False}
    starts = []

    @asynccontextmanager
    async def fake_stdio(_params):
        starts.append(True)
        if behavior["start_error"]:
            raise OSError("process did not start")
        yield "read", "write"

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def initialize(self):
            if behavior["init_error"]:
                raise RuntimeError("initialize failed")
            if behavior["init_slow"]:
                import asyncio
                await asyncio.sleep(0.05)

        async def call_tool(self, _name, _arguments):
            if behavior["call_error"]:
                raise BrokenPipeError("server exited")
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text=VALID_DVC)])

    monkeypatch.setattr(client, "stdio_client", fake_stdio)
    monkeypatch.setattr(client, "ClientSession", lambda *_args: Session())
    params = client.make_server_parameters(
        script_path="mcp_server.py", cwd=".", python_executable="python",
        allowed_tools={"search_dichvucong"}, request_id="t", timeout_seconds=1,
        base_environment={},
    )
    args = (params, "search_dichvucong", {"keyword": "x"})
    kwargs = {"allowed_tools": {"search_dichvucong"}, "timeout_seconds": 1}

    behavior["start_error"] = True
    with pytest.raises(MCPToolCallError) as startup:
        await client.call_stdio_tool(*args, **kwargs)
    assert startup.value.reason_code == "MCP_SERVER_STARTUP_FAILED"

    behavior.update(start_error=False, init_error=True)
    with pytest.raises(MCPToolCallError) as initialize:
        await client.call_stdio_tool(*args, **kwargs)
    assert initialize.value.reason_code == "MCP_INITIALIZATION_FAILED"

    behavior.update(init_error=False, init_slow=True)
    with pytest.raises(MCPToolCallError) as init_timeout:
        await client.call_stdio_tool(
            params, "search_dichvucong", {"keyword": "x"},
            allowed_tools={"search_dichvucong"}, timeout_seconds=0.01,
        )
    assert init_timeout.value.reason_code == "MCP_INITIALIZATION_TIMEOUT"

    behavior.update(init_slow=False, call_error=True)
    with pytest.raises(MCPToolCallError) as died:
        await client.call_stdio_tool(*args, **kwargs)
    assert died.value.reason_code == "MCP_UPSTREAM_ERROR"
    behavior["call_error"] = False
    assert await client.call_stdio_tool(*args, **kwargs) == VALID_DVC
    assert len(starts) == 5  # each retry/reconnect attempt owns a fresh child process


@run_async
async def test_concurrent_calls_own_distinct_sessions(monkeypatch):
    import asyncio

    next_id = 0
    active = 0
    max_active = 0
    seen = []

    @asynccontextmanager
    async def fake_stdio(_params):
        nonlocal next_id
        next_id += 1
        yield next_id, object()

    class Session:
        def __init__(self, session_id):
            self.session_id = session_id

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, _name, arguments):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            seen.append((self.session_id, arguments["keyword"]))
            await asyncio.sleep(0.01)
            active -= 1
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text=VALID_DVC)])

    monkeypatch.setattr(client, "stdio_client", fake_stdio)
    monkeypatch.setattr(client, "ClientSession", lambda read, _write: Session(read))
    params = client.make_server_parameters(
        script_path="mcp_server.py", cwd=".", python_executable="python",
        allowed_tools={"search_dichvucong"}, request_id="t", timeout_seconds=2,
        base_environment={},
    )
    async def invoke(keyword):
        return await client.call_stdio_tool(
            params, "search_dichvucong", {"keyword": keyword},
            allowed_tools={"search_dichvucong"}, timeout_seconds=1,
        )

    assert await asyncio.gather(invoke("a"), invoke("b")) == [VALID_DVC, VALID_DVC]
    assert len({session_id for session_id, _ in seen}) == 2
    assert {keyword for _, keyword in seen} == {"a", "b"}
    assert max_active == 2
