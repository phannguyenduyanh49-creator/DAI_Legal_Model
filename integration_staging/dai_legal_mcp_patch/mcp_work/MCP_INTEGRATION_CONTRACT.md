# DAI Legal MCP — integration contract

## Scope

This patch changes only `mcp_server.py` and adds `backend/mcp/runtime.py`,
`backend/mcp/contract.py`, `backend/mcp/__init__.py`, and isolated tests under
`tests/mcp/`. `backend/mcp/adapter.py` is copied unchanged. No core gateway,
main API, domain registry, prompt, Evidence schema, or verifier files are changed.

## Registered tools

| Tool | Required arguments | Optional arguments | Return |
|---|---|---|---|
| `search_legal_documents` | `query: str` | `limit: int = 5` | text |
| `search_procedure` | `keyword: str` | `limit: int = 5` | text |
| `search_dichvucong` | `keyword: str` | none | text |

The current server contract remains text, so existing `backend/mcp/adapter.py` can
continue parsing the DVC tool result. Non-text MCP content blocks are joined with
newlines by the caller helper.

## Caller function for later integration

The orchestration layer should own and reuse an initialized MCP `ClientSession`,
then call:

```python
from backend.mcp.contract import call_mcp_tool

result_text = await call_mcp_tool(
    session=mcp_session,
    tool_name="search_dichvucong",
    arguments={"keyword": user_query},
    allowed_tools=domain_config.policy.allowed_mcp_tools,
    timeout_seconds=60,
)
```

`allowed_tools` must come from the effective domain policy; do not pass an
unrestricted list from user input. `call_mcp_tool` rejects unknown or disallowed
tools before calling the session, enforces a caller-side timeout, and raises
`MCPToolCallError` for timeout or MCP tool error results. It returns the text
content on success. Catch `MCPToolCallError` at the integration boundary and map it
to the project's existing error/evidence handling without changing Evidence schema.

## Server-side policy

Set `MCP_ALLOWED_TOOLS` in the MCP server process to a comma-separated allowlist,
for example:

```text
MCP_ALLOWED_TOOLS=search_dichvucong,search_procedure
MCP_TOOL_TIMEOUT_SECONDS=60
```

If `MCP_ALLOWED_TOOLS` is absent, the three existing registered tools are allowed
for backward compatibility. If it is set to an empty string, all tools are denied.
The caller-side allowlist and server-side allowlist are independent checks.

## Timeout behavior and limitation

The server wrapper returns a stable Vietnamese text error for denied tools,
timeouts, and tool exceptions. The caller helper raises `MCPToolCallError` on a
client-side timeout. Python cannot forcibly terminate a synchronous function that
is already running in a worker thread; a timed-out call may continue in the
background. The existing DVC HTTP calls retain their own 30-second I/O timeout.
If hard cancellation is required, isolate tools in separate processes or use
cancellable asynchronous I/O.

## Tests

Run from the repository root:

```bash
python -m pytest -q tests/mcp
```

Tests cover allowlist parsing and denial, no execution when denied, timeout,
exception propagation, unchanged successful results, stable server wrapper errors,
caller permission checks, normalized text results, and MCP error results. They do
not initialize Qdrant, Transformers, DVC network calls, or the actual MCP server.
