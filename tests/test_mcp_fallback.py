"""Focused regression tests for DAI Local RAG -> MCP fallback routing.

Place at tests/test_mcp_fallback.py and run from repository root:
    python -m pytest tests/test_mcp_fallback.py -v

The tests avoid importing backend.main because module import initializes Qdrant
and GPU models. They inspect source structure and exercise the real pure policy.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = REPO_ROOT / "backend" / "main.py"


@pytest.fixture(scope="module")
def main_tree() -> ast.Module:
    assert MAIN_FILE.is_file(), f"Cannot find backend entrypoint: {MAIN_FILE}"
    return ast.parse(MAIN_FILE.read_text(encoding="utf-8"), filename=str(MAIN_FILE))


def _function(parent: ast.AST, name: str):
    for node in ast.walk(parent):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} was not found")


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                names.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                names.add(child.func.attr)
    return names


def _source(node: ast.AST) -> str:
    return ast.unparse(node)


def test_backend_main_has_valid_python_syntax(main_tree: ast.Module) -> None:
    assert isinstance(main_tree, ast.Module)


def test_local_rag_evaluates_sufficiency_and_returns_fallback_state(
    main_tree: ast.Module,
) -> None:
    search_fn = _function(main_tree, "search_local_rag")
    source = _source(search_fn)
    assert "check_enough_data" in _called_names(search_fn)
    # ast.unparse normalizes string quotes to single quotes.
    assert "'enough_data'" in source
    assert "'external_fallback'" in source


def test_chat_route_uses_source_selection_policy(main_tree: ast.Module) -> None:
    chat_fn = _function(main_tree, "chat_completions")
    generate_fn = _function(chat_fn, "generate")
    assert "choose_data_source" in _called_names(generate_fn)


def test_chat_route_calls_mcp_and_rechecks_returned_evidence(
    main_tree: ast.Module,
) -> None:
    chat_fn = _function(main_tree, "chat_completions")
    generate_fn = _function(chat_fn, "generate")
    calls = _called_names(generate_fn)
    assert "search_dichvucong" in calls
    assert "check_evidence_sufficiency" in calls


def test_mcp_evidence_is_passed_into_evidence_builder(
    main_tree: ast.Module,
) -> None:
    chat_fn = _function(main_tree, "chat_completions")
    generate_fn = _function(chat_fn, "generate")
    source = _source(generate_fn)
    assert "mcp_evidences=mcp_evidences" in source
    assert "evidence_builder.build" in source


def test_mcp_call_is_gated_by_effective_tool_allowlist(
    main_tree: ast.Module,
) -> None:
    chat_fn = _function(main_tree, "chat_completions")
    generate_fn = _function(chat_fn, "generate")
    generate_source = _source(generate_fn)
    chat_source = _source(chat_fn)

    # The nested generator derives the permission flag from the policy-filtered list.
    assert "'search_dichvucong' in effective_mcp_tools" in generate_source
    # The enclosing route obtains that list through policy middleware.
    assert "policy_middleware.constrain_tools" in chat_source
    assert "effective_mcp_tools" in chat_source
    assert "data_source == 'blocked'" in generate_source
    assert "data_source == 'permission'" in generate_source


@pytest.mark.parametrize(
    ("has_local_evidence", "enough_data", "allow_external", "mcp_tool_allowed", "expected"),
    [
        (True, True, False, True, "local"),
        (True, False, False, True, "local"),
        (False, False, False, True, "permission"),
        (False, False, True, True, "external"),
        (False, False, True, False, "blocked"),
        (False, None, False, False, "blocked"),
    ],
)
def test_choose_data_source_policy(
    has_local_evidence: bool,
    enough_data: bool | None,
    allow_external: bool,
    mcp_tool_allowed: bool,
    expected: str,
) -> None:
    policy = importlib.import_module("backend.rag.pipeline_policy")
    assert policy.choose_data_source(
        has_local_evidence=has_local_evidence,
        enough_data=enough_data,
        allow_external=allow_external,
        mcp_tool_allowed=mcp_tool_allowed,
    ) == expected
