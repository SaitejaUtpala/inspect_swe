"""Clean Inspect-owned tool fault via `BridgedToolsSpec`.

Run:

    inspect eval slice_tests/bridged_tool_fault_slice.py@bridged_tool_fault_slice \
      --model openai/gpt-5.5-2026-04-23 \
      --max-samples 1

This is the clean surface: Inspect owns the Python tool function and exposes it
to Codex via MCP. The tool itself raises `ToolError`, so the fault is injected at
an Inspect-controlled boundary rather than by mutating Responses payloads.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.agent import BridgedToolsSpec, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.tool import Tool, ToolError, tool
from inspect_swe._codex_cli.codex_cli import codex_cli


@tool
def flaky_lookup() -> Tool:
    """Look up a short fact, but fail for slice testing."""

    async def execute(query: str) -> str:
        """Look up a short fact.

        Args:
            query: Lookup query.
        """
        raise ToolError(f"Injected bridged-tool failure for query: {query}")

    return execute


@task
def bridged_tool_fault_slice() -> Task:
    """One-sample task for a clean Inspect-owned tool fault."""
    return Task(
        dataset=[
            Sample(
                id="bridged_tool_fault_slice",
                input=(
                    "Use the MCP tool named flaky_lookup once with query '2+2'. "
                    "If it fails, recover and answer the question yourself: what is 2+2?"
                ),
            )
        ],
        solver=as_solver(
            codex_cli(
                version="0.118.0",
                bridged_tools=[
                    BridgedToolsSpec(name="slice_tools", tools=[flaky_lookup()])
                ],
            )
        ),
        sandbox="docker",
    )

