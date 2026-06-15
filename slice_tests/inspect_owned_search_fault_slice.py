"""Recommended slice: fault an Inspect-owned search tool.

Run:

    inspect eval \
      slice_tests/inspect_owned_search_fault_slice.py@inspect_owned_search_fault_slice \
      --model openai/gpt-5.4-2026-03-05 \
      --max-samples 1

This is an isolated version of the current direction for model-native/server-side
tools:

1. Disable native `web_search`.
2. Define a tiny Inspect-owned search tool in this file.
3. Fault that local tool wrapper directly.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.agent import BridgedToolsSpec, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.tool import Tool, tool
from inspect_swe._codex_cli.codex_cli import codex_cli


@tool
def reliability_search() -> Tool:
    """Search through an Inspect-owned wrapper, but return a failed result."""

    async def execute(query: str) -> str:
        """Search for information related to a query.

        Args:
            query: Search query.
        """
        return (
            "web_search failed. The search wrapper returned no results due to a "
            f"transient tool failure. Query: {query}"
        )

    return execute


@task
def inspect_owned_search_fault_slice() -> Task:
    """One-sample task for the Inspect-owned search replacement path."""
    return Task(
        dataset=[
            Sample(
                id="inspect_owned_search_fault_slice",
                input=(
                    "Native web_search is disabled. Use the MCP tool named "
                    "reliability_search once to search for 'OpenAI homepage title'. "
                    "If the tool returns no useful output, recover and say what you can."
                ),
            )
        ],
        solver=as_solver(
            codex_cli(
                version="0.118.0",
                disallowed_tools=["web_search"],
                bridged_tools=[
                    BridgedToolsSpec(
                        name="reliability_search",
                        tools=[reliability_search()],
                    )
                ],
            )
        ),
        sandbox="docker",
    )

