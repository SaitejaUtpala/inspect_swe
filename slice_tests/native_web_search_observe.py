"""Observe native Codex `web_search` without perturbing it.

Run:

    inspect eval slice_tests/native_web_search_observe.py@native_web_search_observe \
      --model openai/gpt-5.5-2026-04-23 \
      --max-samples 1

This slice is intentionally observation-only. It prints the tool schemas visible
to the model so the Responses/server-side `web_search` path can be inspected in
the resulting `.eval` log without adding reliability-layer mutations.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.agent import as_solver
from inspect_ai.dataset import Sample
from inspect_swe._codex_cli.codex_cli import codex_cli


async def observe_tools_filter(model, messages, tools, tool_choice, config):
    """Print visible tools and allow normal native tool behavior."""
    del model, messages, tool_choice, config
    print("=== native web_search observe slice ===")
    print("tool_names:", [tool.name for tool in tools])
    for tool_info in tools:
        if tool_info.name == "web_search":
            print("web_search description:", tool_info.description)
            print("web_search options:", tool_info.options)
    return None


@task
def native_web_search_observe() -> Task:
    """One-sample task for observing Codex native web search."""
    return Task(
        dataset=[
            Sample(
                id="native_web_search_observe",
                input=(
                    "Use web search once, then answer briefly: "
                    "what is the title of the current OpenAI homepage?"
                ),
            )
        ],
        solver=as_solver(
            codex_cli(
                version="0.118.0",
                filter=observe_tools_filter,
                retry_refusals=1,
            )
        ),
        sandbox="docker",
    )

