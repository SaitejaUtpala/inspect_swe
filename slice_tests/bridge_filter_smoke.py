"""Minimal `codex_cli` bridge filter smoke test.

Run:

    inspect eval slice_tests/bridge_filter_smoke.py@bridge_filter_smoke \
      --model openai/gpt-5.5-2026-04-23 \
      --max-samples 1

This proves that the `sandbox_agent_bridge(filter=...)` surface sees the
messages and model-visible tools before generation.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.agent import as_solver
from inspect_ai.dataset import Sample
from inspect_swe._codex_cli.codex_cli import codex_cli


async def print_bridge_filter(model, messages, tools, tool_choice, config):
    """Print bridge generation inputs and let normal generation continue."""
    del config
    print("=== bridge filter ===")
    print("model:", model.name)
    print("message_count:", len(messages))
    print("tool_names:", [tool.name for tool in tools])
    print("tool_choice:", tool_choice)
    return None


@task
def bridge_filter_smoke() -> Task:
    """One-sample task that exercises the real Codex bridge."""
    return Task(
        dataset=[
            Sample(
                id="bridge_filter_smoke",
                input=(
                    "Answer in one short sentence: what is 2 + 2? "
                    "Do not inspect files."
                ),
            )
        ],
        solver=as_solver(
            codex_cli(
                version="0.118.0",
                filter=print_bridge_filter,
                retry_refusals=1,
            )
        ),
        sandbox="docker",
    )

