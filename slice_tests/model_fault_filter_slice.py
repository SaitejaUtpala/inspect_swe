"""Retryable model/API-style fault via `GenerateFilter`.

Run:

    inspect eval slice_tests/model_fault_filter_slice.py@model_fault_filter_slice \
      --model openai/gpt-5.5-2026-04-23 \
      --max-samples 1

The first bridge generation returns a `content_filter` stop reason. The Codex
wrapper is configured with `retry_refusals=1`, so this tests whether the bridge
retry path continues with a clean second generation.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.agent import as_solver
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_swe._codex_cli.codex_cli import codex_cli

_faulted_once = False


async def retryable_model_fault_filter(model, messages, tools, tool_choice, config):
    """Inject one retryable model fault, then allow normal generation."""
    del messages, tools, tool_choice, config
    global _faulted_once
    if not _faulted_once:
        _faulted_once = True
        print("=== injecting retryable model fault ===")
        return ModelOutput.from_content(
            model.name,
            "Injected transient model API error.",
            stop_reason="content_filter",
            error="Injected transient model API error.",
        )
    print("=== allowing normal generation after retry ===")
    return None


@task
def model_fault_filter_slice() -> Task:
    """One-sample task for testing retryable bridge filter behavior."""
    return Task(
        dataset=[
            Sample(
                id="model_fault_filter_slice",
                input="Answer in one short sentence: what is 3 + 5?",
            )
        ],
        solver=as_solver(
            codex_cli(
                version="0.118.0",
                filter=retryable_model_fault_filter,
                retry_refusals=1,
            )
        ),
        sandbox="docker",
    )

