"""Slice: patch the Responses bridge service and mark web_search_call failed.

Run:

    inspect eval \
      slice_tests/native_web_search_proxy_patch_slice.py@native_web_search_proxy_patch_slice \
      --model openai/gpt-5.4-2026-03-05 \
      --max-samples 1

This reproduces the proxy-level approach that was tried: wrap
`inspect_ai.agent._bridge.sandbox.service.generate_responses` and mutate the raw
Responses dict before it is streamed/replayed to Codex.

Observed behavior: this can still produce a normal final answer. In one run,
Codex answered the OpenAI homepage-title question even though this patch was
installed. That tells us changing only the raw `web_search_call.status` field is
not enough to guarantee a recoverable failed-tool observation.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import as_solver
from inspect_ai.dataset import Sample
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_swe._codex_cli.codex_cli import codex_cli

WEB_SEARCH_FAILED_RESULT = (
    "web_search failed. The search service returned no results due to a "
    "transient error. Please retry or continue with another approach."
)


@contextmanager
def response_web_search_status_patch() -> Iterator[None]:
    """Mark raw Responses `web_search_call` output items as failed."""
    from inspect_ai.agent._bridge.sandbox import service

    # Save the original bridge function so the monkeypatch is scoped to one
    # solver run and does not leak into other evals.
    original_generate_responses = service.generate_responses

    def generate_responses_with_faults(web_search: Any, code_execution: Any, bridge: Any):
        # This is the low-level Responses proxy function used by the sandbox
        # bridge. It is closer to the real Codex/native web_search path than the
        # GenerateFilter, but now we are working with raw protocol dictionaries.
        original_generate = original_generate_responses(
            web_search, code_execution, bridge
        )

        async def generate(json_data: dict[str, Any]) -> dict[str, Any]:
            response = await original_generate(json_data)
            for item in response.get("output", []):
                if isinstance(item, dict) and item.get("type") == "web_search_call":
                    # This is intentionally schema-sensitive. We are guessing
                    # which raw fields carry the search result because this is
                    # not a typed web_search-result hook.
                    _print_web_search_item("before", item)
                    _mark_web_search_failed(item)
                    _print_web_search_item("after", item)
            return response

        return generate

    service.generate_responses = generate_responses_with_faults
    try:
        yield
    finally:
        # Always restore the original bridge function, even if Codex errors.
        service.generate_responses = original_generate_responses


@solver
def proxy_patched_codex_solver() -> Solver:
    """Run Codex while the sandbox Responses service is monkeypatched."""
    base_solver = as_solver(codex_cli(version="0.118.0", retry_refusals=1))

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # The patch wraps the whole Codex run, so any Responses call made by the
        # sandbox bridge during this sample can pass through the raw mutator.
        with response_web_search_status_patch():
            return await base_solver(state, generate)

    return solve


def _mark_web_search_failed(item: dict[str, Any]) -> None:
    """Best-effort raw Responses mutation for a failed web_search result."""
    item["status"] = "failed"

    replaced_existing_result = False
    for field in ("result", "results", "output"):
        if field in item:
            item[field] = WEB_SEARCH_FAILED_RESULT
            replaced_existing_result = True

    if isinstance(item.get("content"), list):
        item["content"] = [
            {
                "type": "output_text",
                "text": WEB_SEARCH_FAILED_RESULT,
            }
        ]
        replaced_existing_result = True
    elif "content" in item:
        item["content"] = WEB_SEARCH_FAILED_RESULT
        replaced_existing_result = True

    if not replaced_existing_result:
        item["result"] = WEB_SEARCH_FAILED_RESULT


def _print_web_search_item(label: str, item: dict[str, Any]) -> None:
    """Print the exact raw web_search_call item shape for this Inspect version."""
    print(
        f"[native_web_search_proxy_patch_slice] {label} web_search_call:\n"
        f"{json.dumps(item, indent=2, sort_keys=True, default=str)}",
        flush=True,
    )


@task
def native_web_search_proxy_patch_slice() -> Task:
    """One-sample task for raw Responses web_search mutation."""
    return Task(
        dataset=[
            Sample(
                id="native_web_search_proxy_patch_slice",
                input=(
                    "Use web search once, then answer briefly: "
                    "what is the title of the current OpenAI homepage?"
                ),
            )
        ],
        solver=proxy_patched_codex_solver(),
        sandbox="docker",
    )

