"""Slice: rewrite native Codex `web_search` output as a failed result.

Run:

    inspect eval \
      slice_tests/native_web_search_filter_result_slice.py@native_web_search_filter_result_slice \
      --model openai/gpt-5.4-2026-03-05 \
      --max-samples 1

This is the recoverability-oriented variant of
`native_web_search_filter_convert_slice.py`: call the real model from inside the
bridge filter, detect native `web_search`, and replace its result text with a
plausible failed observation instead of setting the `error` field.

This helps test whether Codex continues when the failed native tool is presented
as observation content rather than as an Inspect-style tool error.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai._util.content import ContentToolUse
from inspect_ai.agent import as_solver
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_swe._codex_cli.codex_cli import codex_cli

WEB_SEARCH_FAILED_RESULT = (
    "web_search failed. The search service returned no results due to a "
    "transient error. Please retry or continue with another approach."
)


async def web_search_failed_result_filter(model, messages, tools, tool_choice, config):
    """Run the model, then rewrite web_search calls as failed result content."""
    if not any(tool.name == "web_search" for tool in tools):
        return None

    # Let the real model run first. If it wants native web_search, the output
    # here normally contains a pending ToolCall(function="web_search").
    output = await model.generate(
        input=messages,
        tools=tools,
        tool_choice=tool_choice,
        config=config,
    )
    message = output.message
    content = message.content_list
    _print_message_types("before", message, content)
    mutated = False

    remaining_tool_calls = []
    for call in message.tool_calls or []:
        if call.function != "web_search":
            remaining_tool_calls.append(call)
            continue

        # This is the fragile part: instead of letting Codex receive and execute
        # the native pending ToolCall, we replace it with an Inspect content item
        # that looks like a completed web_search observation.
        content.append(
            ContentToolUse(
                tool_type="web_search",
                id=_web_search_call_id(call.id),
                name="search",
                arguments=json.dumps(
                    {"type": "search", "query": _web_search_query(call.arguments)},
                    sort_keys=True,
                ),
                result=WEB_SEARCH_FAILED_RESULT,
            )
        )
        mutated = True

    for item in content:
        if isinstance(item, ContentToolUse) and item.tool_type == "web_search":
            # If the bridge already represented web_search as ContentToolUse,
            # keep the shape but make the result look like a failed search.
            item.result = WEB_SEARCH_FAILED_RESULT
            item.error = None
            mutated = True

    if not mutated:
        return output

    replacement = ChatMessageAssistant(
        content=content,
        tool_calls=remaining_tool_calls or None,
        model=message.model,
        source=message.source,
    )
    _print_message_types("after", replacement, replacement.content_list)
    # The returned message no longer has the original web_search ToolCall. That
    # is why this is useful as a slice test, but not a clean production hook.
    return ModelOutput.from_message(replacement, stop_reason=output.stop_reason)


@task
def native_web_search_filter_result_slice() -> Task:
    """One-sample task for native web_search failed-result mutation."""
    return Task(
        dataset=[
            Sample(
                id="native_web_search_filter_result_slice",
                input=(
                    "Use web search once, then answer briefly: "
                    "how many studio albums did Mercedes Sosa publish from 2000 "
                    "through 2009 inclusive?"
                ),
            )
        ],
        solver=as_solver(
            codex_cli(
                version="0.118.0",
                filter=web_search_failed_result_filter,
                retry_refusals=1,
            )
        ),
        sandbox="docker",
    )


def _web_search_query(arguments: dict[str, Any]) -> str:
    value = arguments.get("query") or arguments.get("input") or arguments.get("q")
    if isinstance(value, str):
        return value
    return json.dumps(arguments, sort_keys=True)


def _web_search_call_id(call_id: str) -> str:
    if call_id.startswith("ws"):
        return call_id
    digest = hashlib.sha256(call_id.encode("utf-8")).hexdigest()
    return f"ws_{digest[:16]}"


def _print_message_types(
    label: str,
    message: ChatMessageAssistant,
    content: list[Any],
) -> None:
    print(
        "[native_web_search_filter_result_slice] "
        f"{label}: message={type(message).__name__}, "
        f"content={[type(item).__name__ for item in content]}, "
        f"tool_calls={[type(call).__name__ for call in message.tool_calls or []]}",
        flush=True,
    )

