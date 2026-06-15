"""Slice: fault Codex `exec_command` by mutating tool-call arguments.

Run:

    inspect eval \
      slice_tests/generate_input_context_fault_slice.py@generate_input_context_fault_slice \
      --model openai/gpt-5.4-2026-03-05 \
      --max-samples 1

This is intentionally isolated. It demonstrates the pre-execution fault surface
for Codex-native tools:

1. The bridge filter calls the real model.
2. The model returns `tool_calls=[exec_command(...)]`.
3. The filter replaces the command with a controlled unavailable-tool failure.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import as_solver
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.tool import ToolCall
from inspect_swe._codex_cli.codex_cli import codex_cli


FAULTED_EXEC_SENTINEL = "/usr/local/bin/exec_command"


async def fault_first_exec_command_filter(model, messages, tools, tool_choice, config):
    """Mutate the first exec_command tool call into an unavailable-tool failure."""
    call_id = getattr(fault_first_exec_command_filter, "call_id", 0)
    fault_first_exec_command_filter.call_id = call_id + 1

    print(f"\n=== generate_input_context_fault_slice call {call_id}: input ===")
    print("tool_names:", [tool.name for tool in tools])
    print("tool_choice:", tool_choice)
    for index, message in enumerate(messages):
        print(f"message[{index}]:", _message_summary(message))

    output = await model.generate(
        input=messages,
        tools=tools,
        tool_choice=tool_choice,
        config=config,
    )

    print(f"=== generate_input_context_fault_slice call {call_id}: output ===")
    print("stop_reason:", output.stop_reason)
    print("assistant:", _message_summary(output.message))
    print("tool_calls:", _tool_call_summaries(output.message))

    if _has_already_faulted_exec(messages) or not _has_exec_command(output.message):
        return output

    mutated_message = _with_faulted_exec_command(output.message)
    print("mutated_tool_calls:", _tool_call_summaries(mutated_message))
    return ModelOutput.from_message(mutated_message, stop_reason=output.stop_reason)


def _has_already_faulted_exec(messages: list[Any]) -> bool:
    return any(FAULTED_EXEC_SENTINEL in getattr(message, "text", "") for message in messages)


def _has_exec_command(message: ChatMessageAssistant) -> bool:
    return any(call.function == "exec_command" for call in message.tool_calls or [])


def _with_faulted_exec_command(message: ChatMessageAssistant) -> ChatMessageAssistant:
    tool_calls = []
    for call in message.tool_calls or []:
        if call.function != "exec_command":
            tool_calls.append(call)
            continue
        arguments = dict(call.arguments)
        arguments["cmd"] = _fail_before_original_command(arguments.get("cmd", ""))
        tool_calls.append(
            ToolCall(
                id=call.id,
                function=call.function,
                arguments=arguments,
                parse_error=call.parse_error,
                view=call.view,
                type=call.type,
            )
        )
    return message.model_copy(update={"tool_calls": tool_calls})


def _fail_before_original_command(command: Any) -> str:
    original_command = str(command).strip()
    if not original_command:
        return FAULTED_EXEC_SENTINEL
    return f"{FAULTED_EXEC_SENTINEL} && {{\n{original_command}\n}}"


def _tool_call_summaries(message: ChatMessageAssistant) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id,
            "function": call.function,
            "arguments": call.arguments,
            "type": call.type,
        }
        for call in message.tool_calls or []
    ]


def _message_summary(message: Any) -> dict[str, Any]:
    return {
        "type": type(message).__name__,
        "role": getattr(message, "role", None),
        "text": _truncate(getattr(message, "text", "")),
        "content_types": [
            type(item).__name__ for item in getattr(message, "content_list", [])
        ],
        "tool_call_id": getattr(message, "tool_call_id", None),
        "function": getattr(message, "function", None),
        "tool_calls": [
            {
                "function": call.function,
                "arguments": call.arguments,
                "type": call.type,
            }
            for call in getattr(message, "tool_calls", None) or []
        ],
        "raw_content": _truncate(json.dumps(_jsonable_content(message), default=str)),
    }


def _jsonable_content(message: Any) -> Any:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return [
            item.model_dump() if hasattr(item, "model_dump") else repr(item)
            for item in content
        ]
    return content


def _truncate(value: str, limit: int = 700) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated>"


def _exec_instruction() -> str:
    return (
        "Run exactly one shell command to inspect the current directory. "
        "Prefer `ls`. If that command fails, recover by trying a corrected command. "
        "Then answer whether you saw any files."
    )


def _solver():
    return as_solver(
        codex_cli(
            version="0.118.0",
            filter=fault_first_exec_command_filter,
            retry_refusals=1,
        )
    )


@task
def generate_input_context_fault_slice() -> Task:
    """One-sample task for pre-execution exec_command argument faulting."""
    return Task(
        dataset=[
            Sample(
                id="generate_input_context_fault_slice",
                input=_exec_instruction(),
            )
        ],
        solver=_solver(),
        sandbox="docker",
    )

