"""Slice: fault Codex `exec_command` by rewriting its observation.

Run:

    inspect eval \
      slice_tests/exec_observation_fault_slice.py@exec_observation_fault_slice \
      --model openai/gpt-5.4-2026-03-05 \
      --max-samples 1

High-level flow:

1. The first bridge filter call sees only the user prompt, so it returns `None`.
2. The bridge lets the real model generate normally.
3. Codex emits an unchanged `exec_command` tool call and its runtime executes it.
4. The next bridge filter call receives the tool observation in `messages`.
5. The filter calls the real model itself with a rewritten observation and
   returns the resulting `ModelOutput`.

Important: this is not a pre-execution fault. The sandbox command already ran.
This slice tests a different failure surface: the observation handed back to the
model is corrupted to look like a transient shell/runner failure. We call
`model.generate(...)` ourselves with the rewritten messages instead of returning
`GenerateInput`, so the mutation is scoped to this intercepted model call.

The first exec observation is always faulted. Later exec observations are
faulted according to `fault_rate`.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from inspect_ai import Task, task
from inspect_ai.agent import as_solver
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageTool
from inspect_swe._codex_cli.codex_cli import codex_cli


FAULTED_EXEC_OBSERVATION = "bash: fork: Resource temporarily unavailable"
DEFAULT_FAULT_RATE = 1.0
FAULT_SEED = 0


def make_fault_exec_observation_filter(fault_rate: float = DEFAULT_FAULT_RATE):
    """Create a stateful bridge filter for observation-level exec faults.

    The closure owns two pieces of per-eval state:

    * `seen_tool_call_ids` prevents old tool observations from being sampled
      again when the full conversation history comes back through the filter.
    * `faulted_count` lets the slice force the first exec observation to fail,
      then use `fault_rate` for retries or later commands.
    """
    if not 0.0 <= fault_rate <= 1.0:
        raise ValueError("fault_rate must be between 0.0 and 1.0")
    seen_tool_call_ids: set[str] = set()
    faulted_count = 0

    async def fault_exec_observation_filter(model, messages, tools, tool_choice, config):
        nonlocal faulted_count
        # The bridge calls this before every model generation. If there is no
        # exec observation yet, returning None lets Codex/model proceed normally.
        # Once an exec observation exists, we can choose to call the real model
        # ourselves with a rewritten copy of the messages and return that output.
        #
        # That is intentionally different from returning GenerateInput: we do
        # not ask the bridge to continue with mutated messages as its next
        # canonical input; we perform the intercepted generation here.
        call_id = getattr(fault_exec_observation_filter, "call_id", 0)
        fault_exec_observation_filter.call_id = call_id + 1

        print(
            f"\n=== exec_observation_fault_slice call {call_id}: input "
            f"(fault_rate={fault_rate}) ==="
        )
        for index, message in enumerate(messages):
            print(f"message[{index}]:", _message_summary(message))

        faulted_messages = _with_faulted_exec_observation(
            messages,
            fault_rate=fault_rate,
            seen_tool_call_ids=seen_tool_call_ids,
            force_fault=faulted_count == 0,
        )
        if faulted_messages is None:
            return None
        faulted_count += 1

        print("=== exec_observation_fault_slice: faulted input ===")
        for index, message in enumerate(faulted_messages):
            print(f"message[{index}]:", _message_summary(message))

        return await model.generate(
            input=faulted_messages,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
        )

    return fault_exec_observation_filter


def _with_faulted_exec_observation(
    messages: list[Any],
    *,
    fault_rate: float,
    seen_tool_call_ids: set[str],
    force_fault: bool,
) -> list[Any] | None:
    """Return a copied message list with one exec observation rewritten.

    The assistant message that requested the command is not touched. Only the
    matching `ChatMessageTool` result is replaced, which keeps the transcript
    shape realistic: Codex asked for `ls`, the runtime produced a tool result,
    and the model is now told that the shell runner failed.
    """
    if _has_already_faulted_exec_observation(messages):
        return None

    # Walk backward so we fault the latest exec_command observation. The earlier
    # assistant tool call still contains the exact command the model requested.
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not _is_exec_observation(message):
            continue
        tool_call_id = str(getattr(message, "tool_call_id", ""))
        if tool_call_id in seen_tool_call_ids:
            continue
        seen_tool_call_ids.add(tool_call_id)

        # The first fault makes the slice deterministic: every one-sample run
        # exercises recovery at least once. Subsequent exec observations use the
        # configured probability, which is closer to a fault campaign.
        if not force_fault and not _should_fault_observation(message, fault_rate):
            print("=== exec_observation_fault_slice: sampled no fault ===")
            return None
        if force_fault:
            print("=== exec_observation_fault_slice: forcing first fault ===")
        faulted_messages = list(messages)
        faulted_messages[index] = message.model_copy(
            update={"content": _faulted_observation_text(message)}
        )
        return faulted_messages
    return None


def _should_fault_observation(message: ChatMessageTool, fault_rate: float) -> bool:
    """Sample a stable fault decision for this specific tool observation."""
    decision_key = ":".join(
        [
            str(FAULT_SEED),
            str(getattr(message, "tool_call_id", "")),
            str(getattr(message, "function", "")),
            message.text,
        ]
    )
    digest = hashlib.sha256(decision_key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16)).random() < fault_rate


def _has_already_faulted_exec_observation(messages: list[Any]) -> bool:
    return any(FAULTED_EXEC_OBSERVATION in getattr(message, "text", "") for message in messages)


def _is_exec_observation(message: Any) -> bool:
    return (
        isinstance(message, ChatMessageTool)
        and getattr(message, "function", None) == "exec_command"
    )


def _faulted_observation_text(message: ChatMessageTool) -> str:
    """Build a realistic-looking terminal observation for a transient failure.

    We preserve the `Command:` line from the real observation so Codex still sees
    the exact command it requested. The rest is shaped like the existing
    `exec_command` transcript: wall time, non-zero exit code, token count, and
    output text. The error itself mirrors a real bash failure that can happen
    when the shell cannot start another process.
    """
    command_line = _extract_command_line(message.text)
    if command_line is None:
        command_line = "Command: /bin/bash -lc <redacted>"
    return (
        f"{command_line}\n"
        "Wall time: 0.0000 seconds\n"
        "Process exited with code 254\n"
        "Original token count: 8\n"
        "Output:\n"
        f"{FAULTED_EXEC_OBSERVATION}\n"
    )


def _extract_command_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("Command: "):
            return line
    return None


def _message_summary(message: Any) -> dict[str, Any]:
    return {
        "type": type(message).__name__,
        "role": getattr(message, "role", None),
        "text": _truncate(getattr(message, "text", "")),
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
    }


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


@task
def exec_observation_fault_slice(fault_rate: float = DEFAULT_FAULT_RATE) -> Task:
    """One-sample task for post-execution exec_command observation faulting."""
    return Task(
        dataset=[
            Sample(
                id="exec_observation_fault_slice",
                input=_exec_instruction(),
            )
        ],
        solver=as_solver(
            codex_cli(
                version="0.118.0",
                filter=make_fault_exec_observation_filter(fault_rate),
                retry_refusals=1,
            )
        ),
        sandbox="docker",
    )
