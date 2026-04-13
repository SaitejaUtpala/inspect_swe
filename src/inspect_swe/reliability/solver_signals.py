"""Solver-based reliability signal collection."""

from __future__ import annotations

import inspect
from typing import Any

from inspect_ai.agent import as_solver, is_agent
from inspect_ai.scorer import Score
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .signals import extract_abstention, extract_confidence, extract_safety_violation


@solver(name="reliability_signal_collector")
def reliability_signal_collector(
    *,
    confidence_keys: tuple[str, ...] = (
        "reliability_confidence",
        "confidence",
        "self_confidence",
    ),
    safety_prefix: str = "reliability_safety_",
    abstention_prefix: str = "reliability_abstention_",
) -> Solver:
    """Collect standardized reliability signals into `state.scores`."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        output_text = _output_text(state)
        scores = dict(state.scores or {})

        if "reliability_confidence_signal" not in scores:
            confidence, source_key = extract_confidence(
                state.metadata, metadata_keys=confidence_keys, text=output_text
            )
            if confidence is not None:
                scores["reliability_confidence_signal"] = Score(
                    value=confidence,
                    metadata={"confidence": confidence, "source_key": source_key},
                )

        if "reliability_safety_violation" not in scores:
            violation, source_key = extract_safety_violation(
                state.metadata, metadata_prefix=safety_prefix
            )
            if violation is not None:
                scores["reliability_safety_violation"] = Score(
                    value=1 if violation else 0,
                    metadata={"violation": violation, "source_key": source_key},
                )

        if "reliability_abstention_signal" not in scores:
            abstained, source_key = extract_abstention(
                state.metadata,
                metadata_prefix=abstention_prefix,
                text=output_text,
            )
            if abstained is not None:
                scores["reliability_abstention_signal"] = Score(
                    value=1 if abstained else 0,
                    metadata={"abstained": abstained, "source_key": source_key},
                )

        state.scores = scores
        return state

    return solve


@solver(name="reliability_instrumented_solver")
def reliability_instrumented_solver(
    base_solver: Any,
    *,
    message_transform: Any = None,
) -> Solver:
    """Wrap a solver/agent with pre/post reliability instrumentation."""
    wrapped = as_solver(base_solver) if is_agent(base_solver) else base_solver
    collector = reliability_signal_collector()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if message_transform is not None:
            transformed = message_transform(state)
            state = await transformed if inspect.isawaitable(transformed) else transformed
        state = await wrapped(state, generate)
        return await collector(state, generate)

    return solve


def _output_text(state: TaskState) -> str | None:
    completion = getattr(state.output, "completion", None)
    if isinstance(completion, str) and completion.strip():
        return completion
    message = getattr(getattr(state.output, "message", None), "text", None)
    if isinstance(message, str) and message.strip():
        return message
    return None
