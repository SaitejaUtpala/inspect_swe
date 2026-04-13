from __future__ import annotations

import asyncio

from inspect_ai.model import ModelOutput
from inspect_ai.solver import TaskState, solver
from inspect_swe.reliability.solver_signals import (
    reliability_instrumented_solver,
    reliability_signal_collector,
)


def test_reliability_signal_collector_uses_metadata_and_text() -> None:
    collector = reliability_signal_collector()
    state = _state(
        metadata={"confidence": "83"},
        completion="I cannot complete this task with the provided information.",
    )

    scored = asyncio.run(collector(state, None))

    assert scored.scores is not None
    assert scored.scores["reliability_confidence_signal"].metadata["confidence"] == 0.83
    assert scored.scores["reliability_abstention_signal"].metadata["abstained"] is True


def test_reliability_instrumented_solver_collects_even_when_completed() -> None:
    @solver
    def _base_solver():
        async def run(state: TaskState, generate):
            del generate
            state.completed = True
            state.output = ModelOutput(model="mock/model", completion="confidence: 90%")
            return state

        return run

    wrapped = reliability_instrumented_solver(_base_solver())
    scored = asyncio.run(wrapped(_state(metadata={}, completion=""), None))

    assert scored.completed is True
    assert scored.scores is not None
    assert scored.scores["reliability_confidence_signal"].metadata["confidence"] == 0.9


def _state(*, metadata: dict[str, object], completion: str) -> TaskState:
    return TaskState(
        model="mock/model",
        sample_id=1,
        epoch=0,
        input="task",
        messages=[],
        metadata=metadata,
        output=ModelOutput(model="mock/model", completion=completion),
    )
