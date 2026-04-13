from __future__ import annotations

import asyncio
from types import SimpleNamespace

from inspect_swe.reliability.scorers import (
    reliability_abstention_signal,
    reliability_confidence_signal,
    reliability_safety_violation,
)


def test_reliability_confidence_signal_extracts_and_clamps() -> None:
    scorer = reliability_confidence_signal()
    state = SimpleNamespace(metadata={"confidence": "1.3"})

    score = asyncio.run(scorer(state, None))

    assert score is not None
    assert score.value == 1.0
    assert score.metadata["source_key"] == "confidence"


def test_reliability_safety_violation_detects_violation_flag() -> None:
    scorer = reliability_safety_violation()
    state = SimpleNamespace(metadata={"reliability_safety_violation": True})

    score = asyncio.run(scorer(state, None))

    assert score is not None
    assert score.value == 1
    assert score.metadata["violation"] is True


def test_reliability_abstention_signal_detects_abstained_flag() -> None:
    scorer = reliability_abstention_signal()
    state = SimpleNamespace(metadata={"reliability_abstention_abstained": "true"})

    score = asyncio.run(scorer(state, None))

    assert score is not None
    assert score.value == 1
    assert score.metadata["abstained"] is True
