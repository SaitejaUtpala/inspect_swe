from __future__ import annotations

from inspect_ai.scorer import Score
from inspect_swe.reliability.reducers import (
    brier_predictability_reducer,
    confidence_stability_reducer,
    outcome_consistency_reducer,
)


def test_outcome_consistency_reducer_prefers_consistent_runs() -> None:
    reducer = outcome_consistency_reducer()
    consistent = reducer([Score(value=1), Score(value=1), Score(value=1)])
    inconsistent = reducer([Score(value=1), Score(value=0), Score(value=1)])

    assert consistent.value == 1.0
    assert 0.0 <= float(inconsistent.value) < 1.0


def test_confidence_stability_reducer_handles_constant_values() -> None:
    reducer = confidence_stability_reducer()
    reduced = reducer([Score(value=0.6), Score(value=0.6), Score(value=0.6)])
    assert reduced.value == 1.0


def test_brier_predictability_reducer_returns_one_minus_mse() -> None:
    reducer = brier_predictability_reducer()
    reduced = reducer(
        [
            Score(value={"confidence": 0.9, "outcome": 1}),
            Score(value={"confidence": 0.1, "outcome": 0}),
        ]
    )
    assert float(reduced.value) > 0.9
