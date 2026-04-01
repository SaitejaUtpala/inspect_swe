"""Inspect-native reliability score reducers."""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Any

from inspect_ai.scorer import Score, ValueToFloat, score_reducer, value_to_float


@score_reducer(name="outcome_consistency")
def outcome_consistency_reducer(
    *,
    epsilon: float = 1e-8,
    value_to_float_fn: ValueToFloat = value_to_float(),
):
    """Compute consistency score across repeated binary outcomes."""

    def reduce(scores: list[Score]) -> Score:
        values = [1 if value_to_float_fn(score.value) >= 0.5 else 0 for score in scores]
        if len(values) < 2:
            return Score(value=1.0)
        p_hat = fmean(float(value) for value in values)
        variance = sum((value - p_hat) ** 2 for value in values) / (len(values) - 1)
        denom = p_hat * (1.0 - p_hat) + epsilon
        result = 1.0 - (variance / denom)
        return Score(value=max(0.0, min(1.0, float(result))))

    return reduce


@score_reducer(name="confidence_stability")
def confidence_stability_reducer(
    *,
    value_to_float_fn: ValueToFloat = value_to_float(),
):
    """Compute confidence stability across repeated runs."""

    def reduce(scores: list[Score]) -> Score:
        values = [float(value_to_float_fn(score.value)) for score in scores]
        if len(values) < 2:
            return Score(value=1.0)
        if len(set(values)) == 1:
            return Score(value=1.0)
        mean_abs = abs(fmean(values))
        std = pstdev(values)
        if mean_abs == 0:
            result = 1.0 / (1.0 + std)
        else:
            result = 1.0 / (1.0 + (std / mean_abs))
        return Score(value=float(result))

    return reduce


@score_reducer(name="brier_predictability")
def brier_predictability_reducer():
    """Compute one-minus-Brier-MSE from confidence/outcome pairs."""

    def reduce(scores: list[Score]) -> Score:
        pairs: list[tuple[float, int]] = []
        for score in scores:
            value = score.value
            if not isinstance(value, dict):
                continue
            confidence = _to_float(value.get("confidence"))
            outcome = _to_binary(value.get("outcome"))
            if confidence is None or outcome is None:
                continue
            confidence = max(0.0, min(1.0, confidence))
            pairs.append((confidence, outcome))
        if not pairs:
            return Score(value=0.0)
        mse = fmean((confidence - outcome) ** 2 for confidence, outcome in pairs)
        return Score(value=float(1.0 - mse))

    return reduce


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _to_binary(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        if value == 1:
            return 1
        if value == 0:
            return 0
        return None
    if isinstance(value, str):
        token = value.strip().upper()
        if token in {"C", "CORRECT", "PASS", "TRUE", "YES", "1"}:
            return 1
        if token in {"I", "INCORRECT", "FAIL", "FALSE", "NO", "0"}:
            return 0
    return None
