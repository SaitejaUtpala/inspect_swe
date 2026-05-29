"""Reliability metric computation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .eval_view import BaselineSampleView


def compute_baseline_metrics(
    views: list[BaselineSampleView], eval_paths: list[Path]
) -> dict[str, Any]:
    """Compute a compact baseline metrics dictionary."""
    success_values = [_binary_success(view.scores) for view in views]
    successes = [value for value in success_values if value is not None]
    confidences_percent = [
        view.confidence_value for view in views if view.confidence_value is not None
    ]
    confidences = [value / 100.0 for value in confidences_percent]
    agent_names = sorted({view.agent for view in views if view.agent})
    unique_sample_ids = sorted({str(view.sample_id) for view in views})
    accuracy = _ratio(sum(successes), len(successes))

    return {
        "agent": (
            "multiple_agents"
            if len(agent_names) > 1
            else (agent_names[0] if agent_names else "unknown_agent")
        ),
        "num_tasks": len(unique_sample_ids),
        "num_runs": len(eval_paths),
        "accuracy": accuracy,
        "accuracy_se": _binomial_se(accuracy, len(successes)),
        "mean_confidence": _mean(confidences),
        "robustness_fault_injection": None,
        "robustness_fault_injection_se": None,
    }


def compute_fault_metrics(
    *,
    baseline_views: list[BaselineSampleView],
    baseline_eval_paths: list[Path],
    fault_views: list[BaselineSampleView],
    fault_eval_paths: list[Path],
) -> dict[str, Any]:
    """Compute fault robustness metrics against a baseline view."""
    metrics = compute_baseline_metrics(fault_views, fault_eval_paths)
    baseline_accuracy = _accuracy(baseline_views)
    fault_accuracy = _accuracy(fault_views)
    robustness = None
    if baseline_accuracy is not None and baseline_accuracy > 1e-8 and fault_accuracy is not None:
        robustness = min(fault_accuracy / baseline_accuracy, 1.0)
    metrics.update(
        {
            "accuracy": fault_accuracy,
            "baseline_acc": baseline_accuracy,
            "fault_acc": fault_accuracy,
            "baseline_num_runs": len(baseline_eval_paths),
            "robustness_fault_injection": robustness,
        }
    )
    return metrics


def _binary_success(scores: dict[str, Any]) -> float | None:
    for key in ("reward", "accuracy", "score", "correct", "passed", "pass"):
        if key in scores:
            parsed = _parse_binary(scores[key])
            if parsed is not None:
                return parsed
    for value in scores.values():
        parsed = _parse_binary(value)
        if parsed is not None:
            return parsed
    return None


def _accuracy(views: list[BaselineSampleView]) -> float | None:
    successes = [
        success
        for success in (_binary_success(view.scores) for view in views)
        if success is not None
    ]
    return _ratio(sum(successes), len(successes))


def _parse_binary(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) >= 0.5 else 0.0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"c", "correct", "pass", "passed", "true", "1", "yes", "success"}:
            return 1.0
        if token in {"i", "incorrect", "fail", "failed", "false", "0", "no", "error"}:
            return 0.0
    if isinstance(value, dict):
        for key in ("value", "score", "reward", "correct", "passed", "pass"):
            if key in value:
                return _parse_binary(value[key])
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(numerator: float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _binomial_se(probability: float | None, count: int) -> float | None:
    if probability is None or count <= 1:
        return None
    return ((probability * (1.0 - probability)) / count) ** 0.5
