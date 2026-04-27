"""Reliability metric computation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .eval_view import BaselineSampleView


def compute_baseline_metrics(
    views: list[BaselineSampleView], eval_paths: list[Path]
) -> dict[str, Any]:
    """Compute baseline metrics using hal-harness field names."""
    success_values = [_binary_success(view.scores) for view in views]
    successes = [value for value in success_values if value is not None]
    confidences_percent = [
        view.confidence_value for view in views if view.confidence_value is not None
    ]
    confidences = [value / 100.0 for value in confidences_percent]

    by_sample_success: dict[str, list[float]] = {}
    by_sample_confidence: dict[str, list[float]] = {}
    by_sample_time: dict[str, list[float]] = {}
    by_sample_calls: dict[str, list[float]] = {}

    agent_names = sorted({view.agent for view in views if view.agent})
    unique_sample_ids = sorted({str(view.sample_id) for view in views})

    for view, success in zip(views, success_values):
        sample_key = str(view.sample_id)
        if success is not None:
            by_sample_success.setdefault(sample_key, []).append(success)
        if view.confidence_value is not None:
            by_sample_confidence.setdefault(sample_key, []).append(view.confidence_value)
        if view.total_time is not None:
            by_sample_time.setdefault(sample_key, []).append(float(view.total_time))
        by_sample_calls.setdefault(sample_key, []).append(float(view.tool_call_count))

    accuracy = _ratio(sum(successes), len(successes))
    accuracy_se = _binomial_se(accuracy, len(successes))

    consistency_outcome_values = [
        _outcome_consistency(values)
        for values in by_sample_success.values()
        if len(values) >= 2
    ]
    consistency_outcome = _mean(consistency_outcome_values)
    consistency_outcome_se = _std_err(consistency_outcome_values)

    confidence_consistency_values = [
        _confidence_consistency(values)
        for values in by_sample_confidence.values()
        if len(values) >= 2
    ]
    consistency_confidence = _mean(confidence_consistency_values)
    consistency_confidence_se = _std_err(confidence_consistency_values)

    time_cvs = [
        _coefficient_of_variation(values)
        for values in by_sample_time.values()
        if len(values) >= 2
    ]
    api_call_cvs = [
        _coefficient_of_variation(values)
        for values in by_sample_calls.values()
        if len(values) >= 2
    ]
    resource_cv_values = [value for value in time_cvs + api_call_cvs if value is not None]
    consistency_resource = _exp_negative_mean(resource_cv_values)
    consistency_resource_se = _std_err(
        [_exp_negative_mean([value]) for value in resource_cv_values]
    )

    mean_conf_cv = _mean(
        [
            value
            for value in (
                _coefficient_of_variation(values)
                for values in by_sample_confidence.values()
                if len(values) >= 2
            )
            if value is not None
        ]
    )

    paired_confidences: list[float] = []
    paired_successes: list[float] = []
    for view, success in zip(views, success_values):
        if success is None or view.confidence_value is None:
            continue
        paired_confidences.append(view.confidence_value / 100.0)
        paired_successes.append(success)

    predictability_calibration = _predictability_calibration(
        paired_confidences, paired_successes
    )
    predictability_roc_auc = _predictability_roc_auc(paired_confidences, paired_successes)
    predictability_brier_score = _predictability_brier_score(
        paired_confidences, paired_successes
    )
    predictability_rate_confidence_correlation = _predictability_rate_confidence_correlation(
        paired_confidences, paired_successes
    )

    agent_name = (
        "multiple_agents"
        if len(agent_names) > 1
        else (agent_names[0] if agent_names else "unknown_agent")
    )

    return {
        "agent": agent_name,
        "num_tasks": len(unique_sample_ids),
        "num_runs": len(eval_paths),
        "accuracy": accuracy,
        "consistency_outcome": consistency_outcome,
        "consistency_trajectory_distribution": None,
        "consistency_trajectory_sequence": None,
        "consistency_confidence": consistency_confidence,
        "consistency_resource": consistency_resource,
        "mean_time_cv": _mean([value for value in time_cvs if value is not None]),
        "mean_cost_cv": None,
        "mean_api_calls_cv": _mean([value for value in api_call_cvs if value is not None]),
        "mean_actions_cv": None,
        "mean_errors_cv": None,
        "mean_call_latency_cv": None,
        "mean_conf_cv": mean_conf_cv,
        "predictability_rate_confidence_correlation": predictability_rate_confidence_correlation,
        "predictability_calibration": predictability_calibration,
        "predictability_roc_auc": predictability_roc_auc,
        "predictability_brier_score": predictability_brier_score,
        "mean_confidence": _mean(confidences),
        "robustness_fault_injection": None,
        "robustness_structural": None,
        "robustness_prompt_variation": None,
        "safety_harm_severity": None,
        "safety_compliance": None,
        "safety_score": None,
        "abstention_rate": None,
        "abstention_precision": None,
        "abstention_recall": None,
        "abstention_selective_accuracy": None,
        "abstention_calibration": None,
        "accuracy_se": accuracy_se,
        "consistency_outcome_se": consistency_outcome_se,
        "consistency_trajectory_distribution_se": None,
        "consistency_trajectory_sequence_se": None,
        "consistency_confidence_se": consistency_confidence_se,
        "consistency_resource_se": consistency_resource_se,
        "predictability_calibration_se": None,
        "predictability_roc_auc_se": None,
        "predictability_brier_score_se": None,
        "robustness_fault_injection_se": None,
        "robustness_structural_se": None,
        "robustness_prompt_variation_se": None,
    }


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
        return None
    if isinstance(value, dict):
        for key in ("value", "score", "reward", "correct", "passed", "pass"):
            if key in value:
                return _parse_binary(value[key])
    return None


def _outcome_consistency(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    denom = (mean * (1.0 - mean)) + 1e-8
    raw = 1.0 - (variance / denom)
    return max(0.0, min(1.0, raw))


def _confidence_consistency(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = variance**0.5
    cv = std / mean
    return 2.718281828459045 ** (-cv)


def _coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean <= 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return (variance**0.5) / mean


def _exp_negative_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return 2.718281828459045 ** (-(sum(values) / len(values)))


def _std_err(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if len(cleaned) < 2:
        return None
    mean = sum(cleaned) / len(cleaned)
    variance = sum((value - mean) ** 2 for value in cleaned) / (len(cleaned) - 1)
    return (variance**0.5) / (len(cleaned) ** 0.5)


def _binomial_se(probability: float | None, count: int) -> float | None:
    if probability is None or count <= 1:
        return None
    return ((probability * (1.0 - probability)) / count) ** 0.5


def _predictability_calibration(
    confidences: list[float], outcomes: list[float], n_bins: int = 10
) -> float | None:
    pairs = list(zip(confidences, outcomes))
    if not pairs:
        return None
    ece = 0.0
    total = len(pairs)
    for bin_index in range(n_bins):
        low = bin_index / n_bins
        high = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            bucket = [pair for pair in pairs if low <= pair[0] <= high]
        else:
            bucket = [pair for pair in pairs if low <= pair[0] < high]
        if not bucket:
            continue
        avg_conf = sum(pair[0] for pair in bucket) / len(bucket)
        avg_acc = sum(pair[1] for pair in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_conf - avg_acc)
    return 1.0 - ece


def _predictability_brier_score(
    confidences: list[float], outcomes: list[float]
) -> float | None:
    pairs = list(zip(confidences, outcomes))
    if not pairs:
        return None
    brier = sum((conf - outcome) ** 2 for conf, outcome in pairs) / len(pairs)
    return 1.0 - brier


def _predictability_roc_auc(
    confidences: list[float], outcomes: list[float]
) -> float | None:
    pairs = sorted(zip(confidences, outcomes), key=lambda item: item[0])
    if not pairs:
        return None
    positives = [pair for pair in pairs if pair[1] >= 0.5]
    negatives = [pair for pair in pairs if pair[1] < 0.5]
    if not positives or not negatives:
        return None

    ranks: list[float] = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        tie_end = index
        while tie_end + 1 < len(pairs) and pairs[tie_end + 1][0] == pairs[index][0]:
            tie_end += 1
        avg_rank = (index + 1 + tie_end + 1) / 2.0
        for i in range(index, tie_end + 1):
            ranks[i] = avg_rank
        index = tie_end + 1

    sum_positive_ranks = sum(
        rank for rank, (_, outcome) in zip(ranks, pairs) if outcome >= 0.5
    )
    n_pos = len(positives)
    n_neg = len(negatives)
    return (sum_positive_ranks - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def _predictability_rate_confidence_correlation(
    confidences: list[float], outcomes: list[float]
) -> float | None:
    pairs = list(zip(confidences, outcomes))
    if len(pairs) < 2:
        return None
    if len({outcome for _, outcome in pairs}) < 2:
        return None
    ranks_conf = _average_ranks([pair[0] for pair in pairs])
    ranks_out = _average_ranks([pair[1] for pair in pairs])
    corr = _pearson(ranks_conf, ranks_out)
    if corr is None:
        return None
    return (corr + 1.0) / 2.0


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        tie_end = index
        while tie_end + 1 < len(indexed) and indexed[tie_end + 1][1] == indexed[index][1]:
            tie_end += 1
        avg_rank = (index + 1 + tie_end + 1) / 2.0
        for item_index in range(index, tie_end + 1):
            original_pos = indexed[item_index][0]
            ranks[original_pos] = avg_rank
        index = tie_end + 1
    return ranks


def _pearson(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(values_x, values_y))
    var_x = sum((x - mean_x) ** 2 for x in values_x)
    var_y = sum((y - mean_y) ** 2 for y in values_y)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / ((var_x**0.5) * (var_y**0.5))


def _mean(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _ratio(numerator: float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / float(denominator)
