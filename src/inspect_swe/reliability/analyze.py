"""Small analysis helpers for reliability eval logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

from .eval_view import BaselineSampleView, extract_baseline_sample_views
from .metrics import compute_baseline_metrics, compute_fault_metrics


def read_reliability_views(
    paths: list[str | Path],
    *,
    expected_phase: str,
    expected_agent: str,
    expected_repeat_id: int = 0,
    strict_identity_tags: bool = False,
) -> tuple[list[BaselineSampleView], int]:
    """Read reliability sample views from one or more `.eval` log paths."""
    views: list[BaselineSampleView] = []
    warnings = 0
    for path in paths:
        log = read_eval_log(str(path))
        log_views, log_warnings = extract_baseline_sample_views(
            log,
            expected_phase=expected_phase,
            expected_agent=expected_agent,
            expected_repeat_id=expected_repeat_id,
            strict_identity_tags=strict_identity_tags,
        )
        views.extend(log_views)
        warnings += log_warnings
    return views, warnings


def summarize_baseline(
    paths: list[str | Path],
    *,
    agent: str,
    strict_identity_tags: bool = False,
) -> dict[str, Any]:
    """Return baseline metrics for eval logs."""
    views, warnings = read_reliability_views(
        paths,
        expected_phase="baseline",
        expected_agent=agent,
        strict_identity_tags=strict_identity_tags,
    )
    metrics = compute_baseline_metrics(views, [Path(path) for path in paths])
    metrics["identity_warnings"] = warnings
    return metrics


def summarize_fault(
    *,
    baseline_paths: list[str | Path],
    fault_paths: list[str | Path],
    agent: str,
    strict_identity_tags: bool = False,
) -> dict[str, Any]:
    """Return fault metrics compared with baseline eval logs."""
    baseline_views, baseline_warnings = read_reliability_views(
        baseline_paths,
        expected_phase="baseline",
        expected_agent=agent,
        strict_identity_tags=strict_identity_tags,
    )
    fault_views, fault_warnings = read_reliability_views(
        fault_paths,
        expected_phase="fault",
        expected_agent=agent,
        strict_identity_tags=strict_identity_tags,
    )
    metrics = compute_fault_metrics(
        baseline_views=baseline_views,
        baseline_eval_paths=[Path(path) for path in baseline_paths],
        fault_views=fault_views,
        fault_eval_paths=[Path(path) for path in fault_paths],
    )
    metrics["identity_warnings"] = baseline_warnings + fault_warnings
    return metrics
