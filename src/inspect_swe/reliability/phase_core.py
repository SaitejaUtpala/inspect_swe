"""Shared helpers for reliability execution phases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from inspect_ai import eval
from inspect_ai.log import EvalLog

from .artifacts import assert_canonical_eval_log_path, load_sidecar_records
from .hooks import ReliabilityHookConfig, configure_reliability_hooks
from .paths import campaign_sidecar_path
from .telemetry import TelemetryCoverageReport, assess_sidecar_coverage


@dataclass(frozen=True)
class PhaseRepeatExecution:
    """Runtime result for one agent/repeat execution."""

    agent: str
    repeat_id: int
    logs: list[EvalLog]
    coverage: TelemetryCoverageReport


def default_campaign_id() -> str:
    """Return a default campaign id with timestamp + short suffix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def resolve_phase_sidecar_path(
    *,
    explicit_sidecar_path: str | None,
    log_root: str,
    benchmark: str,
    phase: str,
    campaign_id: str,
) -> str:
    """Resolve canonical campaign sidecar path for a phase."""
    if explicit_sidecar_path:
        return explicit_sidecar_path
    return str(
        campaign_sidecar_path(
            log_root=log_root,
            benchmark=benchmark,
            phase=phase,
            campaign_id=campaign_id,
            sidecar_filename="records.jsonl",
        )
    )


def configure_hooks_for_phase(
    *,
    configure_hooks: bool,
    sidecar_path: str,
    strict_identity_tags: bool,
) -> None:
    """Configure reliability hooks for one phase run if requested."""
    if not configure_hooks:
        return
    configure_reliability_hooks(
        ReliabilityHookConfig(
            enabled=True,
            sidecar_path=sidecar_path,
            strict_identity_tags=strict_identity_tags,
        )
    )


def build_phase_metadata(
    *,
    base_metadata: dict[str, Any],
    phase: str,
    campaign_id: str,
    repeat_id: int,
    agent: str,
    benchmark: str,
    seed: int,
    perturbation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build standardized phase metadata tags for inspect eval runs."""
    metadata = dict(base_metadata)
    metadata.update(
        {
            "reliability_phase": phase,
            "reliability_campaign_id": campaign_id,
            "reliability_repeat_id": repeat_id,
            "reliability_agent_attempt_id": 0,
            "reliability_agent": agent,
            "reliability_benchmark": benchmark,
            "reliability_seed": seed,
        }
    )
    if perturbation_metadata:
        metadata.update(perturbation_metadata)
    return metadata


def run_eval_for_repeat(
    *,
    tasks: Any,
    metadata: dict[str, Any],
    log_dir: str,
    task_args: dict[str, Any],
    model: str | None,
    solver: Any | None,
    sandbox: str | None,
    limit: int | tuple[int, int] | None,
    sample_id: str | int | list[str] | list[int] | list[str | int] | None,
    max_tasks: int | None,
    max_samples: int | None,
    max_subprocesses: int | None,
    max_sandboxes: int | None,
    max_connections: int | None,
    eval_fn: Any = eval,
) -> list[EvalLog]:
    """Execute one inspect eval run for a phase repeat."""
    eval_kwargs: dict[str, Any] = {
        "tasks": tasks,
        "metadata": metadata,
        "log_dir": log_dir,
        "log_format": "eval",
        "score": True,
        "sample_shuffle": False,
        "max_tasks": max_tasks,
        "max_samples": max_samples,
        "max_subprocesses": max_subprocesses,
        "max_sandboxes": max_sandboxes,
        "sandbox": sandbox,
        "limit": limit,
        "sample_id": sample_id,
    }
    if task_args:
        eval_kwargs["task_args"] = task_args
    if model is not None:
        eval_kwargs["model"] = model
    if solver is not None:
        eval_kwargs["solver"] = solver
    if max_connections is not None:
        eval_kwargs["max_connections"] = max_connections

    eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
    logs = eval_fn(**eval_kwargs)
    for log in logs:
        if log.location:
            assert_canonical_eval_log_path(log.location)
    return logs


def assess_repeat_coverage_for_phase(
    *,
    logs: list[EvalLog],
    sidecar_path: str,
    agent: str,
    repeat_id: int,
    phase: str,
) -> TelemetryCoverageReport:
    """Assess sidecar coverage for a single phase repeat."""
    records = load_sidecar_records(sidecar_path)
    filtered = [
        record
        for record in records
        if (
            record.identity.agent == agent
            or record.metadata.get("reliability_agent") == agent
        )
        and record.identity.repeat_id == repeat_id
        and record.identity.phase == phase
    ]

    reports: list[TelemetryCoverageReport] = []
    for log in logs:
        run_filtered = [
            record for record in filtered if record.identity.run_id == log.eval.run_id
        ]
        reports.append(assess_sidecar_coverage(log, run_filtered))

    return combine_coverage_reports(reports)


def combine_coverage_reports(
    reports: list[TelemetryCoverageReport],
) -> TelemetryCoverageReport:
    """Combine per-log coverage into one repeat-level report."""
    missing: list[str] = []
    duplicates: list[str] = []
    expected = 0
    observed = 0
    warnings = 0
    for report in reports:
        expected += report.expected_samples
        observed += report.observed_records
        missing.extend(report.missing_sample_uuids)
        duplicates.extend(report.duplicate_identity_keys)
        warnings += report.identity_warning_count
    return TelemetryCoverageReport(
        expected_samples=expected,
        observed_records=observed,
        missing_sample_uuids=sorted(set(missing)),
        duplicate_identity_keys=sorted(set(duplicates)),
        identity_warning_count=warnings,
    )


def execute_phase_repeats(
    *,
    agents: list[str],
    repeats: int,
    verify_telemetry: bool,
    fail_on_incomplete_telemetry: bool,
    run_single_repeat: Callable[[str, int], list[EvalLog]],
    assess_repeat_coverage: Callable[[list[EvalLog], str, int], TelemetryCoverageReport],
    on_incomplete_telemetry: Callable[[str, int, TelemetryCoverageReport], Exception]
    | None = None,
) -> list[PhaseRepeatExecution]:
    """Execute all agent/repeat runs with shared telemetry enforcement."""
    executions: list[PhaseRepeatExecution] = []
    for agent in agents:
        for repeat_id in range(repeats):
            logs = run_single_repeat(agent, repeat_id)
            coverage = assess_repeat_coverage(logs, agent, repeat_id)
            if verify_telemetry and fail_on_incomplete_telemetry and not coverage.complete:
                if on_incomplete_telemetry is not None:
                    raise on_incomplete_telemetry(agent, repeat_id, coverage)
                raise RuntimeError(
                    "incomplete reliability telemetry "
                    f"(agent={agent}, repeat_id={repeat_id})"
                )
            executions.append(
                PhaseRepeatExecution(
                    agent=agent,
                    repeat_id=repeat_id,
                    logs=logs,
                    coverage=coverage,
                )
            )
    return executions


def summarize_phase_executions(
    executions: list[PhaseRepeatExecution],
) -> list[dict[str, Any]]:
    """Normalize repeat execution rows for phase result models."""
    rows: list[dict[str, Any]] = []
    for execution in executions:
        coverage = execution.coverage
        rows.append(
            {
                "agent": execution.agent,
                "repeat_id": execution.repeat_id,
                "run_ids": [log.eval.run_id for log in execution.logs],
                "log_paths": [log.location for log in execution.logs if log.location],
                "coverage_complete": coverage.complete,
                "missing_sample_uuids": coverage.missing_sample_uuids,
                "duplicate_identity_keys": coverage.duplicate_identity_keys,
                "identity_warning_count": coverage.identity_warning_count,
            }
        )
    return rows
