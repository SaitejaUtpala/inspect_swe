"""Structural robustness phase execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from inspect_ai import eval
from inspect_ai.agent import as_solver, is_agent
from inspect_ai.log import EvalLog
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .baseline import (
    RELIABILITY_CODEX_CLI_VERSION,
    _benchmark_log_slug,
    _default_solver_for_agent,
    _opencode_reliability_solver,
    _wrap_solver_with_confidence,
    assert_canonical_eval_log_path,
)
from .concurrency import validate_orchestrator_policy
from .posthoc import apply_posthoc_repair
from .spec import ReliabilitySpec
from .structural_perturbations import (
    StructuralContext,
    StructuralEnvironment,
    StructuralKind,
    StructuralStrength,
)


class StructuralPhaseConfig(BaseModel):
    """Options for structural robustness execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repeats: int = Field(default=1, ge=1)
    campaign_id: str | None = None
    log_root: str = "logs/reliability"
    model: str | None = None
    solver: Any | None = None
    task_args: dict[str, Any] = Field(default_factory=dict)
    inject_agent_task_arg: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    sandbox: str | None = None
    limit: int | tuple[int, int] | None = None
    sample_id: str | int | list[str] | list[int] | list[str | int] | None = None
    compute_confidence: bool = True
    posthoc_repair: bool = True
    seed: int = 0
    strength: StructuralStrength = "medium"
    kind: StructuralKind = "gaia"
    include_baseline: bool = True
    bridged_tools: Any | None = None
    taubench_codex_adapter: bool = False
    taubench_message_limit: int | None = None

    @field_validator("campaign_id")
    @classmethod
    def _validate_campaign_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("campaign_id cannot be empty")
        return cleaned


class StructuralRepeatResult(BaseModel):
    """One structural repeat summary."""

    agent: str
    repeat_id: int
    baseline_log_paths: list[str] = Field(default_factory=list)
    perturbed_log_paths: list[str] = Field(default_factory=list)
    baseline_accuracy: float | None = None
    perturbed_accuracy: float | None = None
    r_struct: float | None = None


class StructuralPhaseResult(BaseModel):
    """Complete structural phase summary."""

    benchmark: str
    repeats: int
    campaign_id: str
    strength: StructuralStrength
    kind: StructuralKind
    results: list[StructuralRepeatResult]


def run_structural_phase(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: StructuralPhaseConfig | None = None,
) -> StructuralPhaseResult:
    """Run paired baseline/perturbed structural robustness evals."""
    config = config or StructuralPhaseConfig()
    if "structural" not in spec.phases:
        raise RuntimeError("ReliabilitySpec does not include structural phase.")

    campaign_id = config.campaign_id or _default_campaign_id()
    validate_orchestrator_policy(spec.concurrency)

    results: list[StructuralRepeatResult] = []
    for agent in spec.agents:
        for repeat_id in range(config.repeats):
            baseline_logs: list[EvalLog] = []
            if config.include_baseline:
                baseline_logs = _run_single_structural_eval(
                    spec=spec,
                    tasks=tasks,
                    config=config,
                    campaign_id=campaign_id,
                    agent=agent,
                    repeat_id=repeat_id,
                    variant="baseline",
                )
            perturbed_logs = _run_single_structural_eval(
                spec=spec,
                tasks=tasks,
                config=config,
                campaign_id=campaign_id,
                agent=agent,
                repeat_id=repeat_id,
                variant="perturbed",
            )
            baseline_accuracy = _accuracy_from_logs(baseline_logs)
            perturbed_accuracy = _accuracy_from_logs(perturbed_logs)
            results.append(
                StructuralRepeatResult(
                    agent=agent,
                    repeat_id=repeat_id,
                    baseline_log_paths=[log.location for log in baseline_logs if log.location],
                    perturbed_log_paths=[
                        log.location for log in perturbed_logs if log.location
                    ],
                    baseline_accuracy=baseline_accuracy,
                    perturbed_accuracy=perturbed_accuracy,
                    r_struct=_structural_ratio(
                        baseline_accuracy,
                        perturbed_accuracy,
                    ),
                )
            )

    return StructuralPhaseResult(
        benchmark=spec.benchmark,
        repeats=config.repeats,
        campaign_id=campaign_id,
        strength=config.strength,
        kind=config.kind,
        results=results,
    )


def _run_single_structural_eval(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: StructuralPhaseConfig,
    campaign_id: str,
    agent: str,
    repeat_id: int,
    variant: str,
) -> list[EvalLog]:
    env = StructuralEnvironment(
        kind=config.kind,
        strength=config.strength,
        context=StructuralContext(
            campaign_id=campaign_id,
            phase=f"structural_{variant}",
            agent=agent,
            repeat_id=repeat_id,
            seed=config.seed or spec.seed,
        ),
    )
    solver_value = config.solver or _default_structural_solver_for_agent(
        agent,
        env,
        config,
        perturbed=variant == "perturbed",
    )
    if variant == "perturbed" and solver_value is not None:
        if is_agent(solver_value):
            solver_value = as_solver(solver_value)
        solver_value = env.wrap_solver(solver_value)
    solver_value = apply_posthoc_repair(
        solver_value,
        agent=agent,
        benchmark=spec.benchmark,
        enabled=config.posthoc_repair,
    )
    if config.compute_confidence:
        solver_value = _wrap_solver_with_confidence(solver_value)

    repeat_log_dir = (
        Path(config.log_root)
        / _benchmark_log_slug(spec.benchmark)
        / "structural"
        / campaign_id
        / agent
        / f"rep_{repeat_id:03d}"
        / variant
    )
    run_task_args = dict(config.task_args)
    if config.inject_agent_task_arg:
        run_task_args["agent"] = agent

    run_metadata = dict(config.metadata)
    run_metadata.update(
        {
            "reliability_phase": f"structural_{variant}",
            "reliability_campaign_id": campaign_id,
            "reliability_repeat_id": repeat_id,
            "reliability_agent_attempt_id": 0,
            "reliability_agent": agent,
            "reliability_benchmark": spec.benchmark,
            "reliability_seed": spec.seed,
            "reliability_structural_kind": config.kind,
            "reliability_structural_strength": config.strength,
        }
    )

    eval_kwargs: dict[str, Any] = {
        "tasks": tasks,
        "metadata": run_metadata,
        "log_dir": str(repeat_log_dir),
        "log_format": "eval",
        "score": True,
        "sample_shuffle": False,
        "max_tasks": spec.concurrency.max_tasks,
        "max_samples": spec.concurrency.max_samples,
        "max_subprocesses": spec.concurrency.max_subprocesses,
        "max_sandboxes": spec.concurrency.max_sandboxes,
        "sandbox": config.sandbox,
        "limit": config.limit,
        "sample_id": config.sample_id,
    }
    if run_task_args:
        eval_kwargs["task_args"] = run_task_args
    if config.model is not None:
        eval_kwargs["model"] = config.model
    if solver_value is not None:
        eval_kwargs["solver"] = solver_value
    if spec.concurrency.max_connections is not None:
        eval_kwargs["max_connections"] = spec.concurrency.max_connections

    logs = eval(**{key: value for key, value in eval_kwargs.items() if value is not None})
    for log in logs:
        if log.location:
            assert_canonical_eval_log_path(log.location)
    return logs


def _default_structural_solver_for_agent(
    agent: str,
    env: StructuralEnvironment,
    config: StructuralPhaseConfig,
    *,
    perturbed: bool,
) -> Any | None:
    if not perturbed:
        if agent == "codex_cli" and config.taubench_codex_adapter:
            from .taubench import tau2_airline_codex_solver

            return tau2_airline_codex_solver(
                message_limit=config.taubench_message_limit,
            )
        if agent == "opencode":
            solver_value = _opencode_reliability_solver(config.model)
            return as_solver(solver_value) if is_agent(solver_value) else solver_value
        return _default_solver_for_agent(agent)
    if agent == "codex_cli":
        if config.taubench_codex_adapter:
            from .taubench import tau2_airline_codex_solver

            return tau2_airline_codex_solver(
                structural_env=env,
                message_limit=config.taubench_message_limit,
            )
        from inspect_swe import codex_cli

        return as_solver(
            codex_cli(
                filter=env.model_filter(),
                bridged_tools=env.wrap_bridged_tools(config.bridged_tools),
                retry_refusals=3,
                version=RELIABILITY_CODEX_CLI_VERSION,
            )
        )
    if agent == "opencode":
        solver_value = _opencode_reliability_solver(config.model)
        return as_solver(solver_value) if is_agent(solver_value) else solver_value
    solver_value = _default_solver_for_agent(agent)
    return as_solver(solver_value) if is_agent(solver_value) else solver_value


def _accuracy_from_logs(logs: list[EvalLog]) -> float | None:
    values: list[float] = []
    for log in logs:
        for sample in log.samples or []:
            parsed = _sample_success(getattr(sample, "scores", {}) or {})
            if parsed is not None:
                values.append(parsed)
    if not values:
        return None
    return sum(values) / len(values)


def _sample_success(scores: dict[str, Any]) -> float | None:
    for score in scores.values():
        value = getattr(score, "value", score)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if float(value) >= 0.5 else 0.0
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"c", "correct", "pass", "passed", "true", "1", "yes"}:
                return 1.0
            if token in {"i", "incorrect", "fail", "failed", "false", "0", "no"}:
                return 0.0
    return None


def _structural_ratio(
    baseline_accuracy: float | None,
    perturbed_accuracy: float | None,
) -> float | None:
    if baseline_accuracy is None or perturbed_accuracy is None or baseline_accuracy <= 1e-8:
        return None
    return min(perturbed_accuracy / baseline_accuracy, 1.0)


def _default_campaign_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"
