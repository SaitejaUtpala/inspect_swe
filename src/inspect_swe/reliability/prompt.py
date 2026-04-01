"""Prompt reliability phase execution scaffolding."""

from __future__ import annotations

from typing import Any

from inspect_ai import eval
from inspect_ai.log import EvalLog
from pydantic import BaseModel, Field, field_validator

from .artifacts import write_campaign_json
from .orchestrator import preflight_reliability_spec
from .paths import repeat_log_dir
from .perturbations import PromptPerturbationSpec
from .phase_core import (
    assess_repeat_coverage_for_phase,
    build_phase_metadata,
    configure_hooks_for_phase,
    default_campaign_id,
    resolve_phase_sidecar_path,
    run_eval_for_repeat,
)
from .solver_factory import default_solver_for_agent
from .spec import ReliabilitySpec
from .telemetry import TelemetryCoverageReport


class PromptExecutionError(RuntimeError):
    """Raised when prompt phase execution violates reliability constraints."""


class PromptPhaseConfig(BaseModel):
    """Options for prompt phase execution."""

    repeats: int = Field(default=5, ge=1)
    campaign_id: str | None = None
    log_root: str = "logs/reliability"
    sidecar_path: str | None = None
    model: str | None = None
    solver: Any | None = None
    task_args: dict[str, Any] = Field(default_factory=dict)
    inject_agent_task_arg: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    sandbox: str | None = None
    limit: int | tuple[int, int] | None = None
    sample_id: str | int | list[str] | list[int] | list[str | int] | None = None
    verify_telemetry: bool = True
    fail_on_incomplete_telemetry: bool = True
    configure_hooks: bool = True
    perturbation: PromptPerturbationSpec = Field(default_factory=PromptPerturbationSpec)

    @field_validator("campaign_id")
    @classmethod
    def _validate_campaign_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("campaign_id cannot be empty")
        return cleaned


class PromptRepeatResult(BaseModel):
    """One prompt repeat execution summary."""

    agent: str
    repeat_id: int
    run_ids: list[str]
    log_paths: list[str]
    coverage_complete: bool
    missing_sample_uuids: list[str]
    duplicate_identity_keys: list[str]
    identity_warning_count: int


class PromptPhaseResult(BaseModel):
    """Complete prompt phase execution summary."""

    benchmark: str
    repeats: int
    campaign_id: str
    sidecar_path: str
    campaign_json_path: str | None = None
    perturbation: PromptPerturbationSpec
    results: list[PromptRepeatResult]


def run_prompt_phase(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: PromptPhaseConfig | None = None,
) -> PromptPhaseResult:
    """Run K independent prompt repeats for all agents in the spec."""
    config = config or PromptPhaseConfig()
    if "prompt" not in spec.phases:
        raise PromptExecutionError(
            "ReliabilitySpec does not include prompt phase; refusing prompt run."
        )

    campaign_id = config.campaign_id or default_campaign_id()
    sidecar_path = _resolve_sidecar_path(spec, config, campaign_id)
    configure_hooks_for_phase(
        configure_hooks=config.configure_hooks,
        sidecar_path=sidecar_path,
        strict_identity_tags=spec.strict_identity_tags,
    )

    preflight_reliability_spec(spec)

    repeat_results: list[PromptRepeatResult] = []
    for agent in spec.agents:
        for repeat_id in range(config.repeats):
            logs = _run_single_repeat(
                spec=spec,
                tasks=tasks,
                config=config,
                campaign_id=campaign_id,
                agent=agent,
                repeat_id=repeat_id,
            )

            coverage = _assess_repeat_coverage(
                logs=logs,
                sidecar_path=sidecar_path,
                agent=agent,
                repeat_id=repeat_id,
                phase="prompt",
            )

            if (
                config.verify_telemetry
                and config.fail_on_incomplete_telemetry
                and not coverage.complete
            ):
                raise PromptExecutionError(
                    "Incomplete reliability telemetry detected for prompt repeat "
                    f"(agent={agent}, repeat_id={repeat_id}): "
                    f"missing_sample_uuids={coverage.missing_sample_uuids}, "
                    f"duplicate_identity_keys={coverage.duplicate_identity_keys}"
                )

            repeat_results.append(
                PromptRepeatResult(
                    agent=agent,
                    repeat_id=repeat_id,
                    run_ids=[log.eval.run_id for log in logs],
                    log_paths=[log.location for log in logs if log.location],
                    coverage_complete=coverage.complete,
                    missing_sample_uuids=coverage.missing_sample_uuids,
                    duplicate_identity_keys=coverage.duplicate_identity_keys,
                    identity_warning_count=coverage.identity_warning_count,
                )
            )

    campaign_json_path = write_campaign_json(
        sidecar_path=sidecar_path,
        phase="prompt",
        benchmark=spec.benchmark,
        campaign_id=campaign_id,
    )

    return PromptPhaseResult(
        benchmark=spec.benchmark,
        repeats=config.repeats,
        campaign_id=campaign_id,
        sidecar_path=sidecar_path,
        campaign_json_path=campaign_json_path,
        perturbation=config.perturbation,
        results=repeat_results,
    )


def _resolve_sidecar_path(
    spec: ReliabilitySpec, config: PromptPhaseConfig, campaign_id: str
) -> str:
    return resolve_phase_sidecar_path(
        explicit_sidecar_path=config.sidecar_path,
        log_root=config.log_root,
        benchmark=spec.benchmark,
        phase="prompt",
        campaign_id=campaign_id,
    )


def _run_single_repeat(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: PromptPhaseConfig,
    campaign_id: str,
    agent: str,
    repeat_id: int,
) -> list[EvalLog]:
    run_log_dir = repeat_log_dir(
        log_root=config.log_root,
        benchmark=spec.benchmark,
        agent=agent,
        phase="prompt",
        repeat_id=repeat_id,
        campaign_id=campaign_id,
    )

    run_task_args = dict(config.task_args)
    if config.inject_agent_task_arg:
        run_task_args["agent"] = agent

    run_metadata = build_phase_metadata(
        base_metadata=config.metadata,
        phase="prompt",
        campaign_id=campaign_id,
        repeat_id=repeat_id,
        agent=agent,
        benchmark=spec.benchmark,
        seed=spec.seed,
        perturbation_metadata=config.perturbation.metadata_tags(default_seed=spec.seed),
    )
    repeat_seed = spec.seed + repeat_id
    generate_filter = config.perturbation.build_generate_filter(default_seed=repeat_seed)

    solver = config.solver or _default_solver_for_agent(
        agent,
        generate_filter=generate_filter,
    )
    return run_eval_for_repeat(
        tasks=tasks,
        metadata=run_metadata,
        log_dir=str(run_log_dir),
        task_args=run_task_args,
        model=config.model,
        solver=solver,
        sandbox=config.sandbox,
        limit=config.limit,
        sample_id=config.sample_id,
        max_tasks=spec.concurrency.max_tasks,
        max_samples=spec.concurrency.max_samples,
        max_subprocesses=spec.concurrency.max_subprocesses,
        max_sandboxes=spec.concurrency.max_sandboxes,
        max_connections=spec.concurrency.max_connections,
        eval_fn=eval,
    )


def _default_solver_for_agent(agent: str, *, generate_filter: Any) -> Any | None:
    return default_solver_for_agent(agent, generate_filter=generate_filter)


def _assess_repeat_coverage(
    *,
    logs: list[EvalLog],
    sidecar_path: str,
    agent: str,
    repeat_id: int,
    phase: str,
) -> TelemetryCoverageReport:
    return assess_repeat_coverage_for_phase(
        logs=logs,
        sidecar_path=sidecar_path,
        agent=agent,
        repeat_id=repeat_id,
        phase=phase,
    )
