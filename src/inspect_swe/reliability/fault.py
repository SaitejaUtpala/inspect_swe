"""Fault reliability phase execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from inspect_ai import eval
from inspect_ai.agent import as_solver
from inspect_ai.log import EvalLog
from pydantic import BaseModel

from .baseline import (
    RELIABILITY_CLAUDE_CODE_VERSION,
    RELIABILITY_CODEX_CLI_VERSION,
    _benchmark_log_slug,
    _wrap_solver_with_confidence,
    assert_canonical_eval_log_path,
)
from .concurrency import validate_orchestrator_policy
from .faults import FaultContext, FaultEnvironment, FaultPhaseConfig
from .posthoc import apply_posthoc_repair
from .spec import ReliabilitySpec


class FaultPhaseResult(BaseModel):
    """Complete fault phase execution summary."""

    benchmark: str
    repeats: int
    campaign_id: str
    results: list[dict[str, Any]]


def run_fault_phase(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: FaultPhaseConfig | None = None,
) -> FaultPhaseResult:
    """Run reliability fault injection repeats for all agents in the spec."""
    config = config or FaultPhaseConfig()
    if "fault" not in spec.phases:
        raise RuntimeError("ReliabilitySpec does not include fault phase.")

    campaign_id = config.campaign_id or _default_campaign_id()
    validate_orchestrator_policy(spec.concurrency)

    results: list[dict[str, Any]] = []
    for agent in spec.agents:
        for repeat_id in range(config.repeats):
            logs = _run_single_fault_repeat(
                spec=spec,
                tasks=tasks,
                config=config,
                campaign_id=campaign_id,
                agent=agent,
                repeat_id=repeat_id,
            )
            results.append(
                {
                    "agent": agent,
                    "repeat_id": repeat_id,
                    "run_ids": [log.eval.run_id for log in logs],
                    "log_paths": [log.location for log in logs if log.location],
                }
            )

    return FaultPhaseResult(
        benchmark=spec.benchmark,
        repeats=config.repeats,
        campaign_id=campaign_id,
        results=results,
    )


def _run_single_fault_repeat(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: FaultPhaseConfig,
    campaign_id: str,
    agent: str,
    repeat_id: int,
) -> list[EvalLog]:
    fault_env = FaultEnvironment(
        config.faults,
        FaultContext(
            campaign_id=campaign_id,
            phase="fault",
            agent=agent,
            repeat_id=repeat_id,
            seed=config.seed or spec.seed,
        ),
    )
    solver_value = config.solver or _default_fault_solver_for_agent(agent, fault_env, config)
    if solver_value is not None:
        solver_value = fault_env.wrap_solver(as_solver(solver_value))
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
        / "fault"
        / campaign_id
        / agent
        / f"rep_{repeat_id:03d}"
    )
    run_task_args = dict(config.task_args)
    if config.inject_agent_task_arg:
        run_task_args["agent"] = agent

    run_metadata = dict(config.metadata)
    run_metadata.update(
        {
            "reliability_phase": "fault",
            "reliability_campaign_id": campaign_id,
            "reliability_repeat_id": repeat_id,
            "reliability_agent_attempt_id": 0,
            "reliability_agent": agent,
            "reliability_benchmark": spec.benchmark,
            "reliability_seed": spec.seed,
            "reliability_faults": [fault.model_dump() for fault in config.faults],
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


def _default_fault_solver_for_agent(
    agent: str, fault_env: FaultEnvironment, config: FaultPhaseConfig
) -> Any | None:
    if agent == "codex_cli":
        from inspect_swe import codex_cli

        return codex_cli(**_codex_cli_fault_kwargs(fault_env, config))
    if agent == "claude_code":
        from inspect_swe import claude_code

        return claude_code(
            filter=fault_env.model_filter(),
            version=RELIABILITY_CLAUDE_CODE_VERSION,
        )
    if agent == "gemini_cli":
        from inspect_swe import gemini_cli

        return gemini_cli()
    if agent == "mini_swe_agent":
        from inspect_swe import mini_swe_agent

        return mini_swe_agent()
    if agent == "opencode":
        from inspect_swe import opencode

        return opencode()
    return None


def _codex_cli_fault_kwargs(
    fault_env: FaultEnvironment, config: FaultPhaseConfig
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "filter": fault_env.model_filter(),
        "retry_refusals": 3,
        "version": RELIABILITY_CODEX_CLI_VERSION,
    }
    return kwargs


def _default_campaign_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"
