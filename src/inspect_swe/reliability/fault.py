"""Fault reliability phase execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from inspect_ai import eval
from inspect_ai.agent import as_solver, is_agent
from inspect_ai.log import EvalLog
from pydantic import BaseModel

from .baseline import (
    BaselineExecutionError,
    _default_solver_for_agent,
    _wrap_solver_with_confidence,
    assert_canonical_eval_log_path,
    preflight_reliability_spec,
)
from .faults import FaultContext, FaultEnvironment, FaultPhaseConfig
from .spec import ReliabilitySpec


class FaultRepeatResult(BaseModel):
    """One fault repeat execution summary."""

    agent: str
    repeat_id: int
    run_ids: list[str]
    log_paths: list[str]


class FaultPhaseResult(BaseModel):
    """Complete fault phase execution summary."""

    benchmark: str
    repeats: int
    campaign_id: str
    results: list[FaultRepeatResult]


def run_fault_phase(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: FaultPhaseConfig | None = None,
) -> FaultPhaseResult:
    """Run K independent fault-injected repeats for all agents in the spec."""
    config = config or FaultPhaseConfig()
    if "fault" not in spec.phases:
        raise BaselineExecutionError(
            "ReliabilitySpec does not include fault phase; refusing fault run."
        )

    campaign_id = config.campaign_id or _default_campaign_id()
    preflight_reliability_spec(spec)

    repeat_results: list[FaultRepeatResult] = []
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
            repeat_results.append(
                FaultRepeatResult(
                    agent=agent,
                    repeat_id=repeat_id,
                    run_ids=[log.eval.run_id for log in logs],
                    log_paths=[log.location for log in logs if log.location],
                )
            )

    return FaultPhaseResult(
        benchmark=spec.benchmark,
        repeats=config.repeats,
        campaign_id=campaign_id,
        results=repeat_results,
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
    repeat_log_dir = (
        Path(config.log_root)
        / spec.benchmark
        / "fault"
        / campaign_id
        / agent
        / f"rep_{repeat_id:03d}"
    )

    fault_context = FaultContext(
        campaign_id=campaign_id,
        phase="fault",
        agent=agent,
        repeat_id=repeat_id,
        seed=config.seed or spec.seed,
    )
    fault_env = FaultEnvironment(config.faults, fault_context)

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
            "reliability_fault_specs": [fault.model_dump() for fault in config.faults],
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

    default_solver = _default_solver_for_agent_with_faults(agent, fault_env, config)
    solver = config.solver or default_solver
    if solver is not None:
        solver = as_solver(solver) if is_agent(solver) else solver
        solver = fault_env.wrap_solver(solver)
    if config.compute_confidence:
        solver = _wrap_solver_with_confidence(solver)
    if solver is not None:
        eval_kwargs["solver"] = solver
    if spec.concurrency.max_connections is not None:
        eval_kwargs["max_connections"] = spec.concurrency.max_connections

    eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
    logs = eval(**eval_kwargs)

    for log in logs:
        if not log.location:
            continue
        assert_canonical_eval_log_path(log.location)
    return logs


def _default_solver_for_agent_with_faults(
    agent: str, fault_env: FaultEnvironment, config: FaultPhaseConfig
) -> Any | None:
    if agent == "codex_cli":
        from inspect_swe._codex_cli.codex_cli import codex_cli

        return codex_cli(**_codex_cli_fault_kwargs(fault_env, config))
    if agent == "claude_code":
        from inspect_swe._claude_code.claude_code import claude_code

        return claude_code(filter=fault_env.model_filter(), retry_refusals=3)
    if agent == "gemini_cli":
        from inspect_swe._gemini_cli.gemini_cli import gemini_cli

        return gemini_cli(filter=fault_env.model_filter(), retry_refusals=3)
    if agent == "mini_swe_agent":
        from inspect_swe._mini_swe_agent.mini_swe_agent import mini_swe_agent

        return mini_swe_agent(filter=fault_env.model_filter(), retry_refusals=3)
    return _default_solver_for_agent(agent)


def _codex_cli_fault_kwargs(
    fault_env: FaultEnvironment, config: FaultPhaseConfig
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "version": "0.118.0",
        "filter": fault_env.model_filter(),
        "retry_refusals": 3,
    }
    if config.replace_native_web_search:
        from .search_tool import reliability_search_bridged_tools

        kwargs["disallowed_tools"] = ["web_search"]
        kwargs["bridged_tools"] = fault_env.wrap_bridged_tools(
            reliability_search_bridged_tools()
        )
    return kwargs


def _default_campaign_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{timestamp}_{suffix}"
