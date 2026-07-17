"""Baseline reliability phase execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from inspect_ai import eval
from inspect_ai.agent import as_solver, is_agent
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pydantic import BaseModel, Field, field_validator

from .concurrency import validate_orchestrator_policy
from .posthoc import apply_posthoc_repair
from .spec import ReliabilitySpec

RELIABILITY_CODEX_CLI_VERSION = "0.142.4"
RELIABILITY_CLAUDE_CODE_VERSION = "2.1.181"
RELIABILITY_OPENCODE_VERSION = "1.18.2"


def _opencode_model_for(model: str | None) -> str | None:
    """Pick the ``opencode_model`` (provider client) for a bridge model.

    OpenCode uses ``opencode_model`` only to choose which provider client formats
    the request; the actual generation still goes through the Inspect bridge to
    ``model``. Matching the provider keeps request formatting aligned with the
    underlying model (e.g. openai/* vs anthropic/*).
    """
    if model and "/" in model:
        return model
    return None


def _opencode_reliability_solver(model: str | None, *, filter: Any | None = None) -> Any:
    """Construct the pinned opencode agent used across reliability phases.

    ``filter`` is the bridged-model filter used by the fault phase to inject
    faults into opencode's model traffic; baseline/structural pass ``None``.
    """
    from inspect_swe import opencode

    kwargs: dict[str, Any] = {"version": RELIABILITY_OPENCODE_VERSION}
    opencode_model = _opencode_model_for(model)
    if opencode_model is not None:
        kwargs["opencode_model"] = opencode_model
    if filter is not None:
        kwargs["filter"] = filter
    return opencode(**kwargs)


class BaselineExecutionError(RuntimeError):
    """Raised when reliability execution violates preflight constraints."""


def assert_canonical_eval_log_path(path: str | Path) -> None:
    """Validate that a log path points to Inspect native `.eval` logs."""
    if Path(path).suffix != ".eval":
        raise ValueError(
            f"expected Inspect .eval log path, received: {path!s}. "
            "Reliability execution truth must come from `.eval` logs."
        )


def preflight_reliability_spec(spec: ReliabilitySpec) -> None:
    """Fail-fast validation for reliability runs."""
    validate_orchestrator_policy(spec.concurrency)


class BaselinePhaseConfig(BaseModel):
    """Options for baseline phase execution."""

    repeats: int = Field(default=5, ge=1)
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


class BaselineRepeatResult(BaseModel):
    """One baseline repeat execution summary."""

    agent: str
    repeat_id: int
    run_ids: list[str]
    log_paths: list[str]


class BaselinePhaseResult(BaseModel):
    """Complete baseline phase execution summary."""

    benchmark: str
    repeats: int
    campaign_id: str
    results: list[BaselineRepeatResult]


def run_baseline_phase(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: BaselinePhaseConfig | None = None,
) -> BaselinePhaseResult:
    """Run K independent baseline repeats for all agents in the spec."""
    config = config or BaselinePhaseConfig()
    if "baseline" not in spec.phases:
        raise BaselineExecutionError(
            "ReliabilitySpec does not include baseline phase; refusing baseline run."
        )

    campaign_id = config.campaign_id or _default_campaign_id()
    preflight_reliability_spec(spec)

    repeat_results: list[BaselineRepeatResult] = []
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
            repeat_results.append(
                BaselineRepeatResult(
                    agent=agent,
                    repeat_id=repeat_id,
                    run_ids=[log.eval.run_id for log in logs],
                    log_paths=[log.location for log in logs if log.location],
                )
            )

    return BaselinePhaseResult(
        benchmark=spec.benchmark,
        repeats=config.repeats,
        campaign_id=campaign_id,
        results=repeat_results,
    )


def _run_single_repeat(
    *,
    spec: ReliabilitySpec,
    tasks: Any,
    config: BaselinePhaseConfig,
    campaign_id: str,
    agent: str,
    repeat_id: int,
) -> list[EvalLog]:
    repeat_log_dir = (
        Path(config.log_root)
        / _benchmark_log_slug(spec.benchmark)
        / "baseline"
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
            "reliability_phase": "baseline",
            "reliability_campaign_id": campaign_id,
            "reliability_repeat_id": repeat_id,
            "reliability_agent_attempt_id": 0,
            "reliability_agent": agent,
            "reliability_benchmark": spec.benchmark,
            "reliability_seed": spec.seed,
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
    solver_value = config.solver or _default_solver_for_agent_with_config(agent, config)
    solver_value = apply_posthoc_repair(
        solver_value,
        agent=agent,
        benchmark=spec.benchmark,
        enabled=config.posthoc_repair,
    )
    if config.compute_confidence:
        solver_value = _wrap_solver_with_confidence(solver_value)
    if solver_value is not None:
        eval_kwargs["solver"] = solver_value
    if spec.concurrency.max_connections is not None:
        eval_kwargs["max_connections"] = spec.concurrency.max_connections

    logs = eval(**{key: value for key, value in eval_kwargs.items() if value is not None})
    for log in logs:
        if log.location:
            assert_canonical_eval_log_path(log.location)
    return logs


def _default_solver_for_agent(agent: str) -> Any | None:
    if agent == "codex_cli":
        from inspect_swe import codex_cli

        return codex_cli(version=RELIABILITY_CODEX_CLI_VERSION)
    if agent == "claude_code":
        from inspect_swe import claude_code

        return claude_code(version=RELIABILITY_CLAUDE_CODE_VERSION)
    if agent == "gemini_cli":
        from inspect_swe import gemini_cli

        return gemini_cli()
    if agent == "mini_swe_agent":
        from inspect_swe import mini_swe_agent

        return mini_swe_agent()
    if agent == "opencode":
        return _opencode_reliability_solver(None)
    return None


def _default_solver_for_agent_with_config(
    agent: str,
    config: BaselinePhaseConfig,
) -> Any | None:
    if agent == "codex_cli" and config.taubench_codex_adapter:
        from .taubench import tau2_airline_codex_solver

        return tau2_airline_codex_solver(
            message_limit=config.taubench_message_limit,
        )
    if agent == "opencode":
        return _opencode_reliability_solver(config.model)
    return _default_solver_for_agent(agent)


def _wrap_solver_with_confidence(base_solver: Any | None) -> Solver | Any | None:
    if base_solver is None:
        return None
    wrapped = as_solver(base_solver) if is_agent(base_solver) else base_solver

    @solver(name="baseline_confidence_solver")
    def confidence_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state = await wrapped(state, generate)
            confidence = await _compute_confidence_with_same_model(state)
            if confidence is not None:
                metadata = dict(getattr(state, "metadata", {}) or {})
                metadata["reliability_confidence"] = confidence
                metadata["reliability_confidence_source"] = "same_model_followup"
                state.metadata = metadata
            return state

        return solve

    return confidence_solver()


async def _compute_confidence_with_same_model(state: TaskState) -> float | None:
    messages = list(getattr(state, "messages", []) or [])
    if not messages:
        return None
    prompt = ChatMessageUser(
        content=(
            "Review the full conversation above, including tool use and intermediate steps.\n"
            "Estimate confidence that the final submitted answer is correct.\n"
            "Return only one number between 0 and 100.\n\n"
            "Confidence (0-100):"
        )
    )
    output = await get_model().generate(messages + [prompt])
    return _parse_confidence_value(getattr(output, "completion", None))


def _parse_confidence_value(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    token = value.strip().split()[0].strip().rstrip("%") if value.strip() else ""
    try:
        parsed = float(token)
    except ValueError:
        return None
    if parsed < 0 or parsed > 100:
        return None
    return parsed


def _default_campaign_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _benchmark_log_slug(benchmark: str) -> str:
    """Return a relative log path slug for registry refs and file task refs."""
    cleaned = benchmark.strip()
    if "@" in cleaned:
        path_part, task_part = cleaned.rsplit("@", 1)
        cleaned = f"{Path(path_part).stem}@{task_part}"
    return cleaned.lstrip("/").replace(":", "_") or "benchmark"
