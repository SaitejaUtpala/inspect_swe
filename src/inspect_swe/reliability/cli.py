"""Command-line interface for reliability phases."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .baseline import BaselinePhaseConfig, run_baseline_phase
from .concurrency import OrchestratorConcurrency
from .fault import run_fault_phase
from .faults import FaultPhaseConfig, FaultSpec
from .spec import ReliabilitySpec
from .structural import StructuralPhaseConfig, run_structural_phase


def main(argv: list[str] | None = None) -> int:
    """Run the reliability CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "campaign":
        return _run_campaign(args)
    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspect-swe-reliability")
    subparsers = parser.add_subparsers(dest="command")
    campaign = subparsers.add_parser(
        "campaign", help="run baseline, fault, and/or structural phases"
    )
    campaign.add_argument("--benchmark", required=True, help="Inspect task name or path")
    campaign.add_argument("--agent", action="append", default=None, help="agent name")
    campaign.add_argument(
        "--phase",
        action="append",
        choices=["baseline", "fault", "structural"],
        default=None,
        help="phase to run; can be supplied multiple times",
    )
    campaign.add_argument("--model", default=None, help="Inspect model name")
    campaign.add_argument("--repeats", type=int, default=1)
    campaign.add_argument("--limit", type=int, default=None)
    campaign.add_argument("--max-samples", type=int, default=None)
    campaign.add_argument("--sample-id", action="append", default=None)
    campaign.add_argument("--sandbox", default=None)
    campaign.add_argument("--log-root", default="logs/reliability")
    campaign.add_argument("--campaign-id", default=None)
    campaign.add_argument("--seed", type=int, default=0)
    campaign.add_argument("--task-arg", action="append", default=[])
    campaign.add_argument("--inject-agent-task-arg", action="store_true")
    campaign.add_argument("--no-compute-confidence", action="store_true")
    campaign.add_argument(
        "--no-posthoc-repair",
        action="store_true",
        help=(
            "disable dataset-specific post-hoc answer repair (e.g. the "
            "Claude Code GAIA answer-only reformatting pass)"
        ),
    )
    campaign.add_argument(
        "--fault-surface",
        default="message",
        choices=["model", "message", "tool"],
    )
    campaign.add_argument(
        "--fault-mode",
        default="exec_observation_error",
        help=(
            "fault mode, e.g. api_error, rate_limit, refuse, truncate, corrupt, "
            "mislead, delay, append_instruction, exec_observation_error, "
            "observation_mislead, tool_error, tool_malformed, tool_empty"
        ),
    )
    campaign.add_argument("--fault-probability", type=float, default=0.2)
    campaign.add_argument("--fault-severity", type=float, default=1.0)
    campaign.add_argument("--fault-target", default=None)
    campaign.add_argument(
        "--structural-strength",
        default="medium",
        choices=["mild", "medium", "severe"],
    )
    campaign.add_argument(
        "--structural-kind",
        default="gaia",
        choices=["gaia", "taubench"],
    )
    campaign.add_argument("--no-structural-baseline", action="store_true")
    campaign.add_argument(
        "--taubench-codex-adapter",
        action="store_true",
        help="run Tau2 Airline with Codex as the service agent",
    )
    campaign.add_argument(
        "--taubench-message-limit",
        type=int,
        default=None,
        help="maximum user/service-agent turns for the TauBench Codex adapter",
    )
    return parser


def _run_campaign(args: argparse.Namespace) -> int:
    agents = args.agent or ["codex_cli"]
    phases = args.phase or ["baseline", "fault"]
    spec = ReliabilitySpec(
        benchmark=args.benchmark,
        agents=agents,
        phases=phases,
        seed=args.seed,
        concurrency=OrchestratorConcurrency(max_samples=args.max_samples),
    )
    task_args = _parse_task_args(args.task_arg)
    shared: dict[str, Any] = {
        "repeats": args.repeats,
        "campaign_id": args.campaign_id,
        "log_root": args.log_root,
        "model": args.model,
        "task_args": task_args,
        "inject_agent_task_arg": args.inject_agent_task_arg,
        "sandbox": args.sandbox,
        "limit": args.limit,
        "sample_id": _sample_id_arg(args.sample_id),
        "compute_confidence": not args.no_compute_confidence,
        "posthoc_repair": not args.no_posthoc_repair,
    }

    output: dict[str, Any] = {}
    if "baseline" in phases:
        result = run_baseline_phase(
            spec=spec,
            tasks=args.benchmark,
            config=BaselinePhaseConfig(
                **shared,
                taubench_codex_adapter=args.taubench_codex_adapter,
                taubench_message_limit=args.taubench_message_limit,
            ),
        )
        output["baseline"] = result.model_dump()
    if "fault" in phases:
        fault = FaultSpec(
            surface=args.fault_surface,
            mode=args.fault_mode,
            probability=args.fault_probability,
            severity=args.fault_severity,
            target=args.fault_target,
        )
        result = run_fault_phase(
            spec=spec,
            tasks=args.benchmark,
            config=FaultPhaseConfig(
                **shared,
                seed=args.seed,
                faults=[fault],
            ),
        )
        output["fault"] = result.model_dump()
    if "structural" in phases:
        result = run_structural_phase(
            spec=spec,
            tasks=args.benchmark,
            config=StructuralPhaseConfig(
                **shared,
                seed=args.seed,
                strength=args.structural_strength,
                kind=args.structural_kind,
                include_baseline=not args.no_structural_baseline,
                taubench_codex_adapter=args.taubench_codex_adapter,
                taubench_message_limit=args.taubench_message_limit,
            ),
        )
        output["structural"] = result.model_dump()

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _parse_task_args(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--task-arg must be KEY=VALUE, got {value!r}")
        key, raw_value = value.split("=", 1)
        parsed[key] = _coerce_value(raw_value)
    return parsed


def _coerce_value(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _sample_id_arg(values: list[str] | None) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return _coerce_value(values[0])
    return [_coerce_value(value) for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
