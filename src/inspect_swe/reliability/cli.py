"""Command-line interface for Inspect SWE reliability phases."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Sequence, cast

from .analysis import (
    BaselineAnalysisResult,
    CampaignAnalysisResult,
    analyze_baseline_campaign,
    analyze_reliability_campaign,
)
from .baseline import (
    BaselineExecutionError,
    BaselinePhaseConfig,
    BaselinePhaseResult,
    run_baseline_phase,
)
from .concurrency import OrchestratorConcurrency
from .fault import (
    FaultExecutionError,
    FaultPhaseConfig,
    FaultPhaseResult,
    run_fault_phase,
)
from .orchestrator import (
    ReliabilityCampaignConfig,
    ReliabilityCampaignResult,
    run_reliability_campaign,
)
from .paths import campaign_sidecar_path
from .perturbations import (
    FaultPerturbationSpec,
    PromptPerturbationSpec,
    StructuralPerturbationSpec,
)
from .prompt import (
    PromptExecutionError,
    PromptPhaseConfig,
    PromptPhaseResult,
    run_prompt_phase,
)
from .spec import ReliabilitySpec
from .structural import (
    StructuralExecutionError,
    StructuralPhaseConfig,
    StructuralPhaseResult,
    run_structural_phase,
)


def build_parser() -> argparse.ArgumentParser:
    """Build an argument parser for reliability commands."""
    parser = argparse.ArgumentParser(
        prog="inspect-swe-reliability",
        description="Run reliability phases for inspect_swe.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "baseline",
        help="Run baseline reliability phase.",
        description=(
            "Run K independent baseline repeats and emit sidecar telemetry "
            "alongside canonical Inspect `.eval` logs."
        ),
    )
    baseline.add_argument("--benchmark", required=True, help="Benchmark label.")
    baseline.add_argument(
        "--tasks",
        required=True,
        help="Inspect task path (e.g. inspect_evals/gaia_level1).",
    )
    baseline.add_argument(
        "--agent",
        dest="agents",
        action="append",
        required=True,
        help="Agent identifier (repeat for multi-agent runs).",
    )
    baseline.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent baseline repeats per agent (default: 5).",
    )
    baseline.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Campaign seed recorded in reliability metadata.",
    )
    baseline.add_argument(
        "--model",
        default=None,
        help="Optional model override passed to inspect eval.",
    )
    baseline.add_argument(
        "--task-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task arg pair (repeatable).",
    )
    baseline.add_argument(
        "--inject-agent-task-arg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject task arg `agent=<agent>` for each run.",
    )
    baseline.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional metadata pair (repeatable).",
    )
    baseline.add_argument(
        "--sandbox",
        default=None,
        help="Sandbox setting passed to inspect eval.",
    )
    baseline.add_argument(
        "--limit",
        default=None,
        help="Sample limit (e.g. 3 or 10-20).",
    )
    baseline.add_argument(
        "--sample-id",
        default=None,
        help="Sample id(s): one value or comma-separated list.",
    )
    baseline.add_argument(
        "--campaign-id",
        default=None,
        help=(
            "Campaign identifier used to segregate log/sidecar paths. "
            "Defaults to an auto-generated timestamped id."
        ),
    )
    baseline.add_argument(
        "--log-root",
        default="logs/reliability",
        help="Root directory for eval logs and default sidecars.",
    )
    baseline.add_argument(
        "--sidecar-dir",
        default="reliability_sidecars",
        help="Relative sidecar directory under log root.",
    )
    baseline.add_argument(
        "--sidecar-path",
        default=None,
        help="Explicit sidecar JSONL path (overrides sidecar-dir).",
    )
    baseline.add_argument(
        "--strict-identity-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reliability identity metadata tags.",
    )
    baseline.add_argument(
        "--fail-on-missing-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail preflight when reliability hooks are not active.",
    )
    baseline.add_argument(
        "--verify-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate sidecar coverage against eval logs.",
    )
    baseline.add_argument(
        "--fail-on-incomplete-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail run when telemetry coverage is incomplete.",
    )
    baseline.add_argument(
        "--configure-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Configure reliability hooks automatically.",
    )
    baseline.add_argument(
        "--orchestrator-mode",
        choices=("single_process", "multi_process"),
        default="multi_process",
        help="Orchestrator policy mode.",
    )
    baseline.add_argument(
        "--orchestrator-workers",
        type=int,
        default=1,
        help="Orchestrator worker count.",
    )
    baseline.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Inspect max_tasks setting.",
    )
    baseline.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Inspect max_samples setting.",
    )
    baseline.add_argument(
        "--max-subprocesses",
        type=int,
        default=None,
        help="Inspect max_subprocesses setting.",
    )
    baseline.add_argument(
        "--max-sandboxes",
        type=int,
        default=None,
        help="Inspect max_sandboxes setting.",
    )
    baseline.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Inspect max_connections setting.",
    )
    baseline.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    baseline.set_defaults(handler=_handle_baseline_command)

    fault = subparsers.add_parser(
        "fault",
        help="Run fault perturbation reliability phase.",
        description=(
            "Run K independent fault-phase repeats and emit sidecar telemetry. "
            "Supports deterministic synthetic model/tool fault injection."
        ),
    )
    fault.add_argument("--benchmark", required=True, help="Benchmark label.")
    fault.add_argument(
        "--tasks",
        required=True,
        help="Inspect task path (e.g. inspect_evals/gaia_level1).",
    )
    fault.add_argument(
        "--agent",
        dest="agents",
        action="append",
        required=True,
        help="Agent identifier (repeat for multi-agent runs).",
    )
    fault.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent fault repeats per agent (default: 5).",
    )
    fault.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Campaign seed recorded in reliability metadata.",
    )
    fault.add_argument(
        "--model",
        default=None,
        help="Optional model override passed to inspect eval.",
    )
    fault.add_argument(
        "--task-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task arg pair (repeatable).",
    )
    fault.add_argument(
        "--inject-agent-task-arg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject task arg `agent=<agent>` for each run.",
    )
    fault.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional metadata pair (repeatable).",
    )
    fault.add_argument(
        "--sandbox",
        default=None,
        help="Sandbox setting passed to inspect eval.",
    )
    fault.add_argument(
        "--limit",
        default=None,
        help="Sample limit (e.g. 3 or 10-20).",
    )
    fault.add_argument(
        "--sample-id",
        default=None,
        help="Sample id(s): one value or comma-separated list.",
    )
    fault.add_argument(
        "--campaign-id",
        default=None,
        help=(
            "Campaign identifier used to segregate log/sidecar paths. "
            "Defaults to an auto-generated timestamped id."
        ),
    )
    fault.add_argument(
        "--log-root",
        default="logs/reliability",
        help="Root directory for eval logs and default sidecars.",
    )
    fault.add_argument(
        "--sidecar-dir",
        default="reliability_sidecars",
        help="Relative sidecar directory under log root.",
    )
    fault.add_argument(
        "--sidecar-path",
        default=None,
        help="Explicit sidecar JSONL path (overrides sidecar-dir).",
    )
    fault.add_argument(
        "--fault-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fault perturbation adapter.",
    )
    fault.add_argument(
        "--fault-mode",
        choices=("noop", "model_error", "tool_error", "disabled"),
        default="noop",
        help="Fault adapter mode.",
    )
    fault.add_argument(
        "--fault-target",
        choices=("model", "tool"),
        default="model",
        help="Target surface for fault perturbation.",
    )
    fault.add_argument(
        "--fault-rate",
        type=float,
        default=0.0,
        help="Fault injection rate per eligible attempt.",
    )
    fault.add_argument(
        "--fault-max-per-sample",
        type=int,
        default=1,
        help="Max configured faults per sample.",
    )
    fault.add_argument(
        "--fault-seed",
        type=int,
        default=None,
        help="Optional fault adapter seed override.",
    )
    fault.add_argument(
        "--fault-policy-name",
        default="fault_scaffold_v1",
        help="Fault policy name recorded in metadata.",
    )
    fault.add_argument(
        "--strict-identity-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reliability identity metadata tags.",
    )
    fault.add_argument(
        "--fail-on-missing-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail preflight when reliability hooks are not active.",
    )
    fault.add_argument(
        "--verify-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate sidecar coverage against eval logs.",
    )
    fault.add_argument(
        "--fail-on-incomplete-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail run when telemetry coverage is incomplete.",
    )
    fault.add_argument(
        "--configure-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Configure reliability hooks automatically.",
    )
    fault.add_argument(
        "--orchestrator-mode",
        choices=("single_process", "multi_process"),
        default="multi_process",
        help="Orchestrator policy mode.",
    )
    fault.add_argument(
        "--orchestrator-workers",
        type=int,
        default=1,
        help="Orchestrator worker count.",
    )
    fault.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Inspect max_tasks setting.",
    )
    fault.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Inspect max_samples setting.",
    )
    fault.add_argument(
        "--max-subprocesses",
        type=int,
        default=None,
        help="Inspect max_subprocesses setting.",
    )
    fault.add_argument(
        "--max-sandboxes",
        type=int,
        default=None,
        help="Inspect max_sandboxes setting.",
    )
    fault.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Inspect max_connections setting.",
    )
    fault.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    fault.set_defaults(handler=_handle_fault_command)

    prompt = subparsers.add_parser(
        "prompt",
        help="Run prompt perturbation reliability phase.",
        description=(
            "Run K independent prompt-phase repeats and emit sidecar telemetry. "
            "Supports deterministic prompt rewrites via generate filter."
        ),
    )
    prompt.add_argument("--benchmark", required=True, help="Benchmark label.")
    prompt.add_argument(
        "--tasks",
        required=True,
        help="Inspect task path (e.g. inspect_evals/gaia_level1).",
    )
    prompt.add_argument(
        "--agent",
        dest="agents",
        action="append",
        required=True,
        help="Agent identifier (repeat for multi-agent runs).",
    )
    prompt.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent prompt repeats per agent (default: 5).",
    )
    prompt.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Campaign seed recorded in reliability metadata.",
    )
    prompt.add_argument(
        "--model",
        default=None,
        help="Optional model override passed to inspect eval.",
    )
    prompt.add_argument(
        "--task-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task arg pair (repeatable).",
    )
    prompt.add_argument(
        "--inject-agent-task-arg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject task arg `agent=<agent>` for each run.",
    )
    prompt.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional metadata pair (repeatable).",
    )
    prompt.add_argument(
        "--sandbox",
        default=None,
        help="Sandbox setting passed to inspect eval.",
    )
    prompt.add_argument(
        "--limit",
        default=None,
        help="Sample limit (e.g. 3 or 10-20).",
    )
    prompt.add_argument(
        "--sample-id",
        default=None,
        help="Sample id(s): one value or comma-separated list.",
    )
    prompt.add_argument(
        "--campaign-id",
        default=None,
        help=(
            "Campaign identifier used to segregate log/sidecar paths. "
            "Defaults to an auto-generated timestamped id."
        ),
    )
    prompt.add_argument(
        "--log-root",
        default="logs/reliability",
        help="Root directory for eval logs and default sidecars.",
    )
    prompt.add_argument(
        "--sidecar-dir",
        default="reliability_sidecars",
        help="Relative sidecar directory under log root.",
    )
    prompt.add_argument(
        "--sidecar-path",
        default=None,
        help="Explicit sidecar JSONL path (overrides default).",
    )
    prompt.add_argument(
        "--prompt-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prompt perturbation adapter.",
    )
    prompt.add_argument(
        "--prompt-mode",
        choices=("noop", "rewrite_v1", "rewrite_llm_v1", "disabled"),
        default="rewrite_llm_v1",
        help="Prompt adapter mode.",
    )
    prompt.add_argument(
        "--prompt-variant-count",
        type=int,
        default=1,
        help="Configured prompt variant count.",
    )
    prompt.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help="Optional prompt adapter seed override.",
    )
    prompt.add_argument(
        "--prompt-policy-name",
        default="prompt_rewrite_llm_v1",
        help="Prompt policy name recorded in metadata.",
    )
    prompt.add_argument(
        "--prompt-rewrite-model",
        default=None,
        help=(
            "Optional dedicated rewrite model (Inspect model name). "
            "Defaults to the active eval model."
        ),
    )
    prompt.add_argument(
        "--prompt-rewrite-temperature",
        type=float,
        default=0.2,
        help="Temperature for LLM rewrite mode.",
    )
    prompt.add_argument(
        "--prompt-rewrite-max-tokens",
        type=int,
        default=512,
        help="Max tokens for LLM rewrite mode.",
    )
    prompt.add_argument(
        "--prompt-rewrite-strength",
        choices=("mild", "medium", "strong", "naturalistic"),
        default="medium",
        help="Rewrite strength profile for LLM rewrite mode.",
    )
    prompt.add_argument(
        "--strict-identity-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reliability identity metadata tags.",
    )
    prompt.add_argument(
        "--fail-on-missing-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail preflight when reliability hooks are not active.",
    )
    prompt.add_argument(
        "--verify-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate sidecar coverage against eval logs.",
    )
    prompt.add_argument(
        "--fail-on-incomplete-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail run when telemetry coverage is incomplete.",
    )
    prompt.add_argument(
        "--configure-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Configure reliability hooks automatically.",
    )
    prompt.add_argument(
        "--orchestrator-mode",
        choices=("single_process", "multi_process"),
        default="multi_process",
        help="Orchestrator policy mode.",
    )
    prompt.add_argument(
        "--orchestrator-workers",
        type=int,
        default=1,
        help="Orchestrator worker count.",
    )
    prompt.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Inspect max_tasks setting.",
    )
    prompt.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Inspect max_samples setting.",
    )
    prompt.add_argument(
        "--max-subprocesses",
        type=int,
        default=None,
        help="Inspect max_subprocesses setting.",
    )
    prompt.add_argument(
        "--max-sandboxes",
        type=int,
        default=None,
        help="Inspect max_sandboxes setting.",
    )
    prompt.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Inspect max_connections setting.",
    )
    prompt.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    prompt.set_defaults(handler=_handle_prompt_command)

    structural = subparsers.add_parser(
        "structural",
        help="Run structural perturbation reliability phase.",
        description=(
            "Run K independent structural-phase repeats and emit sidecar telemetry. "
            "Supports deterministic bridged tool-response structure transforms."
        ),
    )
    structural.add_argument("--benchmark", required=True, help="Benchmark label.")
    structural.add_argument(
        "--tasks",
        required=True,
        help="Inspect task path (e.g. inspect_evals/gaia_level1).",
    )
    structural.add_argument(
        "--agent",
        dest="agents",
        action="append",
        required=True,
        help="Agent identifier (repeat for multi-agent runs).",
    )
    structural.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent structural repeats per agent (default: 5).",
    )
    structural.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Campaign seed recorded in reliability metadata.",
    )
    structural.add_argument(
        "--model",
        default=None,
        help="Optional model override passed to inspect eval.",
    )
    structural.add_argument(
        "--task-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task arg pair (repeatable).",
    )
    structural.add_argument(
        "--inject-agent-task-arg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject task arg `agent=<agent>` for each run.",
    )
    structural.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional metadata pair (repeatable).",
    )
    structural.add_argument(
        "--sandbox",
        default=None,
        help="Sandbox setting passed to inspect eval.",
    )
    structural.add_argument(
        "--limit",
        default=None,
        help="Sample limit (e.g. 3 or 10-20).",
    )
    structural.add_argument(
        "--sample-id",
        default=None,
        help="Sample id(s): one value or comma-separated list.",
    )
    structural.add_argument(
        "--campaign-id",
        default=None,
        help=(
            "Campaign identifier used to segregate log/sidecar paths. "
            "Defaults to an auto-generated timestamped id."
        ),
    )
    structural.add_argument(
        "--log-root",
        default="logs/reliability",
        help="Root directory for eval logs and default sidecars.",
    )
    structural.add_argument(
        "--sidecar-dir",
        default="reliability_sidecars",
        help="Relative sidecar directory under log root.",
    )
    structural.add_argument(
        "--sidecar-path",
        default=None,
        help="Explicit sidecar JSONL path (overrides default).",
    )
    structural.add_argument(
        "--structural-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable structural perturbation adapter.",
    )
    structural.add_argument(
        "--structural-mode",
        choices=("noop", "key_case_flip", "envelope_wrap", "disabled"),
        default="key_case_flip",
        help="Structural adapter mode.",
    )
    structural.add_argument(
        "--structural-target",
        choices=("tool_response", "tool_schema"),
        default="tool_response",
        help="Structural perturbation target surface.",
    )
    structural.add_argument(
        "--structural-seed",
        type=int,
        default=None,
        help="Optional structural adapter seed override.",
    )
    structural.add_argument(
        "--structural-policy-name",
        default="structural_tool_response_v1",
        help="Structural policy name recorded in metadata.",
    )
    structural.add_argument(
        "--strict-identity-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reliability identity metadata tags.",
    )
    structural.add_argument(
        "--fail-on-missing-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail preflight when reliability hooks are not active.",
    )
    structural.add_argument(
        "--verify-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate sidecar coverage against eval logs.",
    )
    structural.add_argument(
        "--fail-on-incomplete-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail run when telemetry coverage is incomplete.",
    )
    structural.add_argument(
        "--configure-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Configure reliability hooks automatically.",
    )
    structural.add_argument(
        "--orchestrator-mode",
        choices=("single_process", "multi_process"),
        default="multi_process",
        help="Orchestrator policy mode.",
    )
    structural.add_argument(
        "--orchestrator-workers",
        type=int,
        default=1,
        help="Orchestrator worker count.",
    )
    structural.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Inspect max_tasks setting.",
    )
    structural.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Inspect max_samples setting.",
    )
    structural.add_argument(
        "--max-subprocesses",
        type=int,
        default=None,
        help="Inspect max_subprocesses setting.",
    )
    structural.add_argument(
        "--max-sandboxes",
        type=int,
        default=None,
        help="Inspect max_sandboxes setting.",
    )
    structural.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Inspect max_connections setting.",
    )
    structural.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    structural.set_defaults(handler=_handle_structural_command)

    campaign = subparsers.add_parser(
        "campaign",
        help="Run full reliability campaign.",
        description=(
            "Run baseline/fault/prompt/structural reliability phases with a shared "
            "campaign id, resumable manifest, and optional campaign analysis/report."
        ),
    )
    campaign.add_argument("--benchmark", required=True, help="Benchmark label.")
    campaign.add_argument(
        "--tasks",
        required=True,
        help="Inspect task path (e.g. inspect_evals/gaia_level1).",
    )
    campaign.add_argument(
        "--agent",
        dest="agents",
        action="append",
        required=True,
        help="Agent identifier (repeat for multi-agent runs).",
    )
    campaign.add_argument(
        "--phase",
        dest="phases",
        action="append",
        choices=("baseline", "fault", "prompt", "structural", "safety", "abstention"),
        default=None,
        help="Phase to include (repeatable). Defaults to baseline/fault/prompt/structural.",
    )
    campaign.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent repeats per execution phase (default: 5).",
    )
    campaign.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Campaign seed recorded in reliability metadata.",
    )
    campaign.add_argument(
        "--model",
        default=None,
        help="Optional model override passed to inspect eval.",
    )
    campaign.add_argument(
        "--task-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task arg pair (repeatable).",
    )
    campaign.add_argument(
        "--inject-agent-task-arg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Inject task arg `agent=<agent>` for each run.",
    )
    campaign.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional metadata pair (repeatable).",
    )
    campaign.add_argument(
        "--sandbox",
        default=None,
        help="Sandbox setting passed to inspect eval.",
    )
    campaign.add_argument(
        "--limit",
        default=None,
        help="Sample limit (e.g. 3 or 10-20).",
    )
    campaign.add_argument(
        "--sample-id",
        default=None,
        help="Sample id(s): one value or comma-separated list.",
    )
    campaign.add_argument(
        "--campaign-id",
        default=None,
        help="Campaign identifier used for logs, sidecars, and manifest.",
    )
    campaign.add_argument(
        "--log-root",
        default="logs/reliability",
        help="Root directory for eval logs and artifacts.",
    )
    campaign.add_argument(
        "--strict-identity-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require reliability identity metadata tags.",
    )
    campaign.add_argument(
        "--fail-on-missing-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail preflight when reliability hooks are not active.",
    )
    campaign.add_argument(
        "--verify-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate sidecar coverage against eval logs.",
    )
    campaign.add_argument(
        "--fail-on-incomplete-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail run when telemetry coverage is incomplete.",
    )
    campaign.add_argument(
        "--configure-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Configure reliability hooks automatically.",
    )
    campaign.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing campaign manifest when present.",
    )
    campaign.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop campaign immediately when any phase fails.",
    )
    campaign.add_argument(
        "--run-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run campaign-level analysis after execution phases.",
    )
    campaign.add_argument(
        "--write-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write markdown campaign report after analysis.",
    )
    campaign.add_argument(
        "--analysis-source",
        choices=("eval_preferred", "sidecar_only"),
        default="eval_preferred",
        help=(
            "Campaign analysis source mode: prefer canonical eval logs first "
            "or force sidecar-only analysis."
        ),
    )
    campaign.add_argument(
        "--orchestrator-mode",
        choices=("single_process", "multi_process"),
        default="multi_process",
        help="Orchestrator policy mode.",
    )
    campaign.add_argument(
        "--orchestrator-workers",
        type=int,
        default=1,
        help="Orchestrator worker count.",
    )
    campaign.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Inspect max_tasks setting.",
    )
    campaign.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Inspect max_samples setting.",
    )
    campaign.add_argument(
        "--max-subprocesses",
        type=int,
        default=None,
        help="Inspect max_subprocesses setting.",
    )
    campaign.add_argument(
        "--max-sandboxes",
        type=int,
        default=None,
        help="Inspect max_sandboxes setting.",
    )
    campaign.add_argument(
        "--max-connections",
        type=int,
        default=None,
        help="Inspect max_connections setting.",
    )
    campaign.add_argument(
        "--fault-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fault perturbation in campaign fault phase.",
    )
    campaign.add_argument(
        "--fault-mode",
        choices=("noop", "model_error", "tool_error", "disabled"),
        default="noop",
        help="Fault adapter mode for campaign fault phase.",
    )
    campaign.add_argument(
        "--fault-target",
        choices=("model", "tool"),
        default="model",
        help="Fault injection target for campaign fault phase.",
    )
    campaign.add_argument(
        "--fault-rate",
        type=float,
        default=0.0,
        help="Fraction of candidate events to fault in campaign fault phase.",
    )
    campaign.add_argument(
        "--fault-max-per-sample",
        type=int,
        default=1,
        help="Maximum injected faults per sample in campaign fault phase.",
    )
    campaign.add_argument(
        "--fault-seed",
        type=int,
        default=None,
        help="Optional fault seed override.",
    )
    campaign.add_argument(
        "--fault-policy-name",
        default="fault_scaffold_v1",
        help="Fault policy name recorded in metadata.",
    )
    campaign.add_argument(
        "--prompt-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prompt perturbation in campaign prompt phase.",
    )
    campaign.add_argument(
        "--prompt-mode",
        choices=("noop", "rewrite_v1", "rewrite_llm_v1", "disabled"),
        default="rewrite_llm_v1",
        help="Prompt adapter mode for campaign prompt phase.",
    )
    campaign.add_argument(
        "--prompt-variant-count",
        type=int,
        default=1,
        help="Configured prompt variant count.",
    )
    campaign.add_argument(
        "--prompt-seed",
        type=int,
        default=None,
        help="Optional prompt adapter seed override.",
    )
    campaign.add_argument(
        "--prompt-policy-name",
        default="prompt_rewrite_llm_v1",
        help="Prompt policy name recorded in metadata.",
    )
    campaign.add_argument(
        "--prompt-rewrite-model",
        default=None,
        help="Optional dedicated rewrite model for prompt phase.",
    )
    campaign.add_argument(
        "--prompt-rewrite-temperature",
        type=float,
        default=0.2,
        help="Temperature for LLM rewrite mode.",
    )
    campaign.add_argument(
        "--prompt-rewrite-max-tokens",
        type=int,
        default=512,
        help="Max tokens for LLM rewrite mode.",
    )
    campaign.add_argument(
        "--prompt-rewrite-strength",
        choices=("mild", "medium", "strong", "naturalistic"),
        default="medium",
        help="Rewrite strength profile for LLM rewrite mode.",
    )
    campaign.add_argument(
        "--structural-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable structural perturbation in campaign structural phase.",
    )
    campaign.add_argument(
        "--structural-mode",
        choices=("noop", "key_case_flip", "envelope_wrap", "disabled"),
        default="key_case_flip",
        help="Structural adapter mode for campaign structural phase.",
    )
    campaign.add_argument(
        "--structural-target",
        choices=("tool_response",),
        default="tool_response",
        help="Structural perturbation target.",
    )
    campaign.add_argument(
        "--structural-seed",
        type=int,
        default=None,
        help="Optional structural adapter seed override.",
    )
    campaign.add_argument(
        "--structural-policy-name",
        default="structural_scaffold_v1",
        help="Structural policy name recorded in metadata.",
    )
    campaign.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    campaign.set_defaults(handler=_handle_campaign_command)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze baseline reliability artifacts.",
        description=(
            "Analyze baseline sidecar records and report reliability metrics "
            "in a hal-harness-style summary."
        ),
    )
    analyze.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark label used for default sidecar path resolution.",
    )
    analyze.add_argument(
        "--campaign-id",
        default=None,
        help=(
            "Campaign id used for default sidecar path resolution and filtering. "
            "Recommended for campaign-scoped analysis."
        ),
    )
    analyze.add_argument(
        "--agent",
        default=None,
        help="Optional agent filter (e.g. codex_cli).",
    )
    analyze.add_argument(
        "--sidecar-path",
        default=None,
        help="Explicit sidecar JSONL path to analyze.",
    )
    analyze.add_argument(
        "--log-root",
        default="logs/reliability",
        help="Root directory for log/sidecar paths.",
    )
    analyze.add_argument(
        "--sidecar-dir",
        default="reliability_sidecars",
        help="Relative sidecar directory under log root.",
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output.",
    )
    analyze.add_argument(
        "--all-phases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Analyze campaign-wide metrics across baseline/fault/prompt/structural.",
    )
    analyze.add_argument(
        "--analysis-source",
        choices=("eval_preferred", "sidecar_only"),
        default="eval_preferred",
        help=(
            "Analysis source mode for --all-phases: prefer canonical eval logs first "
            "or force sidecar-only analysis."
        ),
    )
    analyze.set_defaults(handler=_handle_analyze_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run reliability CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast("CommandHandler", args.handler)
    try:
        return handler(args)
    except (
        BaselineExecutionError,
        FaultExecutionError,
        PromptExecutionError,
        StructuralExecutionError,
    ) as ex:
        print(f"reliability phase failed: {ex}", file=sys.stderr)
        return 2
    except ValueError as ex:
        print(f"invalid arguments: {ex}", file=sys.stderr)
        return 2


def _handle_baseline_command(args: argparse.Namespace) -> int:
    spec = ReliabilitySpec(
        benchmark=args.benchmark,
        agents=_unique_agents(args.agents),
        phases=["baseline"],
        seed=args.seed,
        sidecar_dir=args.sidecar_dir,
        strict_identity_tags=args.strict_identity_tags,
        fail_on_missing_hooks=args.fail_on_missing_hooks,
        concurrency=OrchestratorConcurrency(
            orchestrator_mode=args.orchestrator_mode,
            orchestrator_workers=args.orchestrator_workers,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            max_subprocesses=args.max_subprocesses,
            max_sandboxes=args.max_sandboxes,
            max_connections=args.max_connections,
        ),
    )

    config = BaselinePhaseConfig(
        repeats=args.repeats,
        campaign_id=args.campaign_id,
        log_root=args.log_root,
        sidecar_path=args.sidecar_path,
        model=args.model,
        task_args=_parse_key_value_pairs(args.task_arg),
        inject_agent_task_arg=args.inject_agent_task_arg,
        metadata=_parse_key_value_pairs(args.metadata),
        sandbox=args.sandbox,
        limit=_parse_limit_arg(args.limit),
        sample_id=_parse_sample_id_arg(args.sample_id),
        verify_telemetry=args.verify_telemetry,
        fail_on_incomplete_telemetry=args.fail_on_incomplete_telemetry,
        configure_hooks=args.configure_hooks,
    )

    result = run_baseline_phase(
        spec=spec,
        tasks=args.tasks,
        config=config,
    )
    _print_baseline_result(result, json_output=args.json)
    return 0


def _print_baseline_result(result: BaselinePhaseResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.model_dump(), indent=2))
        return

    print(
        "Baseline reliability run complete: "
        f"benchmark={result.benchmark} repeats={result.repeats} "
        f"campaign_id={result.campaign_id}"
    )
    print(f"Sidecar: {result.sidecar_path}")
    if result.campaign_json_path:
        print(f"Campaign JSON: {result.campaign_json_path}")
    for row in result.results:
        status = "complete" if row.coverage_complete else "incomplete"
        print(
            f"- agent={row.agent} repeat={row.repeat_id} "
            f"coverage={status} logs={len(row.log_paths)} "
            f"missing={len(row.missing_sample_uuids)} "
            f"duplicates={len(row.duplicate_identity_keys)}"
        )


def _handle_fault_command(args: argparse.Namespace) -> int:
    spec = ReliabilitySpec(
        benchmark=args.benchmark,
        agents=_unique_agents(args.agents),
        phases=["fault"],
        seed=args.seed,
        sidecar_dir=args.sidecar_dir,
        strict_identity_tags=args.strict_identity_tags,
        fail_on_missing_hooks=args.fail_on_missing_hooks,
        concurrency=OrchestratorConcurrency(
            orchestrator_mode=args.orchestrator_mode,
            orchestrator_workers=args.orchestrator_workers,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            max_subprocesses=args.max_subprocesses,
            max_sandboxes=args.max_sandboxes,
            max_connections=args.max_connections,
        ),
    )
    perturbation = FaultPerturbationSpec(
        enabled=args.fault_enabled,
        mode=args.fault_mode,
        target=args.fault_target,
        fault_rate=args.fault_rate,
        max_faults_per_sample=args.fault_max_per_sample,
        seed=args.fault_seed,
        policy_name=args.fault_policy_name,
    )

    config = FaultPhaseConfig(
        repeats=args.repeats,
        campaign_id=args.campaign_id,
        log_root=args.log_root,
        sidecar_path=args.sidecar_path,
        model=args.model,
        task_args=_parse_key_value_pairs(args.task_arg),
        inject_agent_task_arg=args.inject_agent_task_arg,
        metadata=_parse_key_value_pairs(args.metadata),
        sandbox=args.sandbox,
        limit=_parse_limit_arg(args.limit),
        sample_id=_parse_sample_id_arg(args.sample_id),
        verify_telemetry=args.verify_telemetry,
        fail_on_incomplete_telemetry=args.fail_on_incomplete_telemetry,
        configure_hooks=args.configure_hooks,
        perturbation=perturbation,
    )

    result = run_fault_phase(
        spec=spec,
        tasks=args.tasks,
        config=config,
    )
    _print_fault_result(result, json_output=args.json)
    return 0


def _print_fault_result(result: FaultPhaseResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.model_dump(), indent=2))
        return

    print(
        "Fault reliability run complete: "
        f"benchmark={result.benchmark} repeats={result.repeats} "
        f"campaign_id={result.campaign_id}"
    )
    print(
        "Fault policy: "
        f"name={result.perturbation.policy_name} "
        f"enabled={result.perturbation.effective_enabled()} "
        f"mode={result.perturbation.mode} "
        f"target={result.perturbation.target} "
        f"rate={result.perturbation.fault_rate}"
    )
    print(f"Sidecar: {result.sidecar_path}")
    if result.campaign_json_path:
        print(f"Campaign JSON: {result.campaign_json_path}")
    for row in result.results:
        status = "complete" if row.coverage_complete else "incomplete"
        print(
            f"- agent={row.agent} repeat={row.repeat_id} "
            f"coverage={status} logs={len(row.log_paths)} "
            f"missing={len(row.missing_sample_uuids)} "
            f"duplicates={len(row.duplicate_identity_keys)}"
        )


def _handle_prompt_command(args: argparse.Namespace) -> int:
    spec = ReliabilitySpec(
        benchmark=args.benchmark,
        agents=_unique_agents(args.agents),
        phases=["prompt"],
        seed=args.seed,
        sidecar_dir=args.sidecar_dir,
        strict_identity_tags=args.strict_identity_tags,
        fail_on_missing_hooks=args.fail_on_missing_hooks,
        concurrency=OrchestratorConcurrency(
            orchestrator_mode=args.orchestrator_mode,
            orchestrator_workers=args.orchestrator_workers,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            max_subprocesses=args.max_subprocesses,
            max_sandboxes=args.max_sandboxes,
            max_connections=args.max_connections,
        ),
    )
    perturbation = PromptPerturbationSpec(
        enabled=args.prompt_enabled,
        mode=args.prompt_mode,
        variant_count=args.prompt_variant_count,
        seed=args.prompt_seed,
        policy_name=args.prompt_policy_name,
        rewrite_model=args.prompt_rewrite_model,
        rewrite_temperature=args.prompt_rewrite_temperature,
        rewrite_max_tokens=args.prompt_rewrite_max_tokens,
        rewrite_strength=args.prompt_rewrite_strength,
    )

    config = PromptPhaseConfig(
        repeats=args.repeats,
        campaign_id=args.campaign_id,
        log_root=args.log_root,
        sidecar_path=args.sidecar_path,
        model=args.model,
        task_args=_parse_key_value_pairs(args.task_arg),
        inject_agent_task_arg=args.inject_agent_task_arg,
        metadata=_parse_key_value_pairs(args.metadata),
        sandbox=args.sandbox,
        limit=_parse_limit_arg(args.limit),
        sample_id=_parse_sample_id_arg(args.sample_id),
        verify_telemetry=args.verify_telemetry,
        fail_on_incomplete_telemetry=args.fail_on_incomplete_telemetry,
        configure_hooks=args.configure_hooks,
        perturbation=perturbation,
    )

    result = run_prompt_phase(
        spec=spec,
        tasks=args.tasks,
        config=config,
    )
    _print_prompt_result(result, json_output=args.json)
    return 0


def _print_prompt_result(result: PromptPhaseResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.model_dump(), indent=2))
        return

    print(
        "Prompt reliability run complete: "
        f"benchmark={result.benchmark} repeats={result.repeats} "
        f"campaign_id={result.campaign_id}"
    )
    print(
        "Prompt policy: "
        f"name={result.perturbation.policy_name} "
        f"enabled={result.perturbation.effective_enabled()} "
        f"mode={result.perturbation.mode} "
        f"variants={result.perturbation.variant_count}"
    )
    if result.perturbation.mode == "rewrite_llm_v1":
        print(
            "Prompt rewrite config: "
            f"model={result.perturbation.rewrite_model or 'active_model'} "
            f"temperature={result.perturbation.rewrite_temperature} "
            f"max_tokens={result.perturbation.rewrite_max_tokens} "
            f"strength={result.perturbation.rewrite_strength}"
        )
    print(f"Sidecar: {result.sidecar_path}")
    if result.campaign_json_path:
        print(f"Campaign JSON: {result.campaign_json_path}")
    for row in result.results:
        status = "complete" if row.coverage_complete else "incomplete"
        print(
            f"- agent={row.agent} repeat={row.repeat_id} "
            f"coverage={status} logs={len(row.log_paths)} "
            f"missing={len(row.missing_sample_uuids)} "
            f"duplicates={len(row.duplicate_identity_keys)}"
        )


def _handle_structural_command(args: argparse.Namespace) -> int:
    spec = ReliabilitySpec(
        benchmark=args.benchmark,
        agents=_unique_agents(args.agents),
        phases=["structural"],
        seed=args.seed,
        sidecar_dir=args.sidecar_dir,
        strict_identity_tags=args.strict_identity_tags,
        fail_on_missing_hooks=args.fail_on_missing_hooks,
        concurrency=OrchestratorConcurrency(
            orchestrator_mode=args.orchestrator_mode,
            orchestrator_workers=args.orchestrator_workers,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            max_subprocesses=args.max_subprocesses,
            max_sandboxes=args.max_sandboxes,
            max_connections=args.max_connections,
        ),
    )
    perturbation = StructuralPerturbationSpec(
        enabled=args.structural_enabled,
        mode=args.structural_mode,
        target=args.structural_target,
        seed=args.structural_seed,
        policy_name=args.structural_policy_name,
    )

    config = StructuralPhaseConfig(
        repeats=args.repeats,
        campaign_id=args.campaign_id,
        log_root=args.log_root,
        sidecar_path=args.sidecar_path,
        model=args.model,
        task_args=_parse_key_value_pairs(args.task_arg),
        inject_agent_task_arg=args.inject_agent_task_arg,
        metadata=_parse_key_value_pairs(args.metadata),
        sandbox=args.sandbox,
        limit=_parse_limit_arg(args.limit),
        sample_id=_parse_sample_id_arg(args.sample_id),
        verify_telemetry=args.verify_telemetry,
        fail_on_incomplete_telemetry=args.fail_on_incomplete_telemetry,
        configure_hooks=args.configure_hooks,
        perturbation=perturbation,
    )

    result = run_structural_phase(
        spec=spec,
        tasks=args.tasks,
        config=config,
    )
    _print_structural_result(result, json_output=args.json)
    return 0


def _print_structural_result(result: StructuralPhaseResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.model_dump(), indent=2))
        return

    print(
        "Structural reliability run complete: "
        f"benchmark={result.benchmark} repeats={result.repeats} "
        f"campaign_id={result.campaign_id}"
    )
    print(
        "Structural policy: "
        f"name={result.perturbation.policy_name} "
        f"enabled={result.perturbation.effective_enabled()} "
        f"mode={result.perturbation.mode} "
        f"target={result.perturbation.target}"
    )
    print(f"Sidecar: {result.sidecar_path}")
    if result.campaign_json_path:
        print(f"Campaign JSON: {result.campaign_json_path}")
    for row in result.results:
        status = "complete" if row.coverage_complete else "incomplete"
        print(
            f"- agent={row.agent} repeat={row.repeat_id} "
            f"coverage={status} logs={len(row.log_paths)} "
            f"missing={len(row.missing_sample_uuids)} "
            f"duplicates={len(row.duplicate_identity_keys)}"
        )


def _handle_campaign_command(args: argparse.Namespace) -> int:
    phases = args.phases or ["baseline", "fault", "prompt", "structural"]
    spec = ReliabilitySpec(
        benchmark=args.benchmark,
        agents=_unique_agents(args.agents),
        phases=phases,
        seed=args.seed,
        strict_identity_tags=args.strict_identity_tags,
        fail_on_missing_hooks=args.fail_on_missing_hooks,
        fault_perturbation=FaultPerturbationSpec(
            enabled=args.fault_enabled,
            mode=args.fault_mode,
            target=args.fault_target,
            fault_rate=args.fault_rate,
            max_faults_per_sample=args.fault_max_per_sample,
            seed=args.fault_seed,
            policy_name=args.fault_policy_name,
        ),
        prompt_perturbation=PromptPerturbationSpec(
            enabled=args.prompt_enabled,
            mode=args.prompt_mode,
            variant_count=args.prompt_variant_count,
            seed=args.prompt_seed,
            policy_name=args.prompt_policy_name,
            rewrite_model=args.prompt_rewrite_model,
            rewrite_temperature=args.prompt_rewrite_temperature,
            rewrite_max_tokens=args.prompt_rewrite_max_tokens,
            rewrite_strength=args.prompt_rewrite_strength,
        ),
        structural_perturbation=StructuralPerturbationSpec(
            enabled=args.structural_enabled,
            mode=args.structural_mode,
            target=args.structural_target,
            seed=args.structural_seed,
            policy_name=args.structural_policy_name,
        ),
        concurrency=OrchestratorConcurrency(
            orchestrator_mode=args.orchestrator_mode,
            orchestrator_workers=args.orchestrator_workers,
            max_tasks=args.max_tasks,
            max_samples=args.max_samples,
            max_subprocesses=args.max_subprocesses,
            max_sandboxes=args.max_sandboxes,
            max_connections=args.max_connections,
        ),
    )
    config = ReliabilityCampaignConfig(
        campaign_id=args.campaign_id,
        repeats=args.repeats,
        log_root=args.log_root,
        model=args.model,
        task_args=_parse_key_value_pairs(args.task_arg),
        inject_agent_task_arg=args.inject_agent_task_arg,
        metadata=_parse_key_value_pairs(args.metadata),
        sandbox=args.sandbox,
        limit=_parse_limit_arg(args.limit),
        sample_id=_parse_sample_id_arg(args.sample_id),
        verify_telemetry=args.verify_telemetry,
        fail_on_incomplete_telemetry=args.fail_on_incomplete_telemetry,
        configure_hooks=args.configure_hooks,
        resume=args.resume,
        fail_fast=args.fail_fast,
        run_analysis=args.run_analysis,
        write_report=args.write_report,
        analysis_source=args.analysis_source,
    )
    result = run_reliability_campaign(spec=spec, tasks=args.tasks, config=config)
    _print_campaign_result(result, json_output=args.json)
    return 0


def _print_campaign_result(result: ReliabilityCampaignResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.model_dump(), indent=2))
        return
    print(
        "Reliability campaign complete: "
        f"benchmark={result.benchmark} campaign_id={result.campaign_id}"
    )
    print(f"Manifest: {result.manifest_path}")
    for phase, entry in result.phase_entries.items():
        print(
            f"- phase={phase} status={entry.status} "
            f"sidecar={entry.sidecar_path or 'n/a'}"
        )
    if result.analysis_json_path:
        print(f"Analysis JSON: {result.analysis_json_path}")
    if result.report_markdown_path:
        print(f"Report: {result.report_markdown_path}")


def _handle_analyze_command(args: argparse.Namespace) -> int:
    if args.all_phases:
        if not args.benchmark:
            raise ValueError("--benchmark is required for --all-phases analysis")
        if not args.campaign_id:
            raise ValueError("--campaign-id is required for --all-phases analysis")
        result = analyze_reliability_campaign(
            log_root=args.log_root,
            benchmark=args.benchmark,
            campaign_id=args.campaign_id,
            agent=args.agent,
            source=args.analysis_source,
        )
        _print_campaign_analysis_result(result, json_output=args.json)
        return 0

    sidecar_path = args.sidecar_path or _resolve_analyze_sidecar_path(
        benchmark=args.benchmark,
        campaign_id=args.campaign_id,
        log_root=args.log_root,
        sidecar_dir=args.sidecar_dir,
    )
    result = analyze_baseline_campaign(
        sidecar_path=sidecar_path,
        benchmark=args.benchmark,
        campaign_id=args.campaign_id,
        agent=args.agent,
    )
    _print_analyze_result(result, json_output=args.json)
    return 0


def _resolve_analyze_sidecar_path(
    *, benchmark: str | None, campaign_id: str | None, log_root: str, sidecar_dir: str
) -> str:
    if not benchmark:
        raise ValueError(
            "--benchmark is required when --sidecar-path is not provided"
        )
    if campaign_id:
        return str(
            campaign_sidecar_path(
                log_root=log_root,
                benchmark=benchmark,
                phase="baseline",
                campaign_id=campaign_id,
                sidecar_filename="records.jsonl",
            )
        )
    # Backward-compatible path for pre-campaign sidecars.
    return f"{log_root}/{sidecar_dir}/{benchmark}/baseline_records.jsonl"


def _print_analyze_result(result: BaselineAnalysisResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.model_dump(), indent=2))
        return

    print(
        "Baseline analysis complete: "
        f"benchmark={result.benchmark} campaign_id={result.campaign_id} "
        f"records={result.total_records}"
    )
    accuracy = _fmt_metric(result.accuracy)
    print(f"Accuracy: {accuracy}")
    print(
        "  consistency_outcome: "
        f"{_fmt_metric(result.consistency.outcome)}, "
        "consistency_trajectory_distribution: "
        f"{_fmt_metric(result.consistency.trajectory_distribution)}, "
        "consistency_trajectory_sequence: "
        f"{_fmt_metric(result.consistency.trajectory_sequence)}"
    )
    print(
        "  consistency_confidence: "
        f"{_fmt_metric(result.consistency.confidence)}, "
        "consistency_resource: "
        f"{_fmt_metric(result.consistency.resource)}"
    )
    print(
        "Resources (total_time sec): "
        f"mean={_fmt_metric(result.resources.total_time_mean_sec)}, "
        f"std={_fmt_metric(result.resources.total_time_stddev_sec)}, "
        f"min={_fmt_metric(result.resources.total_time_min_sec)}, "
        f"max={_fmt_metric(result.resources.total_time_max_sec)}"
    )
    if result.notes:
        print("Notes:")
        for note in result.notes:
            print(f"- {note}")


def _print_campaign_analysis_result(
    result: CampaignAnalysisResult, *, json_output: bool
) -> None:
    if json_output:
        print(json.dumps(result.model_dump(), indent=2))
        return

    print(
        "Campaign analysis complete: "
        f"benchmark={result.benchmark} campaign_id={result.campaign_id}"
    )
    print(
        "Campaign resources: "
        f"started_at={result.resources.started_at or 'n/a'} "
        f"completed_at={result.resources.completed_at or 'n/a'} "
        f"wall_time={_fmt_duration(result.resources.wall_time_sec)} "
        f"working_time={_fmt_duration(result.resources.working_time_sec)} "
        f"cost={_fmt_metric(result.resources.total_cost_usd)} "
        f"tokens={result.resources.total_tokens if result.resources.total_tokens is not None else 'n/a'}"
    )
    print("Phase accuracies:")
    for phase in ("baseline", "fault", "prompt", "structural"):
        summary = result.phase_summaries.get(phase)
        accuracy = _fmt_metric(summary.accuracy if summary else None)
        records = summary.total_records if summary else 0
        source = summary.source if summary else "n/a"
        started_at = summary.started_at if summary else None
        completed_at = summary.completed_at if summary else None
        cost = summary.total_cost_usd if summary else None
        print(
            f"- {phase}: accuracy={accuracy} records={records} source={source} "
            f"started_at={started_at or 'n/a'} completed_at={completed_at or 'n/a'} "
            f"wall_time={_fmt_duration(summary.wall_time_sec if summary else None)} "
            f"cost={_fmt_metric(cost)}"
        )

    print(
        "Predictability: "
        f"pairs={result.predictability.pair_count}, "
        f"brier_mse={_fmt_metric(result.predictability.brier_mse)}, "
        "brier_predictability="
        f"{_fmt_metric(result.predictability.brier_predictability)}, "
        "calibration_error="
        f"{_fmt_metric(result.predictability.calibration_error)}, "
        "discrimination_auroc="
        f"{_fmt_metric(result.predictability.discrimination_auroc)}"
    )
    print(
        "Robustness deltas vs baseline: "
        f"fault={_fmt_metric(result.robustness.fault_delta_vs_baseline)}, "
        f"prompt={_fmt_metric(result.robustness.prompt_delta_vs_baseline)}, "
        f"structural={_fmt_metric(result.robustness.structural_delta_vs_baseline)}"
    )
    print(
        "Safety: "
        f"violation_rate={_fmt_metric(result.safety.violation_rate)} "
        f"({result.safety.violation_count}/{result.safety.observed_records})"
    )
    print(
        "Abstention: "
        f"abstention_rate={_fmt_metric(result.abstention.abstention_rate)} "
        f"({result.abstention.abstention_count}/{result.abstention.observed_records}), "
        f"selective_accuracy={_fmt_metric(result.abstention.selective_accuracy)}"
    )
    if result.notes:
        print("Notes:")
        for note in result.notes:
            print(f"- {note}")


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_duration(value: float | None) -> str:
    if value is None:
        return "n/a"
    total_seconds = int(round(value))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_key_value_pairs(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"expected KEY=VALUE pair, received: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"metadata key cannot be empty in pair: {raw}")
        parsed[key] = _coerce_scalar(value.strip())
    return parsed


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _parse_limit_arg(value: str | None) -> int | tuple[int, int] | None:
    if value is None:
        return None
    token = value.strip()
    if not token:
        raise ValueError("limit cannot be empty")
    if "-" in token:
        start_raw, end_raw = token.split("-", 1)
        start = int(start_raw.strip())
        end = int(end_raw.strip())
        if start < 1 or end < start:
            raise ValueError("limit range must be in form start-end with 1 <= start <= end")
        return (start, end)
    parsed = int(token)
    if parsed < 1:
        raise ValueError("limit must be >= 1")
    return parsed


def _parse_sample_id_arg(
    value: str | None,
) -> str | int | list[str] | list[int] | list[str | int] | None:
    if value is None:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("sample-id cannot be empty")

    parsed: list[str | int] = []
    for token in tokens:
        if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
            parsed.append(int(token))
        else:
            parsed.append(token)

    if len(parsed) == 1:
        return parsed[0]
    return parsed


def _unique_agents(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        name = raw.strip()
        if not name or name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    if not ordered:
        raise ValueError("at least one non-empty --agent value is required")
    return ordered


CommandHandler = Callable[[argparse.Namespace], int]


if __name__ == "__main__":
    raise SystemExit(main())
