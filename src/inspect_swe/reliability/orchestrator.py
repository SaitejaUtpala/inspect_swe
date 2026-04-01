"""Reliability campaign orchestration primitives.

This module intentionally focuses on preflight and sharding policy first:
- validates canonical storage contract (`.eval` logs),
- validates concurrency policy constraints,
- validates hook readiness and telemetry requirements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .analysis import analyze_reliability_campaign
from .concurrency import validate_orchestrator_policy
from .hooks import assert_reliability_hooks_active
from .paths import (
    campaign_analysis_path,
    campaign_manifest_path,
    campaign_report_path,
)
from .reporting import write_campaign_analysis_json, write_campaign_markdown_report
from .spec import PhaseName, ReliabilitySpec


@dataclass(frozen=True)
class PhaseShard:
    """One campaign execution shard."""

    benchmark: str
    phase: PhaseName
    agent: str
    shard_index: int
    output_dir: str


def preflight_reliability_spec(spec: ReliabilitySpec) -> None:
    """Fail-fast validation for reliability runs."""
    if spec.canonical_log_format != "eval":
        raise ValueError(
            "Only Inspect `.eval` canonical logs are supported for reliability execution."
        )
    validate_orchestrator_policy(spec.concurrency)
    if spec.fail_on_missing_hooks:
        assert_reliability_hooks_active(require_enabled=True)


def build_phase_shards(spec: ReliabilitySpec, output_root: str | Path) -> list[PhaseShard]:
    """Create deterministic phase shards from a reliability spec."""
    root = Path(output_root)
    shards: list[PhaseShard] = []
    shard_index = 0
    for phase in spec.phases:
        for agent in spec.agents:
            output_dir = root / spec.benchmark / phase / agent
            shards.append(
                PhaseShard(
                    benchmark=spec.benchmark,
                    phase=phase,
                    agent=agent,
                    shard_index=shard_index,
                    output_dir=str(output_dir),
                )
            )
            shard_index += 1
    return shards


CampaignPhaseStatus = Literal["pending", "completed", "failed", "skipped"]


class CampaignPhaseEntry(BaseModel):
    """Manifest entry for one phase."""

    status: CampaignPhaseStatus = "pending"
    sidecar_path: str | None = None
    campaign_json_path: str | None = None
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class ReliabilityCampaignManifest(BaseModel):
    """Persistent campaign manifest used for resume semantics."""

    artifact_type: str = "inspect_swe_reliability_campaign_manifest"
    artifact_version: int = 1
    benchmark: str
    campaign_id: str
    tasks: str
    agents: list[str]
    phases: list[PhaseName]
    created_at: str
    updated_at: str
    phase_entries: dict[str, CampaignPhaseEntry] = Field(default_factory=dict)
    analysis_json_path: str | None = None
    report_markdown_path: str | None = None


class ReliabilityCampaignConfig(BaseModel):
    """Top-level orchestration config for multi-phase campaigns."""

    campaign_id: str | None = None
    repeats: int = Field(default=5, ge=1)
    log_root: str = "logs/reliability"
    model: str | None = None
    task_args: dict[str, Any] = Field(default_factory=dict)
    inject_agent_task_arg: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    sandbox: str | None = None
    limit: int | tuple[int, int] | None = None
    sample_id: str | int | list[str] | list[int] | list[str | int] | None = None
    verify_telemetry: bool = True
    fail_on_incomplete_telemetry: bool = True
    configure_hooks: bool = True
    resume: bool = True
    fail_fast: bool = True
    run_analysis: bool = True
    write_report: bool = True

    @field_validator("campaign_id")
    @classmethod
    def _validate_campaign_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("campaign_id cannot be empty")
        return cleaned


class ReliabilityCampaignResult(BaseModel):
    """End-to-end campaign execution summary."""

    benchmark: str
    campaign_id: str
    manifest_path: str
    phase_entries: dict[str, CampaignPhaseEntry] = Field(default_factory=dict)
    analysis_json_path: str | None = None
    report_markdown_path: str | None = None


def run_reliability_campaign(
    *,
    spec: ReliabilitySpec,
    tasks: str,
    config: ReliabilityCampaignConfig | None = None,
) -> ReliabilityCampaignResult:
    """Run all configured phases with manifest-backed resume support."""
    config = config or ReliabilityCampaignConfig()
    campaign_id = config.campaign_id or _default_campaign_id()
    _preflight_campaign(spec, configure_hooks=config.configure_hooks)

    manifest_path = campaign_manifest_path(
        log_root=config.log_root,
        benchmark=spec.benchmark,
        campaign_id=campaign_id,
    )
    manifest = _load_or_initialize_manifest(
        path=manifest_path,
        spec=spec,
        tasks=tasks,
        campaign_id=campaign_id,
        resume=config.resume,
    )

    for phase in spec.phases:
        phase_key = str(phase)
        existing = manifest.phase_entries.get(phase_key)
        if (
            config.resume
            and existing is not None
            and existing.status == "completed"
            and phase not in {"safety", "abstention"}
        ):
            continue

        if phase in {"safety", "abstention"}:
            manifest.phase_entries[phase_key] = CampaignPhaseEntry(
                status="skipped",
                error=(
                    "posthoc phase; computed from campaign analysis and does not "
                    "execute inspect eval runs"
                ),
            )
            _persist_manifest(manifest, manifest_path)
            continue

        try:
            result = _run_execution_phase(
                phase=phase,
                spec=spec,
                tasks=tasks,
                campaign_id=campaign_id,
                config=config,
            )
            manifest.phase_entries[phase_key] = CampaignPhaseEntry(
                status="completed",
                sidecar_path=result.get("sidecar_path"),
                campaign_json_path=result.get("campaign_json_path"),
                result=result,
            )
        except Exception as ex:
            manifest.phase_entries[phase_key] = CampaignPhaseEntry(
                status="failed",
                error=str(ex),
            )
            _persist_manifest(manifest, manifest_path)
            if config.fail_fast:
                raise

        _persist_manifest(manifest, manifest_path)

    if config.run_analysis:
        analysis = analyze_reliability_campaign(
            log_root=config.log_root,
            benchmark=spec.benchmark,
            campaign_id=campaign_id,
        )
        analysis_path = write_campaign_analysis_json(
            result=analysis,
            output_path=campaign_analysis_path(
                log_root=config.log_root,
                benchmark=spec.benchmark,
                campaign_id=campaign_id,
            ),
        )
        manifest.analysis_json_path = analysis_path
        if config.write_report:
            manifest.report_markdown_path = write_campaign_markdown_report(
                result=analysis,
                output_path=campaign_report_path(
                    log_root=config.log_root,
                    benchmark=spec.benchmark,
                    campaign_id=campaign_id,
                ),
            )

        for phase in spec.phases:
            if phase not in {"safety", "abstention"}:
                continue
            manifest.phase_entries[str(phase)] = CampaignPhaseEntry(
                status="completed",
                result={"mode": "posthoc_analysis"},
            )

    _persist_manifest(manifest, manifest_path)
    return ReliabilityCampaignResult(
        benchmark=spec.benchmark,
        campaign_id=campaign_id,
        manifest_path=str(manifest_path),
        phase_entries=manifest.phase_entries,
        analysis_json_path=manifest.analysis_json_path,
        report_markdown_path=manifest.report_markdown_path,
    )


def _run_execution_phase(
    *,
    phase: PhaseName,
    spec: ReliabilitySpec,
    tasks: str,
    campaign_id: str,
    config: ReliabilityCampaignConfig,
) -> dict[str, Any]:
    phase_spec = spec.model_copy(update={"phases": [phase]})

    if phase == "baseline":
        from .baseline import BaselinePhaseConfig, run_baseline_phase

        result = run_baseline_phase(
            spec=phase_spec,
            tasks=tasks,
            config=BaselinePhaseConfig(
                repeats=config.repeats,
                campaign_id=campaign_id,
                log_root=config.log_root,
                model=config.model,
                task_args=config.task_args,
                inject_agent_task_arg=config.inject_agent_task_arg,
                metadata=config.metadata,
                sandbox=config.sandbox,
                limit=config.limit,
                sample_id=config.sample_id,
                verify_telemetry=config.verify_telemetry,
                fail_on_incomplete_telemetry=config.fail_on_incomplete_telemetry,
                configure_hooks=config.configure_hooks,
            ),
        )
        return result.model_dump(mode="json")

    if phase == "fault":
        from .fault import FaultPhaseConfig, run_fault_phase

        result = run_fault_phase(
            spec=phase_spec,
            tasks=tasks,
            config=FaultPhaseConfig(
                repeats=config.repeats,
                campaign_id=campaign_id,
                log_root=config.log_root,
                model=config.model,
                task_args=config.task_args,
                inject_agent_task_arg=config.inject_agent_task_arg,
                metadata=config.metadata,
                sandbox=config.sandbox,
                limit=config.limit,
                sample_id=config.sample_id,
                verify_telemetry=config.verify_telemetry,
                fail_on_incomplete_telemetry=config.fail_on_incomplete_telemetry,
                configure_hooks=config.configure_hooks,
                perturbation=spec.fault_perturbation,
            ),
        )
        return result.model_dump(mode="json")

    if phase == "prompt":
        from .prompt import PromptPhaseConfig, run_prompt_phase

        result = run_prompt_phase(
            spec=phase_spec,
            tasks=tasks,
            config=PromptPhaseConfig(
                repeats=config.repeats,
                campaign_id=campaign_id,
                log_root=config.log_root,
                model=config.model,
                task_args=config.task_args,
                inject_agent_task_arg=config.inject_agent_task_arg,
                metadata=config.metadata,
                sandbox=config.sandbox,
                limit=config.limit,
                sample_id=config.sample_id,
                verify_telemetry=config.verify_telemetry,
                fail_on_incomplete_telemetry=config.fail_on_incomplete_telemetry,
                configure_hooks=config.configure_hooks,
                perturbation=spec.prompt_perturbation,
            ),
        )
        return result.model_dump(mode="json")

    if phase == "structural":
        from .structural import StructuralPhaseConfig, run_structural_phase

        result = run_structural_phase(
            spec=phase_spec,
            tasks=tasks,
            config=StructuralPhaseConfig(
                repeats=config.repeats,
                campaign_id=campaign_id,
                log_root=config.log_root,
                model=config.model,
                task_args=config.task_args,
                inject_agent_task_arg=config.inject_agent_task_arg,
                metadata=config.metadata,
                sandbox=config.sandbox,
                limit=config.limit,
                sample_id=config.sample_id,
                verify_telemetry=config.verify_telemetry,
                fail_on_incomplete_telemetry=config.fail_on_incomplete_telemetry,
                configure_hooks=config.configure_hooks,
                perturbation=spec.structural_perturbation,
            ),
        )
        return result.model_dump(mode="json")

    raise ValueError(f"unsupported execution phase for orchestrator: {phase}")


def _preflight_campaign(spec: ReliabilitySpec, *, configure_hooks: bool) -> None:
    if spec.canonical_log_format != "eval":
        raise ValueError(
            "Only Inspect `.eval` canonical logs are supported for reliability execution."
        )
    validate_orchestrator_policy(spec.concurrency)
    if spec.fail_on_missing_hooks and not configure_hooks:
        assert_reliability_hooks_active(require_enabled=True)


def _load_or_initialize_manifest(
    *,
    path: Path,
    spec: ReliabilitySpec,
    tasks: str,
    campaign_id: str,
    resume: bool,
) -> ReliabilityCampaignManifest:
    if resume and path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = ReliabilityCampaignManifest.model_validate(json.load(f))
        if loaded.benchmark != spec.benchmark:
            raise ValueError(
                "manifest benchmark mismatch: "
                f"expected {spec.benchmark}, found {loaded.benchmark}"
            )
        if loaded.campaign_id != campaign_id:
            raise ValueError(
                "manifest campaign id mismatch: "
                f"expected {campaign_id}, found {loaded.campaign_id}"
            )
        loaded.updated_at = _timestamp()
        return loaded

    now = _timestamp()
    return ReliabilityCampaignManifest(
        benchmark=spec.benchmark,
        campaign_id=campaign_id,
        tasks=tasks,
        agents=list(spec.agents),
        phases=list(spec.phases),
        created_at=now,
        updated_at=now,
    )


def _persist_manifest(manifest: ReliabilityCampaignManifest, path: Path) -> None:
    manifest.updated_at = _timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2)
        f.write("\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_campaign_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{timestamp}_{suffix}"
