"""Reliability analysis across sidecar and Inspect `.eval` logs."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Literal

from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelCost, get_model_info
from inspect_ai.model._model import compute_model_cost
from pydantic import BaseModel, Field

from .artifacts import ReliabilityRecord, load_sidecar_records
from .identity import ReliabilityRunIdentity
from .outcome import outcome_binary_from_outcome, outcome_label_from_outcome
from .paths import campaign_benchmark_dir, campaign_sidecar_path


class ConsistencyMetrics(BaseModel):
    """Consistency metrics inspired by hal-harness style reporting."""

    outcome: float | None = None
    trajectory_distribution: float | None = None
    trajectory_sequence: float | None = None
    confidence: float | None = None
    resource: float | None = None


class ResourceMetrics(BaseModel):
    """Resource spread summary."""

    total_time_mean_sec: float | None = None
    total_time_stddev_sec: float | None = None
    total_time_min_sec: float | None = None
    total_time_max_sec: float | None = None


class BaselineAnalysisResult(BaseModel):
    """Summary produced by baseline analysis phase."""

    phase: Literal["baseline"] = "baseline"
    benchmark: str | None = None
    campaign_id: str | None = None
    agent: str | None = None
    sidecar_path: str
    total_records: int
    unique_runs: int
    repeats: list[int] = Field(default_factory=list)
    records_per_repeat: dict[str, int] = Field(default_factory=dict)
    sample_count: int
    outcome_label_counts: dict[str, int] = Field(default_factory=dict)
    accuracy: float | None = None
    consistency: ConsistencyMetrics = Field(default_factory=ConsistencyMetrics)
    resources: ResourceMetrics = Field(default_factory=ResourceMetrics)
    notes: list[str] = Field(default_factory=list)


def analyze_baseline_campaign(
    *,
    sidecar_path: str,
    benchmark: str | None = None,
    campaign_id: str | None = None,
    agent: str | None = None,
) -> BaselineAnalysisResult:
    """Analyze baseline sidecar records for one campaign."""
    records = load_sidecar_records(sidecar_path)
    filtered = [
        record
        for record in records
        if record.identity.phase == "baseline"
        and (agent is None or record.identity.agent == agent)
        and (
            campaign_id is None
            or record.metadata.get("reliability_campaign_id") == campaign_id
        )
    ]
    if not filtered:
        raise ValueError(
            "no baseline sidecar records matched the requested filters "
            f"(campaign_id={campaign_id!r}, agent={agent!r})"
        )

    by_sample: dict[str, list[ReliabilityRecord]] = defaultdict(list)
    repeat_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    run_ids: set[str] = set()
    accuracy_values: list[float] = []
    resource_times: list[float] = []

    for record in filtered:
        sample_key = str(record.identity.sample_id)
        by_sample[sample_key].append(record)
        repeat_counts[record.identity.repeat_id] += 1
        run_ids.add(record.identity.run_id)

        label = outcome_label_from_outcome(record.outcome)
        if label is not None:
            label_counts[label] += 1

        outcome_value = outcome_binary_from_outcome(record.outcome)
        if outcome_value is not None:
            accuracy_values.append(float(outcome_value))

        time_value = _to_float(record.resources.get("total_time"))
        if time_value is not None:
            resource_times.append(time_value)

    consistency = ConsistencyMetrics(
        outcome=_consistency_outcome(by_sample.values()),
        trajectory_distribution=_consistency_trajectory_distribution(by_sample.values()),
        trajectory_sequence=_consistency_trajectory_sequence(by_sample.values()),
        confidence=_consistency_confidence(by_sample.values()),
        resource=_consistency_resource(by_sample.values()),
    )

    resources = ResourceMetrics()
    if resource_times:
        resources.total_time_mean_sec = fmean(resource_times)
        resources.total_time_stddev_sec = pstdev(resource_times)
        resources.total_time_min_sec = min(resource_times)
        resources.total_time_max_sec = max(resource_times)

    notes: list[str] = []
    if consistency.confidence is None:
        notes.append(
            "confidence consistency unavailable; no numeric confidence values found"
        )

    return BaselineAnalysisResult(
        benchmark=benchmark,
        campaign_id=campaign_id,
        agent=agent,
        sidecar_path=sidecar_path,
        total_records=len(filtered),
        unique_runs=len(run_ids),
        repeats=sorted(repeat_counts.keys()),
        records_per_repeat={str(k): v for k, v in sorted(repeat_counts.items())},
        sample_count=len(by_sample),
        outcome_label_counts=dict(label_counts),
        accuracy=fmean(accuracy_values) if accuracy_values else None,
        consistency=consistency,
        resources=resources,
        notes=notes,
    )


def _consistency_outcome(sample_records: Iterable[list[ReliabilityRecord]]) -> float | None:
    sample_scores: list[float] = []
    for records in sample_records:
        values = [outcome_binary_from_outcome(record.outcome) for record in records]
        if len(values) < 2 or any(value is None for value in values):
            continue
        counts = Counter(values)
        sample_scores.append(counts.most_common(1)[0][1] / len(values))
    return fmean(sample_scores) if sample_scores else None


def _consistency_trajectory_distribution(
    sample_records: Iterable[list[ReliabilityRecord]],
) -> float | None:
    sample_scores: list[float] = []
    for records in sample_records:
        dists = [_event_distribution(record.behavior) for record in records]
        dists = [dist for dist in dists if dist]
        if len(dists) < 2:
            continue
        pair_scores = [_distribution_overlap(a, b) for a, b in combinations(dists, 2)]
        sample_scores.append(fmean(pair_scores))
    return fmean(sample_scores) if sample_scores else None


def _consistency_trajectory_sequence(
    sample_records: Iterable[list[ReliabilityRecord]],
) -> float | None:
    sample_scores: list[float] = []
    for records in sample_records:
        sequences = [_event_sequence(record.behavior) for record in records]
        sequences = [sequence for sequence in sequences if sequence]
        if len(sequences) < 2:
            continue
        pair_scores = [
            SequenceMatcher(None, a, b).ratio() for a, b in combinations(sequences, 2)
        ]
        sample_scores.append(fmean(pair_scores))
    return fmean(sample_scores) if sample_scores else None


def _consistency_confidence(
    sample_records: Iterable[list[ReliabilityRecord]],
) -> float | None:
    sample_scores: list[float] = []
    for records in sample_records:
        values = [_confidence_value(record) for record in records]
        values = [value for value in values if value is not None]
        score = _stability_score(values)
        if score is not None:
            sample_scores.append(score)
    return fmean(sample_scores) if sample_scores else None


def _consistency_resource(
    sample_records: Iterable[list[ReliabilityRecord]],
) -> float | None:
    sample_scores: list[float] = []
    for records in sample_records:
        values = [_to_float(record.resources.get("total_time")) for record in records]
        values = [value for value in values if value is not None]
        score = _stability_score(values)
        if score is not None:
            sample_scores.append(score)
    return fmean(sample_scores) if sample_scores else None


def _event_sequence(behavior: dict[str, Any]) -> tuple[str, ...]:
    event_types = behavior.get("event_types")
    if not isinstance(event_types, list):
        return tuple()
    cleaned = [event for event in event_types if isinstance(event, str)]
    return tuple(cleaned)


def _event_distribution(behavior: dict[str, Any]) -> dict[str, float]:
    sequence = _event_sequence(behavior)
    if not sequence:
        return {}
    counts = Counter(sequence)
    total = sum(counts.values())
    return {name: count / total for name, count in counts.items()}


def _distribution_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)


def _confidence_value(record: ReliabilityRecord) -> float | None:
    value = _to_float(record.confidence.get("value"))
    if value is not None:
        return value
    for key in ("reliability_confidence", "confidence", "self_confidence"):
        value = _to_float(record.metadata.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _stability_score(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    if len(set(values)) == 1:
        return 1.0

    mean_abs = abs(fmean(values))
    std = pstdev(values)
    if mean_abs == 0:
        return 1.0 / (1.0 + std)
    cv = std / mean_abs
    return 1.0 / (1.0 + cv)


class PhaseAnalysisSummary(BaseModel):
    """Aggregate summary for one reliability phase."""

    phase: str
    sidecar_path: str
    source: Literal["eval", "sidecar"] = "sidecar"
    total_records: int
    sample_count: int
    repeat_count: int
    accuracy: float | None = None
    started_at: str | None = None
    completed_at: str | None = None
    wall_time_sec: float | None = None
    working_time_sec: float | None = None
    total_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    agents: list[str] = Field(default_factory=list)


class PredictabilityMetrics(BaseModel):
    """Confidence calibration and discrimination metrics."""

    pair_count: int = 0
    brier_mse: float | None = None
    brier_predictability: float | None = None
    calibration_error: float | None = None
    discrimination_auroc: float | None = None


class RobustnessMetrics(BaseModel):
    """Robustness deltas relative to baseline."""

    baseline_accuracy: float | None = None
    fault_accuracy: float | None = None
    prompt_accuracy: float | None = None
    structural_accuracy: float | None = None
    fault_delta_vs_baseline: float | None = None
    prompt_delta_vs_baseline: float | None = None
    structural_delta_vs_baseline: float | None = None


class SafetyMetrics(BaseModel):
    """Safety signal summary from sidecar safety payloads."""

    observed_records: int = 0
    violation_count: int = 0
    violation_rate: float | None = None


class AbstentionMetrics(BaseModel):
    """Abstention signal summary from sidecar abstention payloads."""

    observed_records: int = 0
    abstention_count: int = 0
    abstention_rate: float | None = None
    selective_accuracy: float | None = None


class CampaignResourceSummary(BaseModel):
    """Campaign-level runtime and usage summary."""

    started_at: str | None = None
    completed_at: str | None = None
    wall_time_sec: float | None = None
    working_time_sec: float | None = None
    total_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


class CampaignAnalysisResult(BaseModel):
    """Cross-phase campaign analysis summary."""

    benchmark: str
    campaign_id: str
    agent: str | None = None
    phase_summaries: dict[str, PhaseAnalysisSummary] = Field(default_factory=dict)
    predictability: PredictabilityMetrics = Field(default_factory=PredictabilityMetrics)
    robustness: RobustnessMetrics = Field(default_factory=RobustnessMetrics)
    safety: SafetyMetrics = Field(default_factory=SafetyMetrics)
    abstention: AbstentionMetrics = Field(default_factory=AbstentionMetrics)
    resources: CampaignResourceSummary = Field(default_factory=CampaignResourceSummary)
    notes: list[str] = Field(default_factory=list)


def analyze_reliability_campaign(
    *,
    log_root: str,
    benchmark: str,
    campaign_id: str,
    agent: str | None = None,
    source: Literal["eval_preferred", "sidecar_only"] = "eval_preferred",
) -> CampaignAnalysisResult:
    """Analyze a campaign across reliability phases."""
    phase_records: dict[str, list[ReliabilityRecord]] = {}
    phase_summaries: dict[str, PhaseAnalysisSummary] = {}
    notes: list[str] = []

    for phase in ("baseline", "fault", "prompt", "structural"):
        sidecar_path = str(
            campaign_sidecar_path(
                log_root=log_root,
                benchmark=benchmark,
                phase=phase,
                campaign_id=campaign_id,
                sidecar_filename="records.jsonl",
            )
        )
        records: list[ReliabilityRecord] = []
        record_source: Literal["eval", "sidecar"] = "sidecar"
        resource_summary = CampaignResourceSummary()
        if source == "eval_preferred":
            records, resource_summary = _phase_records_from_eval_logs(
                log_root=log_root,
                benchmark=benchmark,
                campaign_id=campaign_id,
                phase=phase,
                agent=agent,
            )
            if records:
                record_source = "eval"

        if not records:
            records = [
                record
                for record in load_sidecar_records(sidecar_path)
                if record.identity.phase == phase
                and (
                    campaign_id == record.metadata.get("reliability_campaign_id")
                    or record.metadata.get("reliability_campaign_id") is None
                )
                and (agent is None or record.identity.agent == agent)
            ]
            resource_summary = _resource_summary_from_sidecar_records(records)
            if records and source == "eval_preferred":
                notes.append(
                    f"phase={phase} used sidecar fallback (no matching eval logs with campaign metadata)"
                )
        phase_records[phase] = records
        if not records:
            notes.append(f"no records found for phase={phase}")
            continue

        outcomes = [outcome_binary_from_outcome(record.outcome) for record in records]
        outcome_values = [float(value) for value in outcomes if value is not None]
        sample_keys = {record.identity.sample_uuid for record in records}
        repeats = {record.identity.repeat_id for record in records}
        agents = sorted({record.identity.agent for record in records})
        phase_summaries[phase] = PhaseAnalysisSummary(
            phase=phase,
            sidecar_path=sidecar_path,
            source=record_source,
            total_records=len(records),
            sample_count=len(sample_keys),
            repeat_count=len(repeats),
            accuracy=fmean(outcome_values) if outcome_values else None,
            started_at=resource_summary.started_at,
            completed_at=resource_summary.completed_at,
            wall_time_sec=resource_summary.wall_time_sec,
            working_time_sec=resource_summary.working_time_sec,
            total_cost_usd=resource_summary.total_cost_usd,
            input_tokens=resource_summary.input_tokens,
            output_tokens=resource_summary.output_tokens,
            cache_read_tokens=resource_summary.cache_read_tokens,
            cache_write_tokens=resource_summary.cache_write_tokens,
            reasoning_tokens=resource_summary.reasoning_tokens,
            total_tokens=resource_summary.total_tokens,
            agents=agents,
        )

    baseline_records = phase_records.get("baseline", [])
    robustness = _robustness_metrics(phase_summaries)
    predictability = _predictability_metrics(baseline_records)
    safety = _safety_metrics(_flatten_phase_records(phase_records))
    abstention = _abstention_metrics(_flatten_phase_records(phase_records))
    resources = _campaign_resource_summary(phase_summaries)

    if predictability.pair_count == 0:
        notes.append("predictability unavailable; no confidence/outcome pairs in baseline")
    if safety.observed_records == 0:
        notes.append("safety metrics unavailable; no safety signals found")
    if abstention.observed_records == 0:
        notes.append("abstention metrics unavailable; no abstention signals found")

    return CampaignAnalysisResult(
        benchmark=benchmark,
        campaign_id=campaign_id,
        agent=agent,
        phase_summaries=phase_summaries,
        predictability=predictability,
        robustness=robustness,
        safety=safety,
        abstention=abstention,
        resources=resources,
        notes=notes,
    )


def _phase_records_from_eval_logs(
    *,
    log_root: str,
    benchmark: str,
    campaign_id: str,
    phase: str,
    agent: str | None,
) -> tuple[list[ReliabilityRecord], CampaignResourceSummary]:
    records: list[ReliabilityRecord] = []
    eval_logs: list[Any] = []
    for run_dir, inferred_agent in _iter_phase_run_dirs(
        log_root=log_root,
        benchmark=benchmark,
        campaign_id=campaign_id,
        phase=phase,
    ):
        if agent is not None and inferred_agent != agent:
            continue
        for eval_path in run_dir.glob("*.eval"):
            try:
                eval_log = read_eval_log(str(eval_path))
            except Exception:
                continue
            if eval_log is None:
                continue
            eval_logs.append(eval_log)
            records.extend(
                _records_from_eval_log(
                    eval_log=eval_log,
                    benchmark=benchmark,
                    phase=phase,
                    campaign_id=campaign_id,
                    default_agent=inferred_agent,
                )
            )
    return records, _resource_summary_from_eval_logs(eval_logs)


def _iter_phase_run_dirs(
    *,
    log_root: str,
    benchmark: str,
    campaign_id: str,
    phase: str,
) -> list[tuple[Path, str]]:
    run_dirs: list[tuple[Path, str]] = []

    new_root = campaign_benchmark_dir(
        log_root=log_root,
        benchmark=benchmark,
        campaign_id=campaign_id,
    )
    if new_root.exists():
        for run_dir in new_root.iterdir():
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            if f"_{phase}_rep_" not in run_name or not run_name.endswith(f"_{campaign_id}"):
                continue
            run_dirs.append(
                (
                    run_dir,
                    _agent_name_from_run_dir(run_name, benchmark, phase, campaign_id),
                )
            )

    legacy_root = Path(log_root) / benchmark
    if legacy_root.exists():
        for agent_dir in legacy_root.iterdir():
            if not agent_dir.is_dir():
                continue
            agent_name = agent_dir.name
            for run_dir in agent_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                run_name = run_dir.name
                if f"_{phase}_rep_" not in run_name or not run_name.endswith(f"_{campaign_id}"):
                    continue
                run_dirs.append((run_dir, agent_name))
    return run_dirs


def _agent_name_from_run_dir(
    run_name: str,
    benchmark: str,
    phase: str,
    campaign_id: str,
) -> str:
    prefix = f"{benchmark}_"
    suffix = f"_{phase}_rep_"
    if run_name.startswith(prefix) and suffix in run_name and run_name.endswith(f"_{campaign_id}"):
        agent_segment = run_name[len(prefix) : run_name.index(suffix)]
        if agent_segment:
            return agent_segment
    return "unknown_agent"


def _records_from_eval_log(
    *,
    eval_log: Any,
    benchmark: str,
    phase: str,
    campaign_id: str,
    default_agent: str,
) -> list[ReliabilityRecord]:
    run_meta = dict(getattr(getattr(eval_log, "eval", None), "metadata", {}) or {})
    samples = list(getattr(eval_log, "samples", None) or [])
    run_id = str(getattr(getattr(eval_log, "eval", None), "run_id", "") or "")
    eval_set_id = str(getattr(getattr(eval_log, "eval", None), "eval_set_id", "") or run_id)
    task_name = str(getattr(getattr(eval_log, "eval", None), "task", "") or benchmark)

    records: list[ReliabilityRecord] = []
    for sample in samples:
        sample_meta = dict(getattr(sample, "metadata", {}) or {})
        metadata = dict(run_meta)
        metadata.update(sample_meta)
        if _campaign_mismatch(metadata, campaign_id):
            continue

        phase_name = str(metadata.get("reliability_phase") or phase)
        if phase_name != phase:
            continue

        agent_name = str(metadata.get("reliability_agent") or default_agent)
        sample_uuid = str(
            metadata.get("reliability_sample_uuid")
            or getattr(sample, "uuid", "")
            or f"{run_id}:{getattr(sample, 'id', '')}"
        )
        identity = ReliabilityRunIdentity(
            eval_set_id=eval_set_id or "eval_set_unknown",
            run_id=run_id or "run_unknown",
            phase=phase,  # preserve explicit phase scope from traversal
            agent=agent_name,
            task=task_name or benchmark,
            sample_id=getattr(sample, "id", sample_uuid),
            sample_uuid=sample_uuid,
            repeat_id=_to_int(metadata.get("reliability_repeat_id"), default=0),
            sample_retry_id=_to_int(metadata.get("reliability_sample_retry_id"), default=0),
            agent_attempt_id=_to_int(metadata.get("reliability_agent_attempt_id"), default=0),
        )
        scores = getattr(sample, "scores", {}) or {}
        record = ReliabilityRecord(
            identity=identity,
            outcome=_outcome_payload_from_scores(scores),
            behavior=_behavior_payload_from_events(getattr(sample, "events", None)),
            resources=_resource_payload_from_sample(sample),
            confidence=_confidence_payload_from_eval(scores, metadata),
            safety=_safety_payload_from_eval(scores, metadata),
            abstention=_abstention_payload_from_eval(scores, metadata),
            perturbation=_prefixed_metadata(metadata, "reliability_perturbation_"),
            metadata=metadata,
        )
        records.append(record)
    return records


def _resource_summary_from_eval_logs(eval_logs: list[Any]) -> CampaignResourceSummary:
    started_at_values: list[str] = []
    completed_at_values: list[str] = []
    wall_time_total = 0.0
    wall_time_seen = False
    working_time_total = 0.0
    working_time_seen = False
    total_cost = 0.0
    cost_seen = False
    input_tokens = 0
    input_seen = False
    output_tokens = 0
    output_seen = False
    cache_read_tokens = 0
    cache_read_seen = False
    cache_write_tokens = 0
    cache_write_seen = False
    reasoning_tokens = 0
    reasoning_seen = False
    total_tokens = 0
    tokens_seen = False

    for eval_log in eval_logs:
        stats = getattr(eval_log, "stats", None)
        started_at = getattr(stats, "started_at", None)
        completed_at = getattr(stats, "completed_at", None)
        if isinstance(started_at, str) and started_at:
            started_at_values.append(started_at)
        if isinstance(completed_at, str) and completed_at:
            completed_at_values.append(completed_at)

        wall_time = _eval_wall_time_seconds(eval_log)
        if wall_time is not None:
            wall_time_total += wall_time
            wall_time_seen = True

        for sample in list(getattr(eval_log, "samples", None) or []):
            working_time = _to_float(getattr(sample, "working_time", None))
            if working_time is not None:
                working_time_total += working_time
                working_time_seen = True

        for model_name, usage in dict(getattr(stats, "model_usage", {}) or {}).items():
            input_count = getattr(usage, "input_tokens", None)
            if isinstance(input_count, int):
                input_tokens += input_count
                input_seen = True
            output_count = getattr(usage, "output_tokens", None)
            if isinstance(output_count, int):
                output_tokens += output_count
                output_seen = True
            cache_read_count = getattr(usage, "input_tokens_cache_read", None)
            if isinstance(cache_read_count, int):
                cache_read_tokens += cache_read_count
                cache_read_seen = True
            cache_write_count = getattr(usage, "input_tokens_cache_write", None)
            if isinstance(cache_write_count, int):
                cache_write_tokens += cache_write_count
                cache_write_seen = True
            reasoning_count = getattr(usage, "reasoning_tokens", None)
            if isinstance(reasoning_count, int):
                reasoning_tokens += reasoning_count
                reasoning_seen = True
            token_count = getattr(usage, "total_tokens", None)
            if isinstance(token_count, int):
                total_tokens += token_count
                tokens_seen = True
            cost = _usage_total_cost(model_name, usage)
            if cost is not None:
                total_cost += cost
                cost_seen = True

    return CampaignResourceSummary(
        started_at=min(started_at_values) if started_at_values else None,
        completed_at=max(completed_at_values) if completed_at_values else None,
        wall_time_sec=wall_time_total if wall_time_seen else None,
        working_time_sec=working_time_total if working_time_seen else None,
        total_cost_usd=total_cost if cost_seen else None,
        input_tokens=input_tokens if input_seen else None,
        output_tokens=output_tokens if output_seen else None,
        cache_read_tokens=cache_read_tokens if cache_read_seen else None,
        cache_write_tokens=cache_write_tokens if cache_write_seen else None,
        reasoning_tokens=reasoning_tokens if reasoning_seen else None,
        total_tokens=total_tokens if tokens_seen else None,
    )


def _resource_summary_from_sidecar_records(
    records: list[ReliabilityRecord],
) -> CampaignResourceSummary:
    working_time_total = 0.0
    working_time_seen = False
    for record in records:
        working_time = _to_float(record.resources.get("working_time"))
        if working_time is not None:
            working_time_total += working_time
            working_time_seen = True

    return CampaignResourceSummary(
        working_time_sec=working_time_total if working_time_seen else None,
    )


def _campaign_resource_summary(
    phase_summaries: dict[str, PhaseAnalysisSummary],
) -> CampaignResourceSummary:
    started_at_values: list[str] = []
    completed_at_values: list[str] = []
    wall_time_total = 0.0
    wall_time_seen = False
    working_time_total = 0.0
    working_time_seen = False
    total_cost = 0.0
    cost_seen = False
    input_tokens = 0
    input_seen = False
    output_tokens = 0
    output_seen = False
    cache_read_tokens = 0
    cache_read_seen = False
    cache_write_tokens = 0
    cache_write_seen = False
    reasoning_tokens = 0
    reasoning_seen = False
    total_tokens = 0
    tokens_seen = False

    for summary in phase_summaries.values():
        if summary.started_at:
            started_at_values.append(summary.started_at)
        if summary.completed_at:
            completed_at_values.append(summary.completed_at)
        if summary.wall_time_sec is not None:
            wall_time_total += summary.wall_time_sec
            wall_time_seen = True
        if summary.working_time_sec is not None:
            working_time_total += summary.working_time_sec
            working_time_seen = True
        if summary.total_cost_usd is not None:
            total_cost += summary.total_cost_usd
            cost_seen = True
        if summary.input_tokens is not None:
            input_tokens += summary.input_tokens
            input_seen = True
        if summary.output_tokens is not None:
            output_tokens += summary.output_tokens
            output_seen = True
        if summary.cache_read_tokens is not None:
            cache_read_tokens += summary.cache_read_tokens
            cache_read_seen = True
        if summary.cache_write_tokens is not None:
            cache_write_tokens += summary.cache_write_tokens
            cache_write_seen = True
        if summary.reasoning_tokens is not None:
            reasoning_tokens += summary.reasoning_tokens
            reasoning_seen = True
        if summary.total_tokens is not None:
            total_tokens += summary.total_tokens
            tokens_seen = True

    return CampaignResourceSummary(
        started_at=min(started_at_values) if started_at_values else None,
        completed_at=max(completed_at_values) if completed_at_values else None,
        wall_time_sec=wall_time_total if wall_time_seen else None,
        working_time_sec=working_time_total if working_time_seen else None,
        total_cost_usd=total_cost if cost_seen else None,
        input_tokens=input_tokens if input_seen else None,
        output_tokens=output_tokens if output_seen else None,
        cache_read_tokens=cache_read_tokens if cache_read_seen else None,
        cache_write_tokens=cache_write_tokens if cache_write_seen else None,
        reasoning_tokens=reasoning_tokens if reasoning_seen else None,
        total_tokens=total_tokens if tokens_seen else None,
    )


def _eval_wall_time_seconds(eval_log: Any) -> float | None:
    stats = getattr(eval_log, "stats", None)
    started_at = getattr(stats, "started_at", None)
    completed_at = getattr(stats, "completed_at", None)
    if not isinstance(started_at, str) or not isinstance(completed_at, str):
        return None
    if not started_at or not completed_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds())


_FALLBACK_MODEL_COSTS: dict[str, ModelCost] = {
    "gpt-5.4": ModelCost(
        input=2.5,
        output=15.0,
        input_cache_write=2.5,
        input_cache_read=0.25,
    ),
    "openai/gpt-5.4": ModelCost(
        input=2.5,
        output=15.0,
        input_cache_write=2.5,
        input_cache_read=0.25,
    ),
    "gpt-5.4-pro": ModelCost(
        input=30.0,
        output=180.0,
        input_cache_write=30.0,
        input_cache_read=3.0,
    ),
    "openai/gpt-5.4-pro": ModelCost(
        input=30.0,
        output=180.0,
        input_cache_write=30.0,
        input_cache_read=3.0,
    ),
}


def _usage_total_cost(model_name: str, usage: Any) -> float | None:
    recorded_cost = _to_float(getattr(usage, "total_cost", None))
    if recorded_cost is not None:
        return recorded_cost

    info = get_model_info(model_name)
    if info is not None and info.cost is not None:
        return compute_model_cost(info.cost, usage)

    fallback_cost = _fallback_model_cost(model_name)
    if fallback_cost is not None:
        return compute_model_cost(fallback_cost, usage)

    return None


def _fallback_model_cost(model_name: str) -> ModelCost | None:
    if model_name in _FALLBACK_MODEL_COSTS:
        return _FALLBACK_MODEL_COSTS[model_name]

    for prefix, cost in _FALLBACK_MODEL_COSTS.items():
        if model_name.startswith(f"{prefix}-"):
            return cost
    return None


def _campaign_mismatch(metadata: dict[str, Any], campaign_id: str) -> bool:
    value = metadata.get("reliability_campaign_id")
    return value is not None and str(value) != campaign_id


def _outcome_payload_from_scores(scores: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, score in scores.items():
        value = getattr(score, "value", None)
        if "raw" not in payload:
            payload["raw"] = value
        payload[name] = value
    binary = outcome_binary_from_outcome(payload)
    if binary is not None:
        payload["pass"] = bool(binary)
    return payload


def _behavior_payload_from_events(events: Any) -> dict[str, Any]:
    if not isinstance(events, list):
        return {}
    event_types = [event_type for event in events if (event_type := getattr(event, "event", None))]
    return {"event_types": event_types, "event_count": len(event_types)}


def _resource_payload_from_sample(sample: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    total_time = _to_float(getattr(sample, "total_time", None))
    working_time = _to_float(getattr(sample, "working_time", None))
    if total_time is not None:
        payload["total_time"] = total_time
    if working_time is not None:
        payload["working_time"] = working_time
    return payload


def _confidence_payload_from_eval(
    scores: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    payload = _scorer_payload(scores, "reliability_confidence_signal")
    if payload:
        return payload
    for key in ("reliability_confidence", "confidence", "self_confidence"):
        if key in metadata:
            return {"value": metadata[key], "source_key": key}
    return {}


def _safety_payload_from_eval(scores: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    payload = _scorer_payload(scores, "reliability_safety_violation")
    if payload:
        return payload
    return _prefixed_metadata(metadata, "reliability_safety_")


def _abstention_payload_from_eval(
    scores: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    payload = _scorer_payload(scores, "reliability_abstention_signal")
    if payload:
        return payload
    return _prefixed_metadata(metadata, "reliability_abstention_")


def _scorer_payload(scores: dict[str, Any], scorer_name: str) -> dict[str, Any]:
    scorer = scores.get(scorer_name)
    if scorer is None:
        return {}
    value = getattr(scorer, "value", None)
    metadata = dict(getattr(scorer, "metadata", {}) or {})
    if "value" not in metadata:
        metadata["value"] = value
    return metadata


def _prefixed_metadata(metadata: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key.startswith(prefix)}


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _flatten_phase_records(
    phase_records: dict[str, list[ReliabilityRecord]],
) -> list[ReliabilityRecord]:
    flattened: list[ReliabilityRecord] = []
    for records in phase_records.values():
        flattened.extend(records)
    return flattened


def _robustness_metrics(
    phase_summaries: dict[str, PhaseAnalysisSummary],
) -> RobustnessMetrics:
    baseline = phase_summaries.get("baseline")
    fault = phase_summaries.get("fault")
    prompt = phase_summaries.get("prompt")
    structural = phase_summaries.get("structural")
    baseline_accuracy = baseline.accuracy if baseline else None
    return RobustnessMetrics(
        baseline_accuracy=baseline_accuracy,
        fault_accuracy=fault.accuracy if fault else None,
        prompt_accuracy=prompt.accuracy if prompt else None,
        structural_accuracy=structural.accuracy if structural else None,
        fault_delta_vs_baseline=_delta(fault.accuracy if fault else None, baseline_accuracy),
        prompt_delta_vs_baseline=_delta(
            prompt.accuracy if prompt else None, baseline_accuracy
        ),
        structural_delta_vs_baseline=_delta(
            structural.accuracy if structural else None, baseline_accuracy
        ),
    )


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _predictability_metrics(records: list[ReliabilityRecord]) -> PredictabilityMetrics:
    pairs: list[tuple[float, int]] = []
    for record in records:
        confidence = _confidence_value(record)
        outcome = outcome_binary_from_outcome(record.outcome)
        if confidence is None or outcome is None:
            continue
        clamped = max(0.0, min(1.0, confidence))
        pairs.append((clamped, outcome))

    if not pairs:
        return PredictabilityMetrics(pair_count=0)

    brier_mse = fmean((confidence - outcome) ** 2 for confidence, outcome in pairs)
    return PredictabilityMetrics(
        pair_count=len(pairs),
        brier_mse=brier_mse,
        brier_predictability=1.0 - brier_mse,
        calibration_error=_expected_calibration_error(pairs),
        discrimination_auroc=_strict_discrimination_auc(pairs),
    )


def _expected_calibration_error(
    pairs: list[tuple[float, int]], bins: int = 10
) -> float | None:
    if not pairs:
        return None
    bucketed: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for confidence, outcome in pairs:
        idx = min(bins - 1, int(confidence * bins))
        bucketed[idx].append((confidence, outcome))
    total = len(pairs)
    weighted_error = 0.0
    for bucket in bucketed.values():
        conf_avg = fmean(confidence for confidence, _ in bucket)
        outcome_avg = fmean(float(outcome) for _, outcome in bucket)
        weighted_error += abs(conf_avg - outcome_avg) * (len(bucket) / total)
    return weighted_error


def _strict_discrimination_auc(pairs: list[tuple[float, int]]) -> float | None:
    successes = [confidence for confidence, outcome in pairs if outcome == 1]
    failures = [confidence for confidence, outcome in pairs if outcome == 0]
    if not successes or not failures:
        return None
    failures_sorted = sorted(failures)
    wins = sum(bisect_left(failures_sorted, confidence) for confidence in successes)
    return wins / (len(successes) * len(failures))


def _safety_metrics(records: list[ReliabilityRecord]) -> SafetyMetrics:
    observed = 0
    violations = 0
    for record in records:
        signal = _extract_safety_violation(record)
        if signal is None:
            continue
        observed += 1
        if signal:
            violations += 1
    return SafetyMetrics(
        observed_records=observed,
        violation_count=violations,
        violation_rate=(violations / observed) if observed > 0 else None,
    )


def _abstention_metrics(records: list[ReliabilityRecord]) -> AbstentionMetrics:
    observed = 0
    abstentions = 0
    non_abstained_outcomes: list[float] = []
    for record in records:
        signal = _extract_abstention(record)
        if signal is None:
            continue
        observed += 1
        if signal:
            abstentions += 1
            continue
        outcome = outcome_binary_from_outcome(record.outcome)
        if outcome is not None:
            non_abstained_outcomes.append(float(outcome))
    return AbstentionMetrics(
        observed_records=observed,
        abstention_count=abstentions,
        abstention_rate=(abstentions / observed) if observed > 0 else None,
        selective_accuracy=fmean(non_abstained_outcomes)
        if non_abstained_outcomes
        else None,
    )


def _extract_safety_violation(record: ReliabilityRecord) -> bool | None:
    safety = dict(record.safety)
    for key, value in safety.items():
        lowered = key.lower()
        bool_value = _coerce_bool(value)
        if bool_value is None:
            continue
        if "violation" in lowered or "unsafe" in lowered or "harm" in lowered:
            return bool_value
        if lowered.endswith("safe"):
            return not bool_value
    return None


def _extract_abstention(record: ReliabilityRecord) -> bool | None:
    abstention = dict(record.abstention)
    for key, value in abstention.items():
        lowered = key.lower()
        bool_value = _coerce_bool(value)
        if bool_value is None:
            continue
        if "abstain" in lowered or "refus" in lowered or "uncertain" in lowered:
            return bool_value
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "y", "1"}:
            return True
        if token in {"false", "no", "n", "0"}:
            return False
    return None
