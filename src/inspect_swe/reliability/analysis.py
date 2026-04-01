"""Baseline reliability analysis over sidecar records."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from statistics import fmean, pstdev
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from .artifacts import ReliabilityRecord, load_sidecar_records
from .outcome import outcome_binary_from_outcome, outcome_label_from_outcome
from .paths import campaign_sidecar_path


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
    total_records: int
    sample_count: int
    repeat_count: int
    accuracy: float | None = None
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
    notes: list[str] = Field(default_factory=list)


def analyze_reliability_campaign(
    *,
    log_root: str,
    benchmark: str,
    campaign_id: str,
    agent: str | None = None,
) -> CampaignAnalysisResult:
    """Analyze a campaign across baseline/fault/prompt/structural sidecars."""
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
            total_records=len(records),
            sample_count=len(sample_keys),
            repeat_count=len(repeats),
            accuracy=fmean(outcome_values) if outcome_values else None,
            agents=agents,
        )

    baseline_records = phase_records.get("baseline", [])
    robustness = _robustness_metrics(phase_summaries)
    predictability = _predictability_metrics(baseline_records)
    safety = _safety_metrics(_flatten_phase_records(phase_records))
    abstention = _abstention_metrics(_flatten_phase_records(phase_records))

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
        notes=notes,
    )


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
