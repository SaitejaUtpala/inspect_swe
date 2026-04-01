from __future__ import annotations

from pathlib import Path

from inspect_swe.reliability.analysis import (
    analyze_baseline_campaign,
    analyze_reliability_campaign,
)
from inspect_swe.reliability.artifacts import ReliabilityRecord, SidecarWriter
from inspect_swe.reliability.identity import ReliabilityRunIdentity
from inspect_swe.reliability.paths import campaign_sidecar_path


def test_analyze_baseline_campaign_computes_core_metrics(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "baseline_records.jsonl"
    writer = SidecarWriter(sidecar_path)

    records = [
        _record(
            run_id="run-0",
            sample_id="sample-a",
            repeat_id=0,
            gaia_scorer="C",
            event_types=["ModelEvent", "ScoreEvent"],
            total_time=10.0,
            confidence=0.8,
        ),
        _record(
            run_id="run-1",
            sample_id="sample-a",
            repeat_id=1,
            gaia_scorer="C",
            event_types=["ModelEvent", "ScoreEvent"],
            total_time=12.0,
            confidence=0.7,
        ),
        _record(
            run_id="run-0",
            sample_id="sample-b",
            repeat_id=0,
            gaia_scorer="C",
            event_types=["ModelEvent"],
            total_time=20.0,
            confidence=0.2,
        ),
        _record(
            run_id="run-1",
            sample_id="sample-b",
            repeat_id=1,
            gaia_scorer="I",
            event_types=["ScoreEvent"],
            total_time=40.0,
            confidence=0.9,
        ),
    ]
    for record in records:
        writer.write(record)

    result = analyze_baseline_campaign(
        sidecar_path=str(sidecar_path),
        benchmark="gaia_level1",
        campaign_id="campaign_a",
        agent="codex_cli",
    )

    assert result.total_records == 4
    assert result.sample_count == 2
    assert result.repeats == [0, 1]
    assert result.records_per_repeat == {"0": 2, "1": 2}
    assert result.accuracy == 0.75
    assert result.consistency.outcome == 0.75
    assert result.consistency.trajectory_distribution == 0.5
    assert result.consistency.trajectory_sequence == 0.5
    assert result.consistency.confidence is not None
    assert result.consistency.resource is not None
    assert result.resources.total_time_mean_sec == 20.5
    assert result.resources.total_time_min_sec == 10.0
    assert result.resources.total_time_max_sec == 40.0


def _record(
    *,
    run_id: str,
    sample_id: str,
    repeat_id: int,
    gaia_scorer: str,
    event_types: list[str],
    total_time: float,
    confidence: float,
) -> ReliabilityRecord:
    return ReliabilityRecord(
        identity=ReliabilityRunIdentity(
            eval_set_id="no_eval_set",
            run_id=run_id,
            phase="baseline",
            agent="codex_cli",
            task="inspect_evals/gaia_level1",
            sample_id=sample_id,
            sample_uuid=f"{run_id}-{sample_id}",
            repeat_id=repeat_id,
            sample_retry_id=0,
            agent_attempt_id=0,
        ),
        outcome={"gaia_scorer": gaia_scorer},
        behavior={"event_types": event_types},
        resources={"total_time": total_time},
        confidence={"value": confidence},
        metadata={"reliability_campaign_id": "campaign_a"},
    )


def test_analyze_reliability_campaign_computes_cross_phase_metrics(
    tmp_path: Path,
) -> None:
    benchmark = "gaia_level1"
    campaign_id = "campaign_z"
    log_root = str(tmp_path)

    baseline_path = campaign_sidecar_path(
        log_root=log_root,
        benchmark=benchmark,
        phase="baseline",
        campaign_id=campaign_id,
        sidecar_filename="records.jsonl",
    )
    fault_path = campaign_sidecar_path(
        log_root=log_root,
        benchmark=benchmark,
        phase="fault",
        campaign_id=campaign_id,
        sidecar_filename="records.jsonl",
    )
    prompt_path = campaign_sidecar_path(
        log_root=log_root,
        benchmark=benchmark,
        phase="prompt",
        campaign_id=campaign_id,
        sidecar_filename="records.jsonl",
    )
    structural_path = campaign_sidecar_path(
        log_root=log_root,
        benchmark=benchmark,
        phase="structural",
        campaign_id=campaign_id,
        sidecar_filename="records.jsonl",
    )

    baseline_writer = SidecarWriter(baseline_path)
    baseline_writer.write(
        _campaign_record(
            phase="baseline",
            run_id="base-run-0",
            sample_id="a",
            repeat_id=0,
            passed=True,
            confidence=0.9,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )
    baseline_writer.write(
        _campaign_record(
            phase="baseline",
            run_id="base-run-1",
            sample_id="a",
            repeat_id=1,
            passed=True,
            confidence=0.8,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )
    baseline_writer.write(
        _campaign_record(
            phase="baseline",
            run_id="base-run-0",
            sample_id="b",
            repeat_id=0,
            passed=False,
            confidence=0.3,
            safety_violation=True,
            abstained=True,
            campaign_id=campaign_id,
        )
    )
    baseline_writer.write(
        _campaign_record(
            phase="baseline",
            run_id="base-run-1",
            sample_id="b",
            repeat_id=1,
            passed=False,
            confidence=0.2,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )

    fault_writer = SidecarWriter(fault_path)
    fault_writer.write(
        _campaign_record(
            phase="fault",
            run_id="fault-run-0",
            sample_id="a",
            repeat_id=0,
            passed=False,
            confidence=None,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )
    fault_writer.write(
        _campaign_record(
            phase="fault",
            run_id="fault-run-1",
            sample_id="b",
            repeat_id=1,
            passed=False,
            confidence=None,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )

    prompt_writer = SidecarWriter(prompt_path)
    prompt_writer.write(
        _campaign_record(
            phase="prompt",
            run_id="prompt-run-0",
            sample_id="a",
            repeat_id=0,
            passed=True,
            confidence=None,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )
    prompt_writer.write(
        _campaign_record(
            phase="prompt",
            run_id="prompt-run-1",
            sample_id="b",
            repeat_id=1,
            passed=True,
            confidence=None,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )

    structural_writer = SidecarWriter(structural_path)
    structural_writer.write(
        _campaign_record(
            phase="structural",
            run_id="struct-run-0",
            sample_id="a",
            repeat_id=0,
            passed=True,
            confidence=None,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )
    structural_writer.write(
        _campaign_record(
            phase="structural",
            run_id="struct-run-1",
            sample_id="b",
            repeat_id=1,
            passed=False,
            confidence=None,
            safety_violation=False,
            abstained=False,
            campaign_id=campaign_id,
        )
    )

    result = analyze_reliability_campaign(
        log_root=log_root,
        benchmark=benchmark,
        campaign_id=campaign_id,
        agent="codex_cli",
    )

    assert result.phase_summaries["baseline"].accuracy == 0.5
    assert result.phase_summaries["fault"].accuracy == 0.0
    assert result.phase_summaries["prompt"].accuracy == 1.0
    assert result.phase_summaries["structural"].accuracy == 0.5
    assert result.robustness.fault_delta_vs_baseline == -0.5
    assert result.robustness.prompt_delta_vs_baseline == 0.5
    assert result.robustness.structural_delta_vs_baseline == 0.0
    assert result.predictability.pair_count == 4
    assert result.predictability.brier_mse is not None
    assert result.safety.violation_count == 1
    assert result.abstention.abstention_count == 1
    assert result.abstention.selective_accuracy is not None


def _campaign_record(
    *,
    phase: str,
    run_id: str,
    sample_id: str,
    repeat_id: int,
    passed: bool,
    confidence: float | None,
    safety_violation: bool,
    abstained: bool,
    campaign_id: str,
) -> ReliabilityRecord:
    confidence_payload = {"value": confidence} if confidence is not None else {}
    return ReliabilityRecord(
        identity=ReliabilityRunIdentity(
            eval_set_id="no_eval_set",
            run_id=run_id,
            phase=phase,  # type: ignore[arg-type]
            agent="codex_cli",
            task="inspect_evals/gaia_level1",
            sample_id=sample_id,
            sample_uuid=f"{run_id}-{sample_id}",
            repeat_id=repeat_id,
            sample_retry_id=0,
            agent_attempt_id=0,
        ),
        outcome={"pass": passed},
        behavior={"event_types": ["ModelEvent", "ScoreEvent"]},
        resources={"total_time": 10.0},
        confidence=confidence_payload,
        safety={"violation": safety_violation},
        abstention={"abstained": abstained},
        metadata={"reliability_campaign_id": campaign_id},
    )
