from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from inspect_ai.log import EvalLog
from inspect_swe.reliability import (
    FaultExecutionError,
    FaultPerturbationSpec,
    FaultPhaseConfig,
    ReliabilitySpec,
)
from inspect_swe.reliability.fault import run_fault_phase
from inspect_swe.reliability.telemetry import TelemetryCoverageReport


class _EvalCallRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[EvalLog]:
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        log = cast(
            EvalLog,
            SimpleNamespace(
                location=f"/tmp/logs/fault-run-{idx}.eval",
                eval=SimpleNamespace(run_id=f"fault-run-{idx}"),
            ),
        )
        return [log]


def test_run_fault_phase_executes_repeats_and_sets_fault_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _EvalCallRecorder()
    monkeypatch.setattr("inspect_swe.reliability.fault.eval", recorder)
    monkeypatch.setattr(
        "inspect_swe.reliability.fault.preflight_reliability_spec", lambda spec: None
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.fault._assess_repeat_coverage",
        lambda **kwargs: TelemetryCoverageReport(
            expected_samples=1,
            observed_records=1,
            missing_sample_uuids=[],
            duplicate_identity_keys=[],
            identity_warning_count=0,
        ),
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.fault._default_solver_for_agent",
        lambda agent, generate_filter, bridged_tools: None,
    )

    spec = ReliabilitySpec(
        benchmark="gaia_level1",
        agents=["codex_cli"],
        phases=["fault"],
        fail_on_missing_hooks=False,
    )
    result = run_fault_phase(
        spec=spec,
        tasks="inspect_evals/gaia_level1",
        config=FaultPhaseConfig(
            repeats=2,
            campaign_id="fault_campaign_a",
            log_root=str(tmp_path),
            perturbation=FaultPerturbationSpec(
                enabled=True,
                mode="noop",
                target="model",
                fault_rate=0.2,
                max_faults_per_sample=2,
                policy_name="fault_policy_test",
            ),
        ),
    )

    assert len(recorder.calls) == 2
    first_call = recorder.calls[0]
    assert first_call["log_format"] == "eval"
    assert first_call["metadata"]["reliability_phase"] == "fault"
    assert first_call["metadata"]["reliability_perturbation_type"] == "fault"
    assert first_call["metadata"]["reliability_perturbation_fault_mode"] == "noop"
    assert first_call["metadata"]["reliability_perturbation_fault_rate"] == 0.2
    assert "fault_campaign_a" in first_call["log_dir"]
    assert first_call["log_dir"].endswith(
        "gaia_level1/codex_cli/"
        "gaia_level1_codex_cli_fault_rep_000_fault_campaign_a"
    )
    assert result.campaign_id == "fault_campaign_a"
    assert "fault_campaign_a" in result.sidecar_path
    assert result.perturbation.policy_name == "fault_policy_test"


def test_run_fault_phase_fails_on_incomplete_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _EvalCallRecorder()
    monkeypatch.setattr("inspect_swe.reliability.fault.eval", recorder)
    monkeypatch.setattr(
        "inspect_swe.reliability.fault.preflight_reliability_spec", lambda spec: None
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.fault._assess_repeat_coverage",
        lambda **kwargs: TelemetryCoverageReport(
            expected_samples=1,
            observed_records=0,
            missing_sample_uuids=["uuid-a"],
            duplicate_identity_keys=[],
            identity_warning_count=0,
        ),
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.fault._default_solver_for_agent",
        lambda agent, generate_filter, bridged_tools: None,
    )

    spec = ReliabilitySpec(
        benchmark="gaia_level1",
        agents=["codex_cli"],
        phases=["fault"],
        fail_on_missing_hooks=False,
    )
    with pytest.raises(FaultExecutionError):
        run_fault_phase(
            spec=spec,
            tasks="inspect_evals/gaia_level1",
            config=FaultPhaseConfig(
                repeats=1,
                log_root=str(tmp_path),
                fail_on_incomplete_telemetry=True,
            ),
        )
