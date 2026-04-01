from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from inspect_ai.log import EvalLog
from inspect_swe.reliability import (
    ReliabilitySpec,
    StructuralExecutionError,
    StructuralPerturbationSpec,
    StructuralPhaseConfig,
)
from inspect_swe.reliability.structural import run_structural_phase
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
                location=f"/tmp/logs/structural-run-{idx}.eval",
                eval=SimpleNamespace(run_id=f"structural-run-{idx}"),
            ),
        )
        return [log]


def test_run_structural_phase_executes_repeats_and_sets_structural_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _EvalCallRecorder()
    monkeypatch.setattr("inspect_swe.reliability.structural.eval", recorder)
    monkeypatch.setattr(
        "inspect_swe.reliability.structural.preflight_reliability_spec",
        lambda spec: None,
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.structural._assess_repeat_coverage",
        lambda **kwargs: TelemetryCoverageReport(
            expected_samples=1,
            observed_records=1,
            missing_sample_uuids=[],
            duplicate_identity_keys=[],
            identity_warning_count=0,
        ),
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.structural._default_solver_for_agent",
        lambda agent, bridged_tools: None,
    )

    spec = ReliabilitySpec(
        benchmark="gaia_level1",
        agents=["codex_cli"],
        phases=["structural"],
        fail_on_missing_hooks=False,
    )
    result = run_structural_phase(
        spec=spec,
        tasks="inspect_evals/gaia_level1",
        config=StructuralPhaseConfig(
            repeats=2,
            campaign_id="structural_campaign_a",
            log_root=str(tmp_path),
            perturbation=StructuralPerturbationSpec(
                enabled=True,
                mode="noop",
                target="tool_response",
                policy_name="structural_policy_test",
            ),
        ),
    )

    assert len(recorder.calls) == 2
    first_call = recorder.calls[0]
    assert first_call["log_format"] == "eval"
    assert first_call["metadata"]["reliability_phase"] == "structural"
    assert first_call["metadata"]["reliability_perturbation_type"] == "structural"
    assert (
        first_call["metadata"]["reliability_perturbation_structural_mode"] == "noop"
    )
    assert (
        first_call["metadata"]["reliability_perturbation_structural_target"]
        == "tool_response"
    )
    assert first_call["log_dir"].endswith(
        "gaia_level1/codex_cli/"
        "gaia_level1_codex_cli_structural_rep_000_structural_campaign_a"
    )
    assert result.campaign_id == "structural_campaign_a"
    assert "structural_campaign_a" in result.sidecar_path
    assert result.perturbation.policy_name == "structural_policy_test"


def test_run_structural_phase_fails_on_incomplete_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = _EvalCallRecorder()
    monkeypatch.setattr("inspect_swe.reliability.structural.eval", recorder)
    monkeypatch.setattr(
        "inspect_swe.reliability.structural.preflight_reliability_spec",
        lambda spec: None,
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.structural._assess_repeat_coverage",
        lambda **kwargs: TelemetryCoverageReport(
            expected_samples=1,
            observed_records=0,
            missing_sample_uuids=["uuid-a"],
            duplicate_identity_keys=[],
            identity_warning_count=0,
        ),
    )
    monkeypatch.setattr(
        "inspect_swe.reliability.structural._default_solver_for_agent",
        lambda agent, bridged_tools: None,
    )

    spec = ReliabilitySpec(
        benchmark="gaia_level1",
        agents=["codex_cli"],
        phases=["structural"],
        fail_on_missing_hooks=False,
    )
    with pytest.raises(StructuralExecutionError):
        run_structural_phase(
            spec=spec,
            tasks="inspect_evals/gaia_level1",
            config=StructuralPhaseConfig(
                repeats=1,
                log_root=str(tmp_path),
                fail_on_incomplete_telemetry=True,
            ),
        )
