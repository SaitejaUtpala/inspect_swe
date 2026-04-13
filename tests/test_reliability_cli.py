from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import inspect_swe.reliability.cli as cli
from inspect_swe.reliability import (
    BaselineAnalysisResult,
    BaselinePhaseResult,
    BaselineRepeatResult,
    CampaignAnalysisResult,
    CampaignPhaseEntry,
    ConsistencyMetrics,
    FaultPhaseResult,
    FaultRepeatResult,
    PromptPhaseResult,
    PromptRepeatResult,
    ReliabilityCampaignResult,
    ResourceMetrics,
    StructuralPhaseResult,
    StructuralRepeatResult,
)


def test_cli_baseline_parses_args_and_invokes_runner(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_baseline_phase(*, spec: Any, tasks: Any, config: Any) -> BaselinePhaseResult:
        captured["spec"] = spec
        captured["tasks"] = tasks
        captured["config"] = config
        return BaselinePhaseResult(
            benchmark=spec.benchmark,
            repeats=config.repeats,
            campaign_id="campaign-test",
            sidecar_path=str(tmp_path / "records.jsonl"),
            results=[
                BaselineRepeatResult(
                    agent="codex_cli",
                    repeat_id=0,
                    run_ids=["run-1"],
                    log_paths=["logs/run-1.eval"],
                    coverage_complete=True,
                    missing_sample_uuids=[],
                    duplicate_identity_keys=[],
                    identity_warning_count=0,
                )
            ],
        )

    monkeypatch.setattr(cli, "run_baseline_phase", _fake_run_baseline_phase)

    code = cli.main(
        [
            "baseline",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--agent",
            "codex_cli",
            "--agent",
            "gemini_cli",
            "--repeats",
            "2",
            "--seed",
            "42",
            "--model",
            "openai/gpt-5.4-2026-03-05",
            "--task-arg",
            "attempts=2",
            "--task-arg",
            "mode=smoke",
            "--metadata",
            "team=reliability",
            "--metadata",
            "dry_run=true",
            "--limit",
            "10-12",
            "--sample-id",
            "1,abc",
            "--log-root",
            "logs/custom",
            "--campaign-id",
            "run-alpha",
            "--sidecar-dir",
            "custom_sidecars",
            "--orchestrator-mode",
            "single_process",
            "--orchestrator-workers",
            "1",
            "--max-samples",
            "1",
            "--max-subprocesses",
            "2",
            "--max-sandboxes",
            "1",
            "--max-connections",
            "5",
            "--no-fail-on-missing-hooks",
            "--no-strict-identity-tags",
            "--no-fail-on-incomplete-telemetry",
        ]
    )

    assert code == 0
    assert captured["tasks"] == "inspect_evals/gaia_level1"
    assert captured["spec"].agents == ["codex_cli", "gemini_cli"]
    assert captured["spec"].fail_on_missing_hooks is False
    assert captured["spec"].strict_identity_tags is False
    assert captured["spec"].concurrency.orchestrator_mode == "single_process"
    assert captured["spec"].concurrency.orchestrator_workers == 1
    assert captured["spec"].concurrency.max_samples == 1
    assert captured["spec"].concurrency.max_subprocesses == 2
    assert captured["spec"].concurrency.max_sandboxes == 1
    assert captured["spec"].concurrency.max_connections == 5
    assert captured["config"].repeats == 2
    assert captured["config"].campaign_id == "run-alpha"
    assert captured["config"].limit == (10, 12)
    assert captured["config"].sample_id == [1, "abc"]
    assert captured["config"].task_args == {"attempts": 2, "mode": "smoke"}
    assert captured["config"].metadata == {"team": "reliability", "dry_run": True}
    assert captured["config"].model == "openai/gpt-5.4-2026-03-05"

    out = capsys.readouterr().out
    assert "Baseline reliability run complete" in out
    assert "coverage=complete" in out


def test_cli_baseline_json_output(monkeypatch: Any, capsys: Any, tmp_path: Path) -> None:
    def _fake_run_baseline_phase(*, spec: Any, tasks: Any, config: Any) -> BaselinePhaseResult:
        return BaselinePhaseResult(
            benchmark=spec.benchmark,
            repeats=config.repeats,
            campaign_id="campaign-test",
            sidecar_path=str(tmp_path / "records.jsonl"),
            results=[],
        )

    monkeypatch.setattr(cli, "run_baseline_phase", _fake_run_baseline_phase)

    code = cli.main(
        [
            "baseline",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark"] == "gaia_level1"
    assert payload["repeats"] == 5
    assert payload["results"] == []


def test_cli_baseline_rejects_bad_key_value_pair(capsys: Any) -> None:
    code = cli.main(
        [
            "baseline",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--metadata",
            "missing_separator",
        ]
    )

    assert code == 2
    assert "invalid arguments: expected KEY=VALUE pair" in capsys.readouterr().err


def test_cli_analyze_uses_default_campaign_sidecar_path(
    monkeypatch: Any, capsys: Any
) -> None:
    captured: dict[str, Any] = {}

    def _fake_analyze_baseline_campaign(
        *, sidecar_path: str, benchmark: str | None, campaign_id: str | None, agent: str | None
    ) -> BaselineAnalysisResult:
        captured["sidecar_path"] = sidecar_path
        captured["benchmark"] = benchmark
        captured["campaign_id"] = campaign_id
        captured["agent"] = agent
        return BaselineAnalysisResult(
            benchmark=benchmark,
            campaign_id=campaign_id,
            agent=agent,
            sidecar_path=sidecar_path,
            total_records=15,
            unique_runs=3,
            repeats=[0, 1, 2],
            records_per_repeat={"0": 5, "1": 5, "2": 5},
            sample_count=5,
            outcome_label_counts={"C": 10, "I": 5},
            accuracy=0.667,
            consistency=ConsistencyMetrics(
                outcome=0.867,
                trajectory_distribution=0.842,
                trajectory_sequence=0.651,
                confidence=None,
                resource=0.741,
            ),
            resources=ResourceMetrics(
                total_time_mean_sec=91.17,
                total_time_stddev_sec=55.18,
                total_time_min_sec=34.64,
                total_time_max_sec=242.99,
            ),
            notes=["confidence consistency unavailable; no numeric confidence values found"],
        )

    monkeypatch.setattr(cli, "analyze_baseline_campaign", _fake_analyze_baseline_campaign)

    code = cli.main(
        [
            "analyze",
            "--benchmark",
            "gaia_level1",
            "--campaign-id",
            "gaia_k3_ms3_r3_run1",
            "--agent",
            "codex_cli",
            "--log-root",
            "logs/reliability",
        ]
    )

    assert code == 0
    assert (
        captured["sidecar_path"]
        == "logs/reliability/gaia_k3_ms3_r3_run1/gaia_level1/gaia_level1_baseline_gaia_k3_ms3_r3_run1_records.jsonl"
    )
    assert captured["benchmark"] == "gaia_level1"
    assert captured["campaign_id"] == "gaia_k3_ms3_r3_run1"
    assert captured["agent"] == "codex_cli"

    out = capsys.readouterr().out
    assert "Accuracy: 0.667" in out
    assert "consistency_outcome: 0.867" in out


def test_cli_analyze_json_output(monkeypatch: Any, capsys: Any) -> None:
    def _fake_analyze_baseline_campaign(
        *, sidecar_path: str, benchmark: str | None, campaign_id: str | None, agent: str | None
    ) -> BaselineAnalysisResult:
        return BaselineAnalysisResult(
            benchmark=benchmark,
            campaign_id=campaign_id,
            agent=agent,
            sidecar_path=sidecar_path,
            total_records=1,
            unique_runs=1,
            repeats=[0],
            records_per_repeat={"0": 1},
            sample_count=1,
            outcome_label_counts={"C": 1},
            accuracy=1.0,
            consistency=ConsistencyMetrics(outcome=1.0),
            resources=ResourceMetrics(),
        )

    monkeypatch.setattr(cli, "analyze_baseline_campaign", _fake_analyze_baseline_campaign)

    code = cli.main(
        [
            "analyze",
            "--sidecar-path",
            "/tmp/baseline_records.jsonl",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["phase"] == "baseline"
    assert payload["accuracy"] == 1.0


def test_cli_analyze_all_phases_uses_campaign_analyzer(
    monkeypatch: Any, capsys: Any
) -> None:
    captured: dict[str, Any] = {}

    def _fake_analyze_reliability_campaign(
        *,
        log_root: str,
        benchmark: str,
        campaign_id: str,
        agent: str | None,
        source: str,
    ) -> CampaignAnalysisResult:
        captured["log_root"] = log_root
        captured["benchmark"] = benchmark
        captured["campaign_id"] = campaign_id
        captured["agent"] = agent
        captured["source"] = source
        return CampaignAnalysisResult(
            benchmark=benchmark,
            campaign_id=campaign_id,
            agent=agent,
        )

    monkeypatch.setattr(
        cli, "analyze_reliability_campaign", _fake_analyze_reliability_campaign
    )

    code = cli.main(
        [
            "analyze",
            "--all-phases",
            "--benchmark",
            "gaia_level1",
            "--campaign-id",
            "campaign_all",
            "--agent",
            "codex_cli",
            "--log-root",
            "logs/reliability",
        ]
    )

    assert code == 0
    assert captured == {
        "log_root": "logs/reliability",
        "benchmark": "gaia_level1",
        "campaign_id": "campaign_all",
        "agent": "codex_cli",
        "source": "eval_preferred",
    }
    out = capsys.readouterr().out
    assert "Campaign analysis complete" in out


def test_cli_campaign_parses_args_and_invokes_runner(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_reliability_campaign(
        *, spec: Any, tasks: Any, config: Any
    ) -> ReliabilityCampaignResult:
        captured["spec"] = spec
        captured["tasks"] = tasks
        captured["config"] = config
        return ReliabilityCampaignResult(
            benchmark=spec.benchmark,
            campaign_id=config.campaign_id or "campaign_default",
            manifest_path=str(tmp_path / "manifest.json"),
            phase_entries={
                "baseline": CampaignPhaseEntry(
                    status="completed",
                    sidecar_path="logs/reliability/baseline_records.jsonl",
                )
            },
            analysis_json_path=str(tmp_path / "analysis.json"),
            report_markdown_path=str(tmp_path / "report.md"),
        )

    monkeypatch.setattr(cli, "run_reliability_campaign", _fake_run_reliability_campaign)

    code = cli.main(
        [
            "campaign",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--phase",
            "baseline",
            "--phase",
            "prompt",
            "--campaign-id",
            "campaign_a",
            "--repeats",
            "3",
            "--max-samples",
            "2",
            "--prompt-mode",
            "rewrite_v1",
            "--fault-mode",
            "noop",
            "--structural-mode",
            "key_case_flip",
        ]
    )

    assert code == 0
    assert captured["tasks"] == "inspect_evals/gaia_level1"
    assert captured["spec"].phases == ["baseline", "prompt"]
    assert captured["config"].campaign_id == "campaign_a"
    assert captured["config"].repeats == 3
    assert captured["spec"].concurrency.max_samples == 2
    out = capsys.readouterr().out
    assert "Reliability campaign complete" in out
    assert "Manifest:" in out


def test_cli_fault_parses_args_and_invokes_runner(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_fault_phase(*, spec: Any, tasks: Any, config: Any) -> FaultPhaseResult:
        captured["spec"] = spec
        captured["tasks"] = tasks
        captured["config"] = config
        return FaultPhaseResult(
            benchmark=spec.benchmark,
            repeats=config.repeats,
            campaign_id=config.campaign_id or "fault_campaign",
            sidecar_path=str(tmp_path / "fault_records.jsonl"),
            perturbation=config.perturbation,
            results=[
                FaultRepeatResult(
                    agent="codex_cli",
                    repeat_id=0,
                    run_ids=["fault-run-1"],
                    log_paths=["logs/fault-run-1.eval"],
                    coverage_complete=True,
                    missing_sample_uuids=[],
                    duplicate_identity_keys=[],
                    identity_warning_count=0,
                )
            ],
        )

    monkeypatch.setattr(cli, "run_fault_phase", _fake_run_fault_phase)

    code = cli.main(
        [
            "fault",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--repeats",
            "2",
            "--campaign-id",
            "fault_campaign_a",
            "--fault-mode",
            "noop",
            "--fault-rate",
            "0.3",
            "--fault-max-per-sample",
            "2",
            "--fault-policy-name",
            "fault_policy_test",
            "--fault-seed",
            "17",
            "--max-samples",
            "3",
        ]
    )

    assert code == 0
    assert captured["tasks"] == "inspect_evals/gaia_level1"
    assert captured["spec"].phases == ["fault"]
    assert captured["config"].campaign_id == "fault_campaign_a"
    assert captured["config"].perturbation.mode == "noop"
    assert captured["config"].perturbation.fault_rate == 0.3
    assert captured["config"].perturbation.max_faults_per_sample == 2
    assert captured["config"].perturbation.policy_name == "fault_policy_test"
    assert captured["config"].perturbation.seed == 17

    out = capsys.readouterr().out
    assert "Fault reliability run complete" in out
    assert "Fault policy:" in out


def test_cli_prompt_parses_args_and_invokes_runner(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_prompt_phase(*, spec: Any, tasks: Any, config: Any) -> PromptPhaseResult:
        captured["spec"] = spec
        captured["tasks"] = tasks
        captured["config"] = config
        return PromptPhaseResult(
            benchmark=spec.benchmark,
            repeats=config.repeats,
            campaign_id=config.campaign_id or "prompt_campaign",
            sidecar_path=str(tmp_path / "prompt_records.jsonl"),
            perturbation=config.perturbation,
            results=[
                PromptRepeatResult(
                    agent="codex_cli",
                    repeat_id=0,
                    run_ids=["prompt-run-1"],
                    log_paths=["logs/prompt-run-1.eval"],
                    coverage_complete=True,
                    missing_sample_uuids=[],
                    duplicate_identity_keys=[],
                    identity_warning_count=0,
                )
            ],
        )

    monkeypatch.setattr(cli, "run_prompt_phase", _fake_run_prompt_phase)

    code = cli.main(
        [
            "prompt",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--repeats",
            "2",
            "--campaign-id",
            "prompt_campaign_a",
            "--prompt-mode",
            "noop",
            "--prompt-variant-count",
            "3",
            "--prompt-policy-name",
            "prompt_policy_test",
            "--prompt-seed",
            "5",
            "--prompt-rewrite-model",
            "openai/gpt-4o-mini",
            "--prompt-rewrite-temperature",
            "0.1",
            "--prompt-rewrite-max-tokens",
            "256",
            "--prompt-rewrite-strength",
            "strong",
            "--max-samples",
            "3",
        ]
    )

    assert code == 0
    assert captured["tasks"] == "inspect_evals/gaia_level1"
    assert captured["spec"].phases == ["prompt"]
    assert captured["config"].campaign_id == "prompt_campaign_a"
    assert captured["config"].perturbation.mode == "noop"
    assert captured["config"].perturbation.variant_count == 3
    assert captured["config"].perturbation.policy_name == "prompt_policy_test"
    assert captured["config"].perturbation.seed == 5
    assert captured["config"].perturbation.rewrite_model == "openai/gpt-4o-mini"
    assert captured["config"].perturbation.rewrite_temperature == 0.1
    assert captured["config"].perturbation.rewrite_max_tokens == 256
    assert captured["config"].perturbation.rewrite_strength == "strong"

    out = capsys.readouterr().out
    assert "Prompt reliability run complete" in out
    assert "Prompt policy:" in out


def test_cli_structural_parses_args_and_invokes_runner(
    monkeypatch: Any, capsys: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_structural_phase(
        *, spec: Any, tasks: Any, config: Any
    ) -> StructuralPhaseResult:
        captured["spec"] = spec
        captured["tasks"] = tasks
        captured["config"] = config
        return StructuralPhaseResult(
            benchmark=spec.benchmark,
            repeats=config.repeats,
            campaign_id=config.campaign_id or "structural_campaign",
            sidecar_path=str(tmp_path / "structural_records.jsonl"),
            perturbation=config.perturbation,
            results=[
                StructuralRepeatResult(
                    agent="codex_cli",
                    repeat_id=0,
                    run_ids=["structural-run-1"],
                    log_paths=["logs/structural-run-1.eval"],
                    coverage_complete=True,
                    missing_sample_uuids=[],
                    duplicate_identity_keys=[],
                    identity_warning_count=0,
                )
            ],
        )

    monkeypatch.setattr(cli, "run_structural_phase", _fake_run_structural_phase)

    code = cli.main(
        [
            "structural",
            "--benchmark",
            "gaia_level1",
            "--tasks",
            "inspect_evals/gaia_level1",
            "--agent",
            "codex_cli",
            "--repeats",
            "2",
            "--campaign-id",
            "structural_campaign_a",
            "--structural-mode",
            "noop",
            "--structural-target",
            "tool_response",
            "--structural-policy-name",
            "structural_policy_test",
            "--structural-seed",
            "7",
            "--max-samples",
            "3",
        ]
    )

    assert code == 0
    assert captured["tasks"] == "inspect_evals/gaia_level1"
    assert captured["spec"].phases == ["structural"]
    assert captured["config"].campaign_id == "structural_campaign_a"
    assert captured["config"].perturbation.mode == "noop"
    assert captured["config"].perturbation.target == "tool_response"
    assert captured["config"].perturbation.policy_name == "structural_policy_test"
    assert captured["config"].perturbation.seed == 7

    out = capsys.readouterr().out
    assert "Structural reliability run complete" in out
    assert "Structural policy:" in out
