from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from inspect_swe.reliability import (
    ReliabilityCampaignConfig,
    ReliabilityHookConfig,
    ReliabilitySpec,
    build_phase_shards,
    configure_reliability_hooks,
    disable_reliability_hooks,
    preflight_reliability_spec,
    run_reliability_campaign,
)


def test_build_phase_shards_cartesian_product(tmp_path: Path) -> None:
    spec = ReliabilitySpec(
        benchmark="swe_bench_verified",
        agents=["codex_cli", "claude_code"],
        phases=["baseline", "fault"],
    )
    shards = build_phase_shards(spec, tmp_path)
    assert len(shards) == 4
    assert shards[0].shard_index == 0
    assert shards[-1].shard_index == 3
    assert shards[0].phase == "baseline"
    assert shards[0].agent == "codex_cli"
    assert shards[-1].phase == "fault"
    assert shards[-1].agent == "claude_code"


def test_preflight_requires_enabled_hooks_when_configured(tmp_path: Path) -> None:
    spec = ReliabilitySpec(
        benchmark="swe_bench_verified",
        agents=["codex_cli"],
        phases=["baseline"],
        fail_on_missing_hooks=True,
    )

    disable_reliability_hooks()
    with pytest.raises(RuntimeError):
        preflight_reliability_spec(spec)

    configure_reliability_hooks(
        ReliabilityHookConfig(
            enabled=True,
            sidecar_path=str(tmp_path / "records.jsonl"),
        )
    )
    preflight_reliability_spec(spec)
    disable_reliability_hooks()


def test_run_reliability_campaign_creates_manifest_and_phase_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class _FakeResult:
        def __init__(self, phase: str) -> None:
            self.phase = phase

        def model_dump(self, mode: str = "python") -> dict[str, Any]:
            return {
                "benchmark": "gaia_level1",
                "campaign_id": "campaign_x",
                "sidecar_path": str(tmp_path / f"{self.phase}.jsonl"),
                "campaign_json_path": str(tmp_path / f"{self.phase}.json"),
            }

    def _fake_baseline(**kwargs: Any) -> Any:
        calls.append("baseline")
        return _FakeResult("baseline")

    def _fake_prompt(**kwargs: Any) -> Any:
        calls.append("prompt")
        return _FakeResult("prompt")

    monkeypatch.setattr("inspect_swe.reliability.baseline.run_baseline_phase", _fake_baseline)
    monkeypatch.setattr("inspect_swe.reliability.prompt.run_prompt_phase", _fake_prompt)

    spec = ReliabilitySpec(
        benchmark="gaia_level1",
        agents=["codex_cli"],
        phases=["baseline", "prompt", "safety"],
        fail_on_missing_hooks=False,
    )
    result = run_reliability_campaign(
        spec=spec,
        tasks="inspect_evals/gaia_level1",
        config=ReliabilityCampaignConfig(
            campaign_id="campaign_x",
            log_root=str(tmp_path),
            run_analysis=False,
            write_report=False,
            configure_hooks=False,
        ),
    )

    assert calls == ["baseline", "prompt"]
    assert Path(result.manifest_path).exists()
    assert result.phase_entries["baseline"].status == "completed"
    assert result.phase_entries["prompt"].status == "completed"
    assert result.phase_entries["safety"].status == "skipped"


def test_run_reliability_campaign_resume_skips_completed_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_count = 0

    class _FakeResult:
        def model_dump(self, mode: str = "python") -> dict[str, Any]:
            return {
                "benchmark": "gaia_level1",
                "campaign_id": "campaign_resume",
                "sidecar_path": str(tmp_path / "baseline.jsonl"),
                "campaign_json_path": str(tmp_path / "baseline.json"),
            }

    def _fake_baseline(**kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return _FakeResult()

    monkeypatch.setattr("inspect_swe.reliability.baseline.run_baseline_phase", _fake_baseline)

    spec = ReliabilitySpec(
        benchmark="gaia_level1",
        agents=["codex_cli"],
        phases=["baseline"],
        fail_on_missing_hooks=False,
    )
    config = ReliabilityCampaignConfig(
        campaign_id="campaign_resume",
        log_root=str(tmp_path),
        run_analysis=False,
        write_report=False,
        configure_hooks=False,
        resume=True,
    )
    run_reliability_campaign(spec=spec, tasks="inspect_evals/gaia_level1", config=config)
    run_reliability_campaign(spec=spec, tasks="inspect_evals/gaia_level1", config=config)

    assert call_count == 1
