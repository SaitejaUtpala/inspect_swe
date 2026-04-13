from inspect_swe.reliability.paths import (
    campaign_analysis_path,
    campaign_manifest_path,
    campaign_report_path,
    campaign_sidecar_path,
    repeat_log_dir,
)


def test_repeat_log_dir_uses_campaign_first_layout() -> None:
    path = repeat_log_dir(
        log_root="logs/reliability",
        benchmark="gaia_level1",
        agent="codex_cli",
        phase="prompt",
        repeat_id=2,
        campaign_id="campaign_x",
    )

    assert (
        str(path)
        == "logs/reliability/campaign_x/gaia_level1/gaia_level1_codex_cli_prompt_rep_002_campaign_x"
    )


def test_campaign_artifacts_live_under_campaign_root() -> None:
    sidecar = campaign_sidecar_path(
        log_root="logs/reliability",
        benchmark="gaia_level1",
        phase="baseline",
        campaign_id="campaign_x",
        sidecar_filename="records.jsonl",
    )
    manifest = campaign_manifest_path(
        log_root="logs/reliability",
        benchmark="gaia_level1",
        campaign_id="campaign_x",
    )
    analysis = campaign_analysis_path(
        log_root="logs/reliability",
        benchmark="gaia_level1",
        campaign_id="campaign_x",
    )
    report = campaign_report_path(
        log_root="logs/reliability",
        benchmark="gaia_level1",
        campaign_id="campaign_x",
    )

    assert (
        str(sidecar)
        == "logs/reliability/campaign_x/gaia_level1/gaia_level1_baseline_campaign_x_records.jsonl"
    )
    assert (
        str(manifest)
        == "logs/reliability/campaign_x/gaia_level1/gaia_level1_campaign_campaign_x_manifest.json"
    )
    assert (
        str(analysis)
        == "logs/reliability/campaign_x/gaia_level1/gaia_level1_campaign_campaign_x_analysis.json"
    )
    assert (
        str(report)
        == "logs/reliability/campaign_x/gaia_level1/gaia_level1_campaign_campaign_x_report.md"
    )
