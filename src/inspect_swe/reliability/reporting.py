"""Campaign analysis persistence and markdown reporting."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from .analysis import CampaignAnalysisResult


def write_campaign_analysis_json(
    *, result: CampaignAnalysisResult, output_path: str | Path
) -> str:
    """Write campaign analysis payload as pretty JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2)
        f.write("\n")
    return str(path)


def write_campaign_markdown_report(
    *, result: CampaignAnalysisResult, output_path: str | Path
) -> str:
    """Write a compact campaign markdown report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Inspect-SWE Reliability Campaign Report")
    lines.append("")
    lines.append(f"- Benchmark: `{result.benchmark}`")
    lines.append(f"- Campaign ID: `{result.campaign_id}`")
    if result.agent:
        lines.append(f"- Agent Filter: `{result.agent}`")
    lines.append(f"- Started At: `{_fmt_timestamp(result.resources.started_at)}`")
    lines.append(f"- Completed At: `{_fmt_timestamp(result.resources.completed_at)}`")
    lines.append(f"- Total Wall Time: {_fmt_seconds(result.resources.wall_time_sec)}")
    lines.append(f"- Total Working Time: {_fmt_seconds(result.resources.working_time_sec)}")
    lines.append(f"- Total Cost (USD): {_fmt_currency(result.resources.total_cost_usd)}")
    lines.append(f"- Input Tokens: {_fmt_int(result.resources.input_tokens)}")
    lines.append(f"- Output Tokens: {_fmt_int(result.resources.output_tokens)}")
    lines.append(f"- Cache Read Tokens: {_fmt_int(result.resources.cache_read_tokens)}")
    lines.append(f"- Cache Write Tokens: {_fmt_int(result.resources.cache_write_tokens)}")
    lines.append(f"- Reasoning Tokens: {_fmt_int(result.resources.reasoning_tokens)}")
    lines.append(f"- Total Tokens: {_fmt_int(result.resources.total_tokens)}")
    lines.append("")

    lines.append("## Phase Summary")
    lines.append("")
    lines.append(
        "| Phase | Source | Records | Samples | Repeats | Accuracy | Started | Completed | Wall Time | Cost | In | Out | Cache Read | Cache Write | Reasoning | Total |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for phase in ("baseline", "fault", "prompt", "structural"):
        summary = result.phase_summaries.get(phase)
        if summary is None:
            lines.append(
                f"| {phase} | n/a | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            "| "
            f"{phase} | {summary.source} | {summary.total_records} | {summary.sample_count} | "
            f"{summary.repeat_count} | {_fmt(summary.accuracy)} | "
            f"`{_fmt_timestamp(summary.started_at)}` | `{_fmt_timestamp(summary.completed_at)}` | "
            f"{_fmt_seconds(summary.wall_time_sec)} | {_fmt_currency(summary.total_cost_usd)} | "
            f"{_fmt_int(summary.input_tokens)} | {_fmt_int(summary.output_tokens)} | "
            f"{_fmt_int(summary.cache_read_tokens)} | {_fmt_int(summary.cache_write_tokens)} | "
            f"{_fmt_int(summary.reasoning_tokens)} | {_fmt_int(summary.total_tokens)} |"
        )
    lines.append("")

    lines.append("## Predictability")
    lines.append("")
    lines.append(
        "- Pairs: "
        f"{result.predictability.pair_count}, "
        f"Brier MSE: {_fmt(result.predictability.brier_mse)}, "
        f"Brier Predictability: {_fmt(result.predictability.brier_predictability)}, "
        f"Calibration Error: {_fmt(result.predictability.calibration_error)}, "
        f"Discrimination AUROC: {_fmt(result.predictability.discrimination_auroc)}"
    )
    lines.append("")

    lines.append("## Robustness Deltas (vs baseline)")
    lines.append("")
    lines.append(
        "- Fault delta: "
        f"{_fmt(result.robustness.fault_delta_vs_baseline)}, "
        f"Prompt delta: {_fmt(result.robustness.prompt_delta_vs_baseline)}, "
        f"Structural delta: {_fmt(result.robustness.structural_delta_vs_baseline)}"
    )
    lines.append("")

    lines.append("## Safety and Abstention")
    lines.append("")
    lines.append(
        "- Safety violation rate: "
        f"{_fmt(result.safety.violation_rate)} "
        f"({result.safety.violation_count}/{result.safety.observed_records})"
    )
    lines.append(
        "- Abstention rate: "
        f"{_fmt(result.abstention.abstention_rate)} "
        f"({result.abstention.abstention_count}/{result.abstention.observed_records}), "
        f"Selective accuracy: {_fmt(result.abstention.selective_accuracy)}"
    )
    lines.append("")

    if result.notes:
        lines.append("## Notes")
        lines.append("")
        for note in result.notes:
            lines.append(f"- {note}")
        lines.append("")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(path)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    rounded = int(round(value))
    return str(timedelta(seconds=rounded))


def _fmt_currency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.3f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def _fmt_timestamp(value: str | None) -> str:
    return value or "n/a"
