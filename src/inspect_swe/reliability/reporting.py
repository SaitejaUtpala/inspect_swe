"""Campaign analysis persistence and markdown reporting."""

from __future__ import annotations

import json
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
    lines.append("")

    lines.append("## Phase Summary")
    lines.append("")
    lines.append("| Phase | Records | Samples | Repeats | Accuracy |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for phase in ("baseline", "fault", "prompt", "structural"):
        summary = result.phase_summaries.get(phase)
        if summary is None:
            lines.append(f"| {phase} | 0 | 0 | 0 | n/a |")
            continue
        lines.append(
            "| "
            f"{phase} | {summary.total_records} | {summary.sample_count} | "
            f"{summary.repeat_count} | {_fmt(summary.accuracy)} |"
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
