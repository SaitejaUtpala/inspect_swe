"""Phase analysis orchestration for reliability runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from pydantic import BaseModel, Field

from .eval_view import BaselineSampleView, extract_baseline_sample_views
from .metrics import compute_baseline_metrics, compute_fault_metrics
from .spec import PhaseName


class PhaseAnalyzeResult(BaseModel):
    """Computed metrics and output log location for one phase."""

    benchmark: str
    campaign_id: str
    phase: PhaseName
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    output_log_path: str


def analyze_phase(
    *,
    benchmark: str,
    campaign_id: str,
    phase: PhaseName,
    log_root: str = "logs/reliability",
    agent: str | None = None,
) -> PhaseAnalyzeResult:
    """Analyze one reliability phase and write a `.log` artifact."""
    output_log_path = str(_analysis_log_path(log_root, benchmark, campaign_id, phase))
    if phase not in {"baseline", "fault"}:
        result = PhaseAnalyzeResult(
            benchmark=benchmark,
            campaign_id=campaign_id,
            phase=phase,
            status="not_implemented",
            metrics={},
            output_log_path=output_log_path,
        )
        _write_analysis_log(result)
        return result

    if phase == "fault":
        baseline_paths, baseline_views = _load_phase_views(
            benchmark=benchmark,
            campaign_id=campaign_id,
            phase="baseline",
            log_root=log_root,
            agent=agent,
        )
        fault_paths, fault_views = _load_phase_views(
            benchmark=benchmark,
            campaign_id=campaign_id,
            phase="fault",
            log_root=log_root,
            agent=agent,
        )
        metrics = compute_fault_metrics(
            baseline_views=baseline_views,
            baseline_eval_paths=baseline_paths,
            fault_views=fault_views,
            fault_eval_paths=fault_paths,
        )
        result = PhaseAnalyzeResult(
            benchmark=benchmark,
            campaign_id=campaign_id,
            phase=phase,
            status="ok",
            metrics=metrics,
            output_log_path=output_log_path,
        )
        _write_analysis_log(result)
        return result

    eval_paths, views = _load_phase_views(
        benchmark=benchmark,
        campaign_id=campaign_id,
        phase="baseline",
        log_root=log_root,
        agent=agent,
    )
    metrics = compute_baseline_metrics(views, eval_paths)
    result = PhaseAnalyzeResult(
        benchmark=benchmark,
        campaign_id=campaign_id,
        phase=phase,
        status="ok",
        metrics=metrics,
        output_log_path=output_log_path,
    )
    _write_analysis_log(result)
    return result


def _load_phase_views(
    *,
    benchmark: str,
    campaign_id: str,
    phase: str,
    log_root: str,
    agent: str | None,
) -> tuple[list[Path], list[BaselineSampleView]]:
    phase_dir = Path(log_root) / benchmark / phase / campaign_id
    eval_paths = sorted(phase_dir.glob("**/*.eval"))
    views: list[BaselineSampleView] = []
    for eval_path in eval_paths:
        eval_log = read_eval_log(str(eval_path), header_only=False)
        run_metadata = dict(getattr(eval_log.eval, "metadata", {}) or {})
        run_repeat_id = _to_int(run_metadata.get("reliability_repeat_id"), 0)
        run_agent = agent or _to_text(run_metadata.get("reliability_agent")) or "unknown_agent"
        run_views, _ = extract_baseline_sample_views(
            eval_log,
            expected_phase=phase,
            expected_agent=run_agent,
            expected_repeat_id=run_repeat_id,
            strict_identity_tags=True,
        )
        views.extend(run_views)
    return eval_paths, views


def _analysis_log_path(log_root: str, benchmark: str, campaign_id: str, phase: str) -> Path:
    return Path(log_root) / benchmark / "analysis" / f"{campaign_id}_{phase}_analysis.log"


def _write_analysis_log(result: PhaseAnalyzeResult) -> None:
    path = Path(result.output_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": result.benchmark,
        "campaign_id": result.campaign_id,
        "phase": result.phase,
        "status": result.status,
        "metrics": result.metrics,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _to_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _to_text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None
