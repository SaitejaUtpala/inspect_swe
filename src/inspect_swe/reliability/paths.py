"""Canonical reliability path builders."""

from __future__ import annotations

from pathlib import Path


def repeat_log_dir(
    *,
    log_root: str,
    benchmark: str,
    agent: str,
    phase: str,
    repeat_id: int,
    campaign_id: str,
) -> Path:
    """Return repeat log directory using the simplified canonical layout.

    Layout:
    <log_root>/<benchmark>/<agent>/<benchmark>_<agent>_<phase>_rep_<id>_<campaign>
    """
    run_dir_name = (
        f"{benchmark}_{agent}_{phase}_rep_{repeat_id:03d}_{campaign_id}"
    )
    return Path(log_root) / benchmark / agent / run_dir_name


def campaign_sidecar_path(
    *,
    log_root: str,
    benchmark: str,
    phase: str,
    campaign_id: str,
    sidecar_filename: str,
) -> Path:
    """Return campaign sidecar path in a compact campaign-scoped layout."""
    return Path(log_root) / benchmark / f"{benchmark}_{phase}_{campaign_id}_{sidecar_filename}"


def campaign_manifest_path(*, log_root: str, benchmark: str, campaign_id: str) -> Path:
    """Return campaign manifest path for resumable orchestration."""
    return Path(log_root) / benchmark / f"{benchmark}_campaign_{campaign_id}_manifest.json"


def campaign_analysis_path(*, log_root: str, benchmark: str, campaign_id: str) -> Path:
    """Return campaign analysis JSON path."""
    return Path(log_root) / benchmark / f"{benchmark}_campaign_{campaign_id}_analysis.json"


def campaign_report_path(*, log_root: str, benchmark: str, campaign_id: str) -> Path:
    """Return campaign markdown report path."""
    return Path(log_root) / benchmark / f"{benchmark}_campaign_{campaign_id}_report.md"

