"""Reliability evaluation primitives for Inspect SWE.

This package treats Inspect `.eval` logs as canonical run records.
"""

from .baseline import (
    BaselineExecutionError,
    BaselinePhaseConfig,
    BaselinePhaseResult,
    BaselineRepeatResult,
    run_baseline_phase,
)
from .analyze import PhaseAnalyzeResult, analyze_phase
from .concurrency import (
    ConcurrencyPolicyError,
    OrchestratorConcurrency,
    validate_orchestrator_policy,
)
from .eval_view import BaselineSampleView, extract_baseline_sample_views
from .spec import PhaseName, ReliabilitySpec

__all__ = [
    "BaselineExecutionError",
    "PhaseAnalyzeResult",
    "BaselinePhaseConfig",
    "BaselinePhaseResult",
    "BaselineRepeatResult",
    "ConcurrencyPolicyError",
    "OrchestratorConcurrency",
    "PhaseName",
    "BaselineSampleView",
    "ReliabilitySpec",
    "analyze_phase",
    "extract_baseline_sample_views",
    "run_baseline_phase",
    "validate_orchestrator_policy",
]
