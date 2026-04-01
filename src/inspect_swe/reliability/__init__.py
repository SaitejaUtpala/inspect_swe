"""Reliability evaluation primitives for Inspect SWE.

This package intentionally treats Inspect `.eval` logs as canonical run records.
Sidecar JSONL artifacts are derived projections used for reliability analysis.
"""

from .analysis import (
    AbstentionMetrics,
    BaselineAnalysisResult,
    CampaignAnalysisResult,
    ConsistencyMetrics,
    PhaseAnalysisSummary,
    PredictabilityMetrics,
    ResourceMetrics,
    RobustnessMetrics,
    SafetyMetrics,
    analyze_baseline_campaign,
    analyze_reliability_campaign,
)
from .artifacts import ReliabilityRecord, SidecarWriter, load_sidecar_records
from .baseline import (
    BaselineExecutionError,
    BaselinePhaseConfig,
    BaselinePhaseResult,
    BaselineRepeatResult,
    run_baseline_phase,
)
from .concurrency import (
    ConcurrencyPolicyError,
    OrchestratorConcurrency,
    validate_orchestrator_policy,
)
from .fault import (
    FaultExecutionError,
    FaultPhaseConfig,
    FaultPhaseResult,
    FaultRepeatResult,
    run_fault_phase,
)
from .hooks import (
    ReliabilityHookConfig,
    assert_reliability_hooks_active,
    configure_reliability_hooks,
    disable_reliability_hooks,
)
from .identity import ReliabilityRunIdentity
from .orchestrator import (
    CampaignPhaseEntry,
    PhaseShard,
    ReliabilityCampaignConfig,
    ReliabilityCampaignManifest,
    ReliabilityCampaignResult,
    build_phase_shards,
    preflight_reliability_spec,
    run_reliability_campaign,
)
from .perturbations import (
    FaultPerturbationSpec,
    PromptPerturbationSpec,
    StructuralPerturbationSpec,
)
from .prompt import (
    PromptExecutionError,
    PromptPhaseConfig,
    PromptPhaseResult,
    PromptRepeatResult,
    run_prompt_phase,
)
from .spec import PhaseName, ReliabilitySpec
from .structural import (
    StructuralExecutionError,
    StructuralPhaseConfig,
    StructuralPhaseResult,
    StructuralRepeatResult,
    run_structural_phase,
)
from .telemetry import TelemetryCoverageReport, assess_sidecar_coverage

__all__ = [
    "BaselineAnalysisResult",
    "CampaignAnalysisResult",
    "PredictabilityMetrics",
    "RobustnessMetrics",
    "SafetyMetrics",
    "AbstentionMetrics",
    "PhaseAnalysisSummary",
    "BaselineExecutionError",
    "BaselinePhaseConfig",
    "BaselinePhaseResult",
    "BaselineRepeatResult",
    "ConsistencyMetrics",
    "PhaseShard",
    "CampaignPhaseEntry",
    "ReliabilityCampaignConfig",
    "ReliabilityCampaignManifest",
    "ReliabilityCampaignResult",
    "ResourceMetrics",
    "FaultExecutionError",
    "FaultPerturbationSpec",
    "FaultPhaseConfig",
    "FaultPhaseResult",
    "FaultRepeatResult",
    "PromptExecutionError",
    "PromptPerturbationSpec",
    "PromptPhaseConfig",
    "PromptPhaseResult",
    "PromptRepeatResult",
    "StructuralExecutionError",
    "StructuralPerturbationSpec",
    "StructuralPhaseConfig",
    "StructuralPhaseResult",
    "StructuralRepeatResult",
    "ConcurrencyPolicyError",
    "OrchestratorConcurrency",
    "PhaseName",
    "ReliabilityHookConfig",
    "ReliabilityRecord",
    "ReliabilityRunIdentity",
    "ReliabilitySpec",
    "SidecarWriter",
    "TelemetryCoverageReport",
    "analyze_baseline_campaign",
    "analyze_reliability_campaign",
    "assess_sidecar_coverage",
    "build_phase_shards",
    "run_baseline_phase",
    "run_fault_phase",
    "run_prompt_phase",
    "run_reliability_campaign",
    "run_structural_phase",
    "assert_reliability_hooks_active",
    "configure_reliability_hooks",
    "disable_reliability_hooks",
    "load_sidecar_records",
    "preflight_reliability_spec",
    "validate_orchestrator_policy",
]
