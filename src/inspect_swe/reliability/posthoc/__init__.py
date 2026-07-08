"""Post-hoc repair framework for reliability runs.

Each dataset can register its own post-hoc repair (applied after the agent
finishes, before scoring), scoped to specific agents. Phase runners call
``apply_posthoc_repair`` and remain dataset-agnostic.
"""

from .base import PostHocRepair, PostHocRepairFn, wrap_solver_with_posthoc
from .gaia import (
    GAIA_ANSWER_FORMAT_INSTRUCTIONS,
    GAIA_ANSWER_REPAIR,
    repair_gaia_answer,
)
from .registry import (
    POSTHOC_REPAIRS,
    apply_posthoc_repair,
    detect_dataset,
    resolve_posthoc_repair,
)

__all__ = [
    "PostHocRepair",
    "PostHocRepairFn",
    "wrap_solver_with_posthoc",
    "GAIA_ANSWER_FORMAT_INSTRUCTIONS",
    "GAIA_ANSWER_REPAIR",
    "repair_gaia_answer",
    "POSTHOC_REPAIRS",
    "apply_posthoc_repair",
    "detect_dataset",
    "resolve_posthoc_repair",
]
