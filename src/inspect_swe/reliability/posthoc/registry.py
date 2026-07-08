"""Dataset-keyed registry of post-hoc repairs.

Adding a new dataset repair is: implement it in its own module (like
``gaia.py``), then register it here under the dataset key returned by
``detect_dataset``. Per-agent scoping lives on the ``PostHocRepair`` itself.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.solver import Solver

from .base import PostHocRepair, wrap_solver_with_posthoc
from .gaia import GAIA_ANSWER_REPAIR

# Dataset key -> repair. One repair per dataset; the repair declares which
# agents it applies to.
POSTHOC_REPAIRS: dict[str, PostHocRepair] = {
    "gaia": GAIA_ANSWER_REPAIR,
}


def detect_dataset(benchmark: str | None) -> str | None:
    """Map a benchmark ref (registry name or file task ref) to a dataset key."""
    if not benchmark:
        return None
    ref = str(benchmark).lower()
    if "gaia" in ref:
        return "gaia"
    if "tau2" in ref or "taubench" in ref:
        return "taubench"
    return None


def resolve_posthoc_repair(agent: str, benchmark: str | None) -> PostHocRepair | None:
    """Return the repair configured for this dataset+agent, if any."""
    dataset = detect_dataset(benchmark)
    if dataset is None:
        return None
    repair = POSTHOC_REPAIRS.get(dataset)
    if repair is None or not repair.applies_to_agent(agent):
        return None
    return repair


def apply_posthoc_repair(
    base_solver: Solver | Any | None,
    *,
    agent: str,
    benchmark: str | None,
    enabled: bool = True,
) -> Solver | Any | None:
    """Wrap ``base_solver`` with the applicable post-hoc repair, or no-op.

    Returns the base solver unchanged when disabled, when there is no repair
    registered for the dataset, or when the repair does not apply to the agent.
    """
    if base_solver is None or not enabled:
        return base_solver
    repair = resolve_posthoc_repair(agent, benchmark)
    if repair is None:
        return base_solver
    return wrap_solver_with_posthoc(base_solver, repair)
