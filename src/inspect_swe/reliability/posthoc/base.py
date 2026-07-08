"""Generic post-hoc repair framework for reliability runs.

A "post-hoc repair" is a dataset-specific pass that runs after the agent has
finished, once per sample, and standardizes the final ``state.output`` before
scoring. Repairs are opt-in per dataset and per agent (see ``registry.py``),
so different benchmarks can plug in their own post-processing without touching
the phase runners.

Repairs must be auditable: they append their work to the message history and
record what changed on ``state.metadata`` rather than silently rewriting
output.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from inspect_ai.agent import as_solver, is_agent
from inspect_ai.solver import Generate, Solver, TaskState, solver

PostHocRepairFn = Callable[[TaskState], Awaitable[TaskState]]


@dataclass(frozen=True)
class PostHocRepair:
    """A dataset-specific post-hoc repair applied after an agent finishes.

    Attributes:
        name: Stable identifier used for the wrapping solver name.
        repair: Async callable that receives the finished ``TaskState`` and
            returns a (possibly) repaired ``TaskState``.
        agents: Agents this repair applies to. Empty means all agents.
    """

    name: str
    repair: PostHocRepairFn
    agents: frozenset[str] = field(default_factory=frozenset)

    def applies_to_agent(self, agent: str) -> bool:
        return not self.agents or agent in self.agents


def wrap_solver_with_posthoc(
    base_solver: Solver | Any | None,
    repair: PostHocRepair,
) -> Solver | Any | None:
    """Wrap ``base_solver`` so ``repair`` runs after it completes."""
    if base_solver is None:
        return None

    wrapped = as_solver(base_solver) if is_agent(base_solver) else base_solver

    @solver(name=f"{repair.name}_solver")
    def posthoc_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state = await wrapped(state, generate)
            return await repair.repair(state)

        return solve

    return posthoc_solver()
