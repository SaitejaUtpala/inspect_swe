"""Fault perturbation adapter scaffolding.

This module defines the initial adapter contract for fault perturbations.
This implementation supports deterministic synthetic failures for both model
generation and bridged host-side tool execution.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Sequence
from functools import wraps
from hashlib import sha256
from typing import Any, Literal

from inspect_ai.log._samples import sample_active
from inspect_ai.model import GenerateFilter, ModelOutput
from inspect_ai.tool import ToolError
from pydantic import BaseModel, Field


class FaultPerturbationSpec(BaseModel):
    """Fault perturbation policy for reliability fault phase."""

    enabled: bool = True
    mode: Literal["noop", "model_error", "tool_error", "disabled"] = "noop"
    target: Literal["model", "tool"] = "model"
    fault_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_faults_per_sample: int = Field(default=1, ge=0)
    seed: int | None = None
    policy_name: str = "fault_scaffold_v1"

    def effective_enabled(self) -> bool:
        """Return whether perturbation should be wired for execution."""
        return self.enabled and self.mode != "disabled"

    def metadata_tags(self, *, default_seed: int) -> dict[str, str | int | float | bool]:
        """Emit standardized fault metadata tags for sidecar attribution."""
        return {
            "reliability_perturbation_type": "fault",
            "reliability_perturbation_policy_name": self.policy_name,
            "reliability_perturbation_fault_enabled": self.effective_enabled(),
            "reliability_perturbation_fault_mode": self.mode,
            "reliability_perturbation_fault_target": self.target,
            "reliability_perturbation_fault_rate": self.fault_rate,
            "reliability_perturbation_fault_max_per_sample": self.max_faults_per_sample,
            "reliability_perturbation_fault_seed": self.seed
            if self.seed is not None
            else default_seed,
        }

    def build_generate_filter(self, *, default_seed: int) -> GenerateFilter | None:
        """Build model generate filter for fault injection.

        Current behavior:
        - `noop`: pass-through behavior.
        - `model_error`: deterministic synthetic model failures based on
          `fault_rate` and `max_faults_per_sample`.
        - `tool_error`: no model filter (tool wrappers handle injection).
        - `disabled`: no filter.
        - returns None otherwise.
        """
        if not self.effective_enabled() or self.target != "model":
            return None
        if self.mode == "noop":
            async def _noop_fault_filter(model, input, tools, tool_choice, config):
                return None
            return _noop_fault_filter

        if self.mode != "model_error":
            return None

        seed = self.seed if self.seed is not None else default_seed
        faults_by_sample: dict[str, int] = {}

        async def _model_error_fault_filter(model, input, tools, tool_choice, config):
            sample_key = _sample_key(input)
            used_faults = faults_by_sample.get(sample_key, 0)
            if used_faults >= self.max_faults_per_sample:
                return None

            if not _should_inject_fault(
                seed=seed,
                sample_key=sample_key,
                attempt_index=used_faults,
                fault_rate=self.fault_rate,
            ):
                return None

            faults_by_sample[sample_key] = used_faults + 1
            model_name = getattr(model, "name", str(model))
            return ModelOutput.from_content(
                model=model_name,
                content=(
                    "FAULT_INJECTION: Synthetic model failure injected by "
                    "Inspect-SWE reliability fault adapter."
                ),
                error="reliability_fault_injected_model_error",
            )

        return _model_error_fault_filter

    def build_bridged_tools(
        self,
        *,
        bridged_tools: Sequence[Any] | None,
        default_seed: int,
    ) -> list[Any] | None:
        """Build bridged tool wrappers for tool-side fault injection."""
        if bridged_tools is None:
            return None
        if not self.effective_enabled() or self.target != "tool":
            return list(bridged_tools)
        if self.mode == "noop":
            return list(bridged_tools)
        if self.mode != "tool_error":
            return list(bridged_tools)

        seed = self.seed if self.seed is not None else default_seed
        call_count_by_sample: dict[str, int] = {}
        fault_count_by_sample: dict[str, int] = {}

        wrapped_specs: list[Any] = []
        for spec in bridged_tools:
            wrapped_tools = [
                _wrap_tool_for_fault(
                    tool=tool,
                    server_name=spec.name,
                    seed=seed,
                    fault_rate=self.fault_rate,
                    max_faults_per_sample=self.max_faults_per_sample,
                    call_count_by_sample=call_count_by_sample,
                    fault_count_by_sample=fault_count_by_sample,
                )
                for tool in spec.tools
            ]
            wrapped_specs.append(type(spec)(name=spec.name, tools=wrapped_tools))
        return wrapped_specs


def _sample_key(messages: list[object]) -> str:
    chunks: list[str] = []
    for message in messages:
        role = getattr(message, "role", None) or type(message).__name__
        content = getattr(message, "content", "")
        chunks.append(f"{role}:{content!r}")
    joined = "|".join(chunks)
    digest = sha256(joined.encode("utf-8")).hexdigest()
    return digest


def _should_inject_fault(
    *, seed: int, sample_key: str, attempt_index: int, fault_rate: float
) -> bool:
    if fault_rate <= 0:
        return False
    if fault_rate >= 1:
        return True

    payload = f"{seed}:{sample_key}:{attempt_index}".encode("utf-8")
    digest = sha256(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64
    return value < fault_rate


def _wrap_tool_for_fault(
    *,
    tool: Any,
    server_name: str,
    seed: int,
    fault_rate: float,
    max_faults_per_sample: int,
    call_count_by_sample: dict[str, int],
    fault_count_by_sample: dict[str, int],
) -> Any:
    tool_name = getattr(tool, "__name__", type(tool).__name__)

    @wraps(tool)
    async def wrapped_tool(*args: Any, **kwargs: Any) -> Any:
        sample_key = _active_sample_key()
        sample_call_count = call_count_by_sample.get(sample_key, 0)
        call_count_by_sample[sample_key] = sample_call_count + 1

        sample_fault_count = fault_count_by_sample.get(sample_key, 0)
        if sample_fault_count < max_faults_per_sample:
            call_key = f"{sample_key}:{server_name}:{tool_name}"
            if _should_inject_fault(
                seed=seed,
                sample_key=call_key,
                attempt_index=sample_call_count,
                fault_rate=fault_rate,
            ):
                fault_count_by_sample[sample_key] = sample_fault_count + 1
                raise ToolError(
                    "FAULT_INJECTION: Synthetic tool failure injected by "
                    "Inspect-SWE reliability fault adapter."
                )

        return await tool(*args, **kwargs)

    with contextlib.suppress(Exception):
        wrapped_tool.__signature__ = inspect.signature(tool)
    return wrapped_tool


def _active_sample_key() -> str:
    active = sample_active()
    if active is None:
        return "no_active_sample"
    run_id = active.run_id or "no_run_id"
    sample_id = str(active.sample.id)
    return f"{run_id}:{sample_id}"
