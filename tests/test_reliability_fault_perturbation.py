from __future__ import annotations

from types import SimpleNamespace

import pytest
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import ToolError, tool
from inspect_swe.reliability.perturbations.fault import FaultPerturbationSpec


@pytest.mark.anyio
async def test_model_error_filter_injects_and_respects_max_faults() -> None:
    spec = FaultPerturbationSpec(
        enabled=True,
        mode="model_error",
        target="model",
        fault_rate=1.0,
        max_faults_per_sample=1,
        seed=7,
    )
    generate_filter = spec.build_generate_filter(default_seed=0)
    assert generate_filter is not None

    model = SimpleNamespace(name="test-model")
    messages = [SimpleNamespace(role="user", content="sample prompt")]

    first = await generate_filter(model, messages, [], None, None)
    second = await generate_filter(model, messages, [], None, None)

    assert first is not None
    assert getattr(first, "error", None) == "reliability_fault_injected_model_error"
    assert second is None


@pytest.mark.anyio
async def test_model_error_filter_skips_when_rate_zero() -> None:
    spec = FaultPerturbationSpec(
        enabled=True,
        mode="model_error",
        target="model",
        fault_rate=0.0,
        max_faults_per_sample=1,
    )
    generate_filter = spec.build_generate_filter(default_seed=11)
    assert generate_filter is not None

    model = SimpleNamespace(name="test-model")
    messages = [SimpleNamespace(role="user", content="sample prompt")]
    output = await generate_filter(model, messages, [], None, None)

    assert output is None


@pytest.mark.anyio
async def test_tool_error_wrappers_inject_and_respect_max_faults() -> None:
    @tool
    def add_one():
        async def execute(x: int) -> int:
            return x + 1

        return execute

    spec = FaultPerturbationSpec(
        enabled=True,
        mode="tool_error",
        target="tool",
        fault_rate=1.0,
        max_faults_per_sample=1,
        seed=5,
    )
    wrapped_specs = spec.build_bridged_tools(
        bridged_tools=[BridgedToolsSpec(name="fault_tools", tools=[add_one()])],
        default_seed=0,
    )
    assert wrapped_specs is not None
    wrapped_tool = wrapped_specs[0].tools[0]

    with pytest.raises(ToolError):
        await wrapped_tool(x=1)
    assert await wrapped_tool(x=1) == 2


def test_tool_error_wrapper_noop_mode_preserves_tool_reference() -> None:
    @tool
    def add_one():
        async def execute(x: int) -> int:
            return x + 1

        return execute

    original = add_one()
    spec = FaultPerturbationSpec(
        enabled=True,
        mode="noop",
        target="tool",
        fault_rate=1.0,
    )
    wrapped_specs = spec.build_bridged_tools(
        bridged_tools=[BridgedToolsSpec(name="fault_tools", tools=[original])],
        default_seed=0,
    )
    assert wrapped_specs is not None
    assert wrapped_specs[0].tools[0] is original
