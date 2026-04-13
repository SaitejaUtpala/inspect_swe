from __future__ import annotations

import json

import pytest
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import tool
from inspect_swe.reliability.perturbations.structural import StructuralPerturbationSpec


@pytest.mark.anyio
async def test_structural_key_case_flip_transforms_dict_keys() -> None:
    @tool
    def fetch_payload():
        async def execute() -> dict[str, str]:
            return {"created_at": "2026-04-01", "user_name": "alice"}

        return execute

    spec = StructuralPerturbationSpec(
        enabled=True,
        mode="key_case_flip",
        target="tool_response",
    )
    wrapped_specs = spec.build_bridged_tools(
        bridged_tools=[BridgedToolsSpec(name="struct_tools", tools=[fetch_payload()])],
        default_seed=0,
    )
    assert wrapped_specs is not None

    wrapped_tool = wrapped_specs[0].tools[0]
    result = await wrapped_tool()
    assert isinstance(result, dict)
    assert "createdAt" in result
    assert "userName" in result


@pytest.mark.anyio
async def test_structural_envelope_wrap_transforms_string_payload() -> None:
    @tool
    def fetch_payload():
        async def execute() -> str:
            return '{"created_at":"2026-04-01","ok":true}'

        return execute

    spec = StructuralPerturbationSpec(
        enabled=True,
        mode="envelope_wrap",
        target="tool_response",
    )
    wrapped_specs = spec.build_bridged_tools(
        bridged_tools=[BridgedToolsSpec(name="struct_tools", tools=[fetch_payload()])],
        default_seed=0,
    )
    assert wrapped_specs is not None

    wrapped_tool = wrapped_specs[0].tools[0]
    result = await wrapped_tool()
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["schema_version"] == 1
    assert "payload" in parsed


def test_structural_noop_preserves_tool_reference() -> None:
    @tool
    def fetch_payload():
        async def execute() -> dict[str, str]:
            return {"created_at": "2026-04-01"}

        return execute

    original = fetch_payload()
    spec = StructuralPerturbationSpec(enabled=True, mode="noop", target="tool_response")
    wrapped_specs = spec.build_bridged_tools(
        bridged_tools=[BridgedToolsSpec(name="struct_tools", tools=[original])],
        default_seed=0,
    )
    assert wrapped_specs is not None
    assert wrapped_specs[0].tools[0] is original
