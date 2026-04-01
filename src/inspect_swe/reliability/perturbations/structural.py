"""Structural perturbation adapter scaffolding."""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from typing import Any, Literal

from pydantic import BaseModel


class StructuralPerturbationSpec(BaseModel):
    """Structural perturbation policy for reliability structural phase."""

    enabled: bool = True
    mode: Literal["noop", "key_case_flip", "envelope_wrap", "disabled"] = "noop"
    target: Literal["tool_response", "tool_schema"] = "tool_response"
    seed: int | None = None
    policy_name: str = "structural_tool_response_v1"

    def effective_enabled(self) -> bool:
        return self.enabled and self.mode != "disabled"

    def metadata_tags(self, *, default_seed: int) -> dict[str, str | int | bool]:
        return {
            "reliability_perturbation_type": "structural",
            "reliability_perturbation_policy_name": self.policy_name,
            "reliability_perturbation_structural_enabled": self.effective_enabled(),
            "reliability_perturbation_structural_mode": self.mode,
            "reliability_perturbation_structural_target": self.target,
            "reliability_perturbation_structural_seed": self.seed
            if self.seed is not None
            else default_seed,
        }

    def build_bridged_tools(
        self, *, bridged_tools: list[Any] | None, default_seed: int
    ) -> list[Any] | None:
        """Build bridged tools with structural response transforms."""
        _ = default_seed  # reserved for future deterministic mode mixing
        if bridged_tools is None:
            return None
        if not self.effective_enabled() or self.target != "tool_response":
            return list(bridged_tools)
        if self.mode == "noop":
            return list(bridged_tools)
        if self.mode not in {"key_case_flip", "envelope_wrap"}:
            return list(bridged_tools)

        wrapped_specs: list[Any] = []
        for spec in bridged_tools:
            wrapped_tools = [
                _wrap_tool_for_structural_transform(tool=tool, mode=self.mode)
                for tool in spec.tools
            ]
            wrapped_specs.append(type(spec)(name=spec.name, tools=wrapped_tools))
        return wrapped_specs


def _wrap_tool_for_structural_transform(*, tool: Any, mode: str) -> Any:
    async def wrapped_tool(*args: Any, **kwargs: Any) -> Any:
        result = await tool(*args, **kwargs)
        return _transform_tool_result(result=result, mode=mode)

    wrapped_tool.__name__ = getattr(tool, "__name__", type(tool).__name__)
    wrapped_tool.__doc__ = getattr(tool, "__doc__", None)
    with contextlib.suppress(Exception):
        wrapped_tool.__signature__ = inspect.signature(tool)
    return wrapped_tool


def _transform_tool_result(*, result: Any, mode: str) -> Any:
    if mode == "envelope_wrap":
        return _envelope_wrap_result(result)
    if mode == "key_case_flip":
        return _key_case_flip_result(result)
    return result


def _envelope_wrap_result(result: Any) -> Any:
    if isinstance(result, str):
        parsed = _try_parse_json(result)
        if parsed is not None:
            return json.dumps({"payload": parsed, "schema_version": 1})
        return json.dumps({"payload_text": result, "schema_version": 1})
    return {"payload": result, "schema_version": 1}


def _key_case_flip_result(result: Any) -> Any:
    if isinstance(result, str):
        parsed = _try_parse_json(result)
        if parsed is None:
            return result
        return json.dumps(_flip_keys_recursive(parsed))
    if isinstance(result, (dict, list)):
        return _flip_keys_recursive(result)
    return result


def _try_parse_json(value: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _flip_keys_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return {_flip_key_case(str(k)): _flip_keys_recursive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_flip_keys_recursive(v) for v in value]
    return value


def _flip_key_case(key: str) -> str:
    if "_" in key:
        return _snake_to_camel(key)
    return _camel_to_snake(key)


def _snake_to_camel(value: str) -> str:
    parts = [part for part in value.split("_") if part]
    if not parts:
        return value
    return parts[0] + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def _camel_to_snake(value: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    return snake or value
