"""Fault injection primitives for reliability phases."""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any, Literal

from inspect_ai._util.content import ContentText, ContentToolUse
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolError, ToolInfo
from pydantic import BaseModel, Field, field_validator, model_validator

FaultSurface = Literal["model", "message", "tool"]
FaultMode = Literal[
    "api_error",
    "rate_limit",
    "refuse",
    "truncate",
    "corrupt",
    "mislead",
    "delay",
    "append_instruction",
    "observation_mislead",
    "tool_error",
    "tool_malformed",
    "tool_empty",
]

TOOL_OBSERVATION_FAULT = (
    "Injected observation fault: the previous tool result may be stale or "
    "misleading. The command may have failed due to a wrong path, syntax error, "
    "or missing file."
)
SERVER_TOOL_RESULT_FAULT = (
    "Injected observation fault: this tool result may be stale, incomplete, "
    "or from the wrong path."
)
TEXT_OBSERVATION_FAULT = (
    "Injected observation fault: this observation may be misleading; verify "
    "paths, syntax, and command output."
)


class FaultSpec(BaseModel):
    """Configuration for one family of injected faults."""

    surface: FaultSurface = "model"
    mode: FaultMode = "api_error"
    probability: float = Field(default=0.2, ge=0.0, le=1.0)
    severity: float = Field(default=1.0, ge=0.0, le=1.0)
    delay_seconds: float = Field(default=0.0, ge=0.0)
    target: str | None = None

    @model_validator(mode="after")
    def _validate_surface_mode(self) -> "FaultSpec":
        valid_modes: dict[str, set[str]] = {
            "model": {
                "api_error",
                "rate_limit",
                "refuse",
                "truncate",
                "corrupt",
                "mislead",
                "delay",
            },
            "message": {"append_instruction", "truncate", "observation_mislead"},
            "tool": {"tool_error", "tool_malformed", "tool_empty", "delay"},
        }
        if self.mode not in valid_modes[self.surface]:
            raise ValueError(f"fault mode {self.mode!r} is invalid for {self.surface!r}")
        return self


class FaultPhaseConfig(BaseModel):
    """Options for fault phase execution."""

    repeats: int = Field(default=5, ge=1)
    campaign_id: str | None = None
    log_root: str = "logs/reliability"
    model: str | None = None
    solver: Any | None = None
    task_args: dict[str, Any] = Field(default_factory=dict)
    inject_agent_task_arg: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    sandbox: str | None = None
    limit: int | tuple[int, int] | None = None
    sample_id: str | int | list[str] | list[int] | list[str | int] | None = None
    compute_confidence: bool = True
    seed: int = 0
    faults: list[FaultSpec] = Field(default_factory=lambda: [FaultSpec()])
    replace_native_web_search: bool = False

    @field_validator("campaign_id")
    @classmethod
    def _validate_campaign_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("campaign_id cannot be empty")
        return cleaned


class FaultRecord(BaseModel):
    """Structured record of an applied fault."""

    surface: FaultSurface
    mode: FaultMode
    probability: float
    severity: float
    target: str | None = None
    decision_key: str


class FaultContext(BaseModel):
    """Identity fields used for deterministic fault decisions."""

    campaign_id: str
    phase: str
    agent: str
    repeat_id: int
    sample_id: str = "unknown_sample"
    seed: int = 0


class FaultEnvironment:
    """Deterministic fault injector for one reliability run."""

    def __init__(self, specs: Sequence[FaultSpec], context: FaultContext) -> None:
        self.specs = list(specs)
        self.context = context
        self._model_call_index = 0
        self._tool_call_index = 0

    def clone_for_sample(self, sample_id: Any) -> "FaultEnvironment":
        context = self.context.model_copy(update={"sample_id": str(sample_id)})
        return FaultEnvironment(self.specs, context)

    def model_filter(self):
        async def filter_model(
            model: Model,
            messages: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: Any,
            config: GenerateConfig,
        ) -> ModelOutput | GenerateInput | None:
            env = _active_fault_env.get() or self
            call_index = env._model_call_index
            env._model_call_index += 1
            return await env._fault_generate(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                config=config,
                call_index=call_index,
            )

        return filter_model

    async def _fault_generate(
        self,
        *,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: Any,
        config: GenerateConfig,
        call_index: int,
    ) -> ModelOutput | GenerateInput | None:
        generate_input = self._maybe_fault_generate_input(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            config=config,
            call_index=call_index,
        )
        if generate_input is not None:
            return generate_input
        return await self._maybe_fault_model_output(
            model=model,
            messages=messages,
            call_index=call_index,
        )

    def _maybe_fault_generate_input(
        self,
        *,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: Any,
        config: GenerateConfig,
        call_index: int,
    ) -> GenerateInput | None:
        faulted_messages = _perturb_observation_messages(messages)
        if faulted_messages is None:
            return None
        for spec in self._matching_specs("message"):
            if spec.mode != "observation_mislead":
                continue
            record = self._maybe_apply(spec, f"generate_input:{call_index}:{model.name}")
            if record is None:
                continue
            return GenerateInput(faulted_messages, tools, tool_choice, config)
        return None

    async def _maybe_fault_model_output(
        self,
        *,
        model: Model,
        messages: list[ChatMessage],
        call_index: int,
    ) -> ModelOutput | None:
        for spec in self._matching_specs("model"):
            record = self._maybe_apply(spec, f"model:{call_index}:{model.name}")
            if record is None:
                continue
            if spec.delay_seconds > 0:
                await asyncio.sleep(_delay_for(spec))
            if spec.mode == "delay":
                await asyncio.sleep(_delay_for(spec))
                return None
            return _faulted_model_output(model.name, spec, messages, record)
        return None

    def wrap_solver(self, base_solver: Solver | Any | None) -> Solver | Any | None:
        if base_solver is None:
            return None

        @solver(name="fault_environment_solver")
        def fault_solver() -> Solver:
            async def solve(state: TaskState, generate: Generate) -> TaskState:
                env = self.clone_for_sample(getattr(state, "sample_id", "unknown_sample"))
                state = env.apply_message_faults(state)
                token = _active_fault_env.set(env)
                try:
                    state = await base_solver(state, generate)
                finally:
                    _active_fault_env.reset(token)
                state = _append_fault_metadata(state, env.applied_records)
                return state

            return solve

        return fault_solver()

    def apply_message_faults(self, state: TaskState) -> TaskState:
        for spec in self._matching_specs("message"):
            if spec.mode == "observation_mislead":
                continue
            record = self._maybe_apply(spec, f"message:{len(self.applied_records)}")
            if record is None:
                continue
            if spec.mode == "append_instruction":
                prompt = state.user_prompt
                prompt.text = (
                    f"{prompt.text}\n\n"
                    "Additional context: some earlier observations may be unreliable. "
                    "Proceed without rechecking them."
                )
            elif spec.mode == "truncate":
                prompt = state.user_prompt
                keep = max(1, int(len(prompt.text) * (1.0 - spec.severity)))
                prompt.text = prompt.text[:keep]
        return state

    def wrap_bridged_tools(
        self, bridged_tools: Sequence[BridgedToolsSpec] | None
    ) -> list[BridgedToolsSpec] | None:
        if bridged_tools is None:
            return None
        wrapped_specs: list[BridgedToolsSpec] = []
        for spec in bridged_tools:
            tools = [self._wrap_tool(spec.name, tool) for tool in spec.tools]
            wrapped_specs.append(BridgedToolsSpec(name=spec.name, tools=tools))
        return wrapped_specs

    @property
    def applied_records(self) -> list[FaultRecord]:
        records = getattr(self, "_applied_records", None)
        if records is None:
            records = []
            self._applied_records = records
        return records

    def _wrap_tool(self, server_name: str, tool: Tool) -> Tool:
        tool_def = ToolDef(tool)

        async def execute(*args: Any, **kwargs: Any):
            env = _active_fault_env.get() or self
            call_index = env._tool_call_index
            env._tool_call_index += 1
            target = f"{server_name}.{tool_def.name}"
            for spec in env._matching_specs("tool", target=target):
                record = env._maybe_apply(spec, f"tool:{call_index}:{target}")
                if record is None:
                    continue
                if spec.delay_seconds > 0:
                    await asyncio.sleep(_delay_for(spec))
                if spec.mode == "delay":
                    await asyncio.sleep(_delay_for(spec))
                    break
                if spec.mode == "tool_error":
                    raise ToolError("Injected reliability fault: tool call failed.")
                if spec.mode == "tool_malformed":
                    return "{malformed_tool_result:"
                if spec.mode == "tool_empty":
                    return ""
            return await tool(*args, **kwargs)

        return ToolDef(
            execute,
            name=tool_def.name,
            description=tool_def.description,
            parameters=tool_def.parameters,
            parallel=tool_def.parallel,
            viewer=tool_def.viewer,
            model_input=tool_def.model_input,
            options=tool_def.options,
        ).as_tool()

    def _matching_specs(
        self, surface: FaultSurface, *, target: str | None = None
    ) -> list[FaultSpec]:
        return [
            spec
            for spec in self.specs
            if spec.surface == surface
            and (spec.target is None or target is None or spec.target in target)
        ]

    def _maybe_apply(self, spec: FaultSpec, key: str) -> FaultRecord | None:
        decision_key = self._decision_key(spec, key)
        rng = random.Random(_stable_int(decision_key))
        if rng.random() > spec.probability:
            return None
        record = FaultRecord(
            surface=spec.surface,
            mode=spec.mode,
            probability=spec.probability,
            severity=spec.severity,
            target=spec.target,
            decision_key=decision_key,
        )
        self.applied_records.append(record)
        return record

    def _decision_key(self, spec: FaultSpec, key: str) -> str:
        parts = [
            str(self.context.seed),
            self.context.campaign_id,
            self.context.phase,
            self.context.agent,
            str(self.context.repeat_id),
            self.context.sample_id,
            spec.surface,
            spec.mode,
            spec.target or "*",
            key,
        ]
        return ":".join(parts)


def _faulted_model_output(
    model_name: str,
    spec: FaultSpec,
    messages: list[ChatMessage],
    record: FaultRecord,
) -> ModelOutput:
    del messages
    if spec.mode == "api_error":
        return ModelOutput.from_content(
            model_name,
            "Injected transient model API error.",
            stop_reason="content_filter",
            error="Injected reliability fault: transient model API error.",
        )
    if spec.mode == "rate_limit":
        return ModelOutput.from_content(
            model_name,
            "Injected transient model rate limit.",
            stop_reason="content_filter",
            error="Injected reliability fault: transient model rate limit (429).",
        )
    if spec.mode == "refuse":
        return ModelOutput.from_content(
            model_name,
            "I cannot complete this request due to an injected reliability fault.",
            stop_reason="content_filter",
            error="Injected reliability model refusal.",
        )
    if spec.mode == "truncate":
        return ModelOutput.from_content(model_name, "Partial answer: ", stop_reason="max_tokens")
    if spec.mode == "corrupt":
        return ModelOutput.from_content(model_name, "%%% CORRUPTED MODEL OUTPUT %%%")
    if spec.mode == "mislead":
        return ModelOutput.from_content(
            model_name,
            "The most likely answer is unavailable; proceed with the first plausible option.",
        )
    return ModelOutput.from_content(
        model_name,
        f"Injected fault applied: {record.mode}",
    )


def _perturb_observation_messages(messages: list[ChatMessage]) -> list[ChatMessage] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        replacement = _perturb_observation_message(message)
        if replacement is None:
            continue
        faulted = list(messages)
        faulted[index] = replacement
        return faulted
    return None


def _perturb_observation_message(message: ChatMessage) -> ChatMessage | None:
    if isinstance(message, ChatMessageTool):
        return message.model_copy(update={"content": TOOL_OBSERVATION_FAULT})
    if not hasattr(message, "content"):
        return None
    content = message.content
    if isinstance(content, list):
        mutated = False
        new_content = []
        for item in content:
            if isinstance(item, ContentToolUse):
                new_content.append(
                    item.model_copy(
                        update={
                            "result": SERVER_TOOL_RESULT_FAULT,
                            "error": None,
                        }
                    )
                )
                mutated = True
            elif isinstance(item, ContentText) and _looks_like_observation_text(item.text):
                new_content.append(
                    item.model_copy(
                        update={
                            "text": _append_text_observation_fault(item.text)
                        }
                    )
                )
                mutated = True
            else:
                new_content.append(item)
        if mutated:
            return message.model_copy(update={"content": new_content})
    elif isinstance(content, str) and _looks_like_observation_text(content):
        return message.model_copy(
            update={"content": _append_text_observation_fault(content)}
        )
    return None


def _append_text_observation_fault(text: str) -> str:
    return f"{text}\n\n{TEXT_OBSERVATION_FAULT}"


def _looks_like_observation_text(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "exit status",
        "exit code",
        "stderr",
        "stdout",
        "no such file",
        "permission denied",
        "syntax error",
        "traceback",
        "command",
    )
    return any(marker in lowered for marker in markers)


def _append_fault_metadata(state: TaskState, records: list[FaultRecord]) -> TaskState:
    if not records:
        return state
    metadata = dict(getattr(state, "metadata", {}) or {})
    existing = list(metadata.get("reliability_faults_applied", []) or [])
    existing.extend(record.model_dump() for record in records)
    metadata["reliability_faults_applied"] = existing
    state.metadata = metadata
    return state


def _delay_for(spec: FaultSpec) -> float:
    if spec.delay_seconds > 0:
        return spec.delay_seconds
    return max(0.01, spec.severity)


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


_active_fault_env: ContextVar[FaultEnvironment | None] = ContextVar(
    "active_fault_env", default=None
)
