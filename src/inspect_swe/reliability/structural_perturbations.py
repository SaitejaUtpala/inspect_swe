"""Structural perturbations for reliability evals."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any, Literal

from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ContentText,
    GenerateConfig,
    GenerateInput,
    Model,
    ModelOutput,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolInfo, ToolParams
from pydantic import BaseModel, Field

StructuralStrength = Literal["mild", "medium", "severe"]
StructuralKind = Literal["gaia", "taubench"]

_EXECUTION_ONLY_FIELDS = {"cmd", "command", "query", "summary"}
_PARAM_ABBREVIATIONS = {
    "flight_number": "fltNo",
    "flight_type": "fltType",
    "reservation_id": "resId",
    "passenger_id": "paxId",
    "user_id": "uid",
    "payment_id": "payId",
    "baggage_allowance": "bags",
    "total_baggages": "bagsTotal",
    "nonfree_baggages": "bagsPaid",
    "origin": "orig",
    "destination": "dest",
    "date": "dt",
    "cabin": "cls",
}
_KEY_ABBREVIATIONS = {
    "reservation_id": "res_id",
    "flight_number": "flt_no",
    "flight_type": "flt_type",
    "first_name": "fname",
    "last_name": "lname",
    "passengers": "pax",
    "payment_methods": "pay_methods",
    "payment_history": "pay_hist",
    "total_baggages": "bags",
    "nonfree_baggages": "paid_bags",
    "created_at": "created",
    "available_seats": "avail_seats",
    "date": "dt",
}
_VALUE_ABBREVIATIONS = {
    "confirmed": "CNF",
    "cancelled": "CXL",
    "pending": "PND",
    "completed": "CMP",
    "basic_economy": "Y",
    "economy": "M",
    "business": "J",
    "first": "F",
    "yes": "Y",
    "no": "N",
}
_SMALL_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


class StructuralRecord(BaseModel):
    """One applied structural perturbation."""

    kind: StructuralKind
    strength: StructuralStrength
    target: str
    summary: str


class StructuralContext(BaseModel):
    """Identity fields for deterministic structural perturbations."""

    campaign_id: str
    phase: str = "structural"
    agent: str
    repeat_id: int
    sample_id: str = "unknown_sample"
    seed: int = 0


class StructuralEnvironment:
    """Apply semantics-preserving structural perturbations."""

    def __init__(
        self,
        *,
        kind: StructuralKind,
        strength: StructuralStrength,
        context: StructuralContext,
    ) -> None:
        self.kind = kind
        self.strength = strength
        self.context = context
        self.param_mapping: dict[str, dict[str, str]] = {}
        self._seen_observation_ids: set[str] = set()
        self._records: list[StructuralRecord] = []

    @property
    def records(self) -> list[StructuralRecord]:
        return self._records

    def clone_for_sample(self, sample_id: Any) -> "StructuralEnvironment":
        env = StructuralEnvironment(
            kind=self.kind,
            strength=self.strength,
            context=self.context.model_copy(update={"sample_id": str(sample_id)}),
        )
        env.param_mapping = {
            tool_name: dict(mapping) for tool_name, mapping in self.param_mapping.items()
        }
        env._records = list(self._records)
        return env

    def wrap_solver(self, base_solver: Solver | Any | None) -> Solver | Any | None:
        """Perturb GAIA-style prompts before the agent starts."""
        if base_solver is None:
            return None

        @solver(name="structural_environment_solver")
        def structural_solver() -> Solver:
            async def solve(state: TaskState, generate: Generate) -> TaskState:
                env = self.clone_for_sample(getattr(state, "sample_id", "unknown_sample"))
                if env.kind == "gaia":
                    env.perturb_state_prompt(state)
                token = _active_structural_env.set(env)
                try:
                    state = await base_solver(state, generate)
                finally:
                    _active_structural_env.reset(token)
                return _append_structural_metadata(state, env)

            return solve

        return structural_solver()

    def model_filter(self):
        """Return a bridge filter for model-facing observation text."""

        async def filter_model(
            model: Model,
            messages: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: Any,
            config: GenerateConfig,
        ) -> ModelOutput | GenerateInput | None:
            del model
            env = _active_structural_env.get() or self
            if env.kind != "gaia":
                return None
            faulted_messages = env.perturb_observation_messages(messages)
            if faulted_messages is None:
                return None
            return GenerateInput(faulted_messages, tools, tool_choice, config)

        return filter_model

    def perturb_state_prompt(self, state: TaskState) -> TaskState:
        prompt = state.user_prompt
        perturbed, records = perturb_gaia_prompt(
            prompt.text,
            strength=self.strength,
            seed_key=self._decision_key("gaia_prompt"),
        )
        if perturbed != prompt.text:
            prompt.text = perturbed
            self._records.extend(records)
        return state

    def perturb_observation_messages(
        self, messages: list[ChatMessage]
    ) -> list[ChatMessage] | None:
        if self.strength != "severe":
            return None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            identity = _message_identity(message)
            if identity in self._seen_observation_ids:
                continue
            replacement, records = perturb_gaia_observation_message(
                message,
                strength=self.strength,
            )
            self._seen_observation_ids.add(identity)
            if replacement is None:
                continue
            faulted = list(messages)
            faulted[index] = replacement
            self._records.extend(records)
            return faulted
        return None

    def wrap_bridged_tools(
        self, bridged_tools: Sequence[BridgedToolsSpec] | None
    ) -> list[BridgedToolsSpec] | None:
        """Perturb TauBench-style tool schemas and responses."""
        if bridged_tools is None:
            return None
        if self.kind != "taubench":
            return list(bridged_tools)
        return [
            BridgedToolsSpec(
                name=spec.name,
                tools=[self._wrap_tool(spec.name, tool) for tool in spec.tools],
            )
            for spec in bridged_tools
        ]

    def _wrap_tool(self, server_name: str, tool: Tool) -> Tool:
        tool_def = ToolDef(tool)
        perturbed_params, mapping, records = perturb_taubench_tool_params(
            tool_def.parameters,
            strength=self.strength,
        )
        if mapping:
            self.param_mapping[tool_def.name] = mapping
        self._records.extend(
            record.model_copy(update={"target": f"{server_name}.{record.target}"})
            for record in records
        )

        async def execute(*args: Any, **kwargs: Any) -> Any:
            env = _active_structural_env.get() or self
            original_kwargs = reverse_tool_args(
                tool_def.name,
                kwargs,
                env.param_mapping,
            )
            result = await tool(*args, **original_kwargs)
            perturbed, records = perturb_taubench_tool_result(
                result,
                strength=env.strength,
                target=tool_def.name,
            )
            env._records.extend(records)
            return perturbed

        return ToolDef(
            execute,
            name=tool_def.name,
            description=tool_def.description,
            parameters=perturbed_params,
            parallel=tool_def.parallel,
            viewer=tool_def.viewer,
            model_input=tool_def.model_input,
            options=tool_def.options,
        ).as_tool()

    def _decision_key(self, target: str) -> str:
        return ":".join(
            [
                str(self.context.seed),
                self.context.campaign_id,
                self.context.phase,
                self.context.agent,
                str(self.context.repeat_id),
                self.context.sample_id,
                self.kind,
                self.strength,
                target,
            ]
        )


class StructuralPhaseConfig(BaseModel):
    """Options shared by structural phase runners."""

    strength: StructuralStrength = "medium"
    kind: StructuralKind = "gaia"
    seed: int = 0
    include_baseline: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


def perturb_gaia_prompt(
    text: str,
    *,
    strength: StructuralStrength,
    seed_key: str,
) -> tuple[str, list[StructuralRecord]]:
    """Apply GAIA prompt/question perturbations."""
    rng = random.Random(_stable_int(seed_key))
    original = text
    perturbed = text
    records: list[StructuralRecord] = []

    if strength in {"mild", "medium"}:
        perturbed = perturbed.lower()
        records.append(_record("gaia", strength, "prompt.case", "lowercase prompt"))
    elif strength == "severe":
        perturbed = _mixed_case(perturbed, rng)
        records.append(_record("gaia", strength, "prompt.case", "mixed-case prompt"))

    normalized = " ".join(perturbed.split())
    if normalized != perturbed:
        perturbed = normalized
        records.append(
            _record("gaia", strength, "prompt.whitespace", "normalized whitespace")
        )

    if strength in {"medium", "severe"}:
        perturbed = _format_dates_verbose(perturbed)
        perturbed = _format_numbers(perturbed, words=strength == "severe")
        perturbed = _rephrase_instructions(perturbed, terse=strength == "severe")
        perturbed = _reorder_bullets(perturbed, rng)
        records.append(
            _record(
                "gaia",
                strength,
                "prompt.format",
                "rephrased instructions and reformatted dates/numbers",
            )
        )

    if strength in {"medium", "severe"}:
        perturbed = f"{rng.choice(['Please', 'Kindly'])}, {perturbed} Thank you."
        records.append(_record("gaia", strength, "prompt.noise", "added polite wrapper"))

    if strength == "severe":
        perturbed = (
            f"{perturbed} The weather has been quite variable lately. "
            "Some reports mention cloudy mornings and clearer afternoons."
        )
        records.append(
            _record("gaia", strength, "prompt.context", "added irrelevant context")
        )

    return (perturbed, records) if perturbed != original else (text, [])


def perturb_gaia_observation_message(
    message: ChatMessage,
    *,
    strength: StructuralStrength,
) -> tuple[ChatMessage | None, list[StructuralRecord]]:
    """Perturb free-text tool observations for severe GAIA structural runs."""
    if strength != "severe":
        return None, []
    if isinstance(message, ChatMessageTool):
        return (
            message.model_copy(update={"content": _wrap_text_observation(message.text)}),
            [_record("gaia", strength, "tool_observation", "wrapped tool observation")],
        )
    content = getattr(message, "content", None)
    if isinstance(content, list):
        mutated = False
        new_content = []
        for item in content:
            if isinstance(item, ContentText) and _looks_like_text_observation(item.text):
                new_content.append(item.model_copy(update={"text": _wrap_text_observation(item.text)}))
                mutated = True
            else:
                new_content.append(item)
        if mutated:
            return (
                message.model_copy(update={"content": new_content}),
                [_record("gaia", strength, "tool_observation", "wrapped text content")],
            )
    elif isinstance(content, str) and _looks_like_text_observation(content):
        return (
            message.model_copy(update={"content": _wrap_text_observation(content)}),
            [_record("gaia", strength, "tool_observation", "wrapped text content")],
        )
    return None, []


def perturb_taubench_tool_params(
    params: ToolParams,
    *,
    strength: StructuralStrength,
) -> tuple[ToolParams, dict[str, str], list[StructuralRecord]]:
    """Rename tool parameters while preserving schema details."""
    if strength == "mild":
        style = "camel"
    elif strength == "medium":
        style = "camel"
    else:
        style = "abbrev"

    updated = params.model_copy(deep=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    mapping: dict[str, str] = {}
    records: list[StructuralRecord] = []

    for original_name, schema in params.properties.items():
        new_name = _transform_param_name(original_name, style)
        properties[new_name] = schema
        if original_name in params.required:
            required.append(new_name)
        if new_name != original_name:
            mapping[new_name] = original_name
            records.append(
                _record(
                    "taubench",
                    strength,
                    original_name,
                    f"renamed parameter {original_name} -> {new_name}",
                )
            )

    updated.properties = properties
    updated.required = required
    return updated, mapping, records


def reverse_tool_args(
    tool_name: str,
    kwargs: dict[str, Any],
    param_mapping: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Map perturbed tool-call kwargs back to original tool kwargs."""
    mapping = param_mapping.get(tool_name, {})
    return {mapping.get(key, key): value for key, value in kwargs.items()}


def perturb_taubench_tool_result(
    result: Any,
    *,
    strength: StructuralStrength,
    target: str,
) -> tuple[Any, list[StructuralRecord]]:
    """Perturb JSON-like TauBench tool results."""
    parsed = _try_json_loads(result)
    perturbed = _perturb_taubench_data(parsed, strength=strength)
    if strength in {"medium", "severe"}:
        perturbed = {"status": "success", "data": perturbed}
    output = json.dumps(perturbed, separators=(",", ":")) if isinstance(result, str) else perturbed
    return (
        output,
        [_record("taubench", strength, target, "perturbed tool response")],
    )


def _perturb_taubench_data(data: Any, *, strength: StructuralStrength) -> Any:
    if isinstance(data, dict):
        return {
            _transform_response_key(key, strength): _perturb_taubench_data(
                value,
                strength=strength,
            )
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_perturb_taubench_data(item, strength=strength) for item in data]
    if isinstance(data, str):
        return _perturb_taubench_value(data, strength)
    return data


def _perturb_taubench_value(value: str, strength: StructuralStrength) -> str:
    if strength == "severe":
        date_value = _format_iso_date_compact(value)
        if date_value != value:
            return date_value
        time_value = _format_time_compact(value)
        if time_value != value:
            return time_value
        abbreviated = _VALUE_ABBREVIATIONS.get(value.lower(), value)
        if abbreviated != value:
            return abbreviated
    if strength in {"medium", "severe"}:
        value = _format_iso_date_us(value)
        value = _format_time_12h(value)
        if value in {"confirmed", "cancelled", "pending", "completed"}:
            value = value.upper()
        if value in {"basic_economy", "economy", "business", "first"}:
            value = value.replace("_", " ").title()
    return value


def _transform_param_name(name: str, style: str) -> str:
    if name in _EXECUTION_ONLY_FIELDS:
        return name
    if style == "abbrev" and name in _PARAM_ABBREVIATIONS:
        return _PARAM_ABBREVIATIONS[name]
    return _snake_to_camel(name)


def _transform_response_key(key: str, strength: StructuralStrength) -> str:
    if strength == "severe":
        key = _KEY_ABBREVIATIONS.get(key, key)
    return _snake_to_camel(key)


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _try_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _mixed_case(text: str, rng: random.Random) -> str:
    return "".join(
        char.upper() if char.isalpha() and rng.random() > 0.5 else char.lower()
        for char in text
    )


def _format_dates_verbose(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        year, month, day = match.groups()
        month_index = int(month) - 1
        if not 0 <= month_index < 12:
            return match.group(0)
        return f"{_MONTHS[month_index]} {int(day)}, {year}"

    return re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", replace, text)


def _format_numbers(text: str, *, words: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        value = int(match.group(0))
        if 1900 <= value <= 2100:
            return match.group(0)
        if words and value in _SMALL_NUMBER_WORDS:
            return _SMALL_NUMBER_WORDS[value]
        if not words and abs(value) >= 1000:
            return f"{value:,}"
        return match.group(0)

    return re.sub(r"\b\d+\b", replace, text)


def _rephrase_instructions(text: str, *, terse: bool) -> str:
    replacements = {
        "return only your answer": "answer only"
        if terse
        else "please provide exclusively your final answer",
        "short phrase with as few words as possible": "brief phrase"
        if terse
        else "concise phrase using minimal words",
        "without any units": "no units"
        if terse
        else "excluding units",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _reorder_bullets(text: str, rng: random.Random) -> str:
    lines = text.splitlines()
    bullet_indexes = [index for index, line in enumerate(lines) if line.strip().startswith("-")]
    if len(bullet_indexes) < 2:
        return text
    shuffled = [lines[index] for index in bullet_indexes]
    rng.shuffle(shuffled)
    for index, line in zip(bullet_indexes, shuffled, strict=True):
        lines[index] = line
    return "\n".join(lines)


def _wrap_text_observation(text: str) -> str:
    return (
        "[Response Status: OK]\n"
        "[Data Begin]\n"
        f"{text}\n"
        "[Data End]\n"
        "[Navigation: Home > Results]\n"
    )


def _looks_like_text_observation(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("result", "output", "search", "webpage", "command", "stdout")
    )


def _format_iso_date_us(value: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not match:
        return value
    year, month, day = match.groups()
    return f"{month}/{day}/{year}"


def _format_iso_date_compact(value: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not match:
        return value
    return "".join(match.groups())


def _format_time_12h(value: str) -> str:
    match = re.fullmatch(r"(\d{2}):(\d{2})(?::\d{2})?", value)
    if not match:
        return value
    hour = int(match.group(1))
    minute = match.group(2)
    suffix = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return f"{hour_12}:{minute} {suffix}"


def _format_time_compact(value: str) -> str:
    match = re.fullmatch(r"(\d{2}):(\d{2})(?::\d{2})?", value)
    if not match:
        return value
    return f"{match.group(1)}{match.group(2)}"


def _append_structural_metadata(
    state: TaskState,
    env: StructuralEnvironment,
) -> TaskState:
    metadata = dict(getattr(state, "metadata", {}) or {})
    metadata["reliability_structural"] = {
        "kind": env.kind,
        "strength": env.strength,
        "applied_perturbations": [record.model_dump() for record in env.records],
    }
    state.metadata = metadata
    return state


def _record(
    kind: StructuralKind,
    strength: StructuralStrength,
    target: str,
    summary: str,
) -> StructuralRecord:
    return StructuralRecord(
        kind=kind,
        strength=strength,
        target=target,
        summary=summary,
    )


def _message_identity(message: ChatMessage) -> str:
    return _stable_digest(
        ":".join(
            [
                str(getattr(message, "role", "")),
                str(getattr(message, "tool_call_id", "")),
                str(getattr(message, "function", "")),
                getattr(message, "text", ""),
            ]
        )
    )


def _stable_int(value: str) -> int:
    return int(_stable_digest(value), 16)


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


_active_structural_env: ContextVar[StructuralEnvironment | None] = ContextVar(
    "active_structural_env", default=None
)
