"""Compatibility checks for Inspect AI APIs used by inspect_swe agents."""

from __future__ import annotations

import inspect

from inspect_ai.agent import (
    AgentAttempts,
    BridgedToolsSpec,
    agent,
    agent_with,
    as_solver,
    is_agent,
    sandbox_agent_bridge,
)
from inspect_ai.model import ChatMessageTool, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolError, ToolInfo, ToolParams
from inspect_ai.util import StoreModel, checkpointer, store_as


def test_sandbox_agent_bridge_supports_checkpointer_kwarg() -> None:
    """Merged agent checkpoint support requires a newer Inspect AI bridge API."""
    signature = inspect.signature(sandbox_agent_bridge)

    assert "checkpointer" in signature.parameters, (
        "Installed inspect_ai is too old for inspect_swe checkpoint-enabled agents. "
        "Update inspect_ai from the dependency in pyproject.toml before running "
        "Claude Code or Codex CLI evaluations."
    )


def test_sandbox_agent_bridge_supports_model_event_sink_kwarg() -> None:
    signature = inspect.signature(sandbox_agent_bridge)

    assert "model_event_sink" in signature.parameters


def test_sandbox_agent_bridge_supports_model_alias_protocol_kwargs() -> None:
    signature = inspect.signature(sandbox_agent_bridge)

    for name in ["model", "model_aliases", "filter", "retry_refusals"]:
        assert name in signature.parameters


def test_sandbox_agent_bridge_supports_bridged_tool_kwargs() -> None:
    signature = inspect.signature(sandbox_agent_bridge)

    for name in ["sandbox", "port", "bridged_tools"]:
        assert name in signature.parameters


def test_checkpointer_factory_returns_async_context_manager() -> None:
    context = checkpointer()

    assert hasattr(context, "__aenter__")
    assert hasattr(context, "__aexit__")


def test_bridged_tools_spec_exposes_name_and_tools_fields() -> None:
    spec = BridgedToolsSpec(name="reliability_search", tools=[])

    assert spec.name == "reliability_search"
    assert spec.tools == []


def test_agent_attempts_accepts_int_like_configuration() -> None:
    attempts = AgentAttempts(2)

    assert attempts.attempts == 2
    assert isinstance(attempts.incorrect_message, str)


def test_agent_decorator_helpers_remain_importable_and_callable() -> None:
    for helper in [agent, agent_with, as_solver, is_agent]:
        assert callable(helper)


def test_chat_message_tool_preserves_observation_protocol_fields() -> None:
    message = ChatMessageTool(
        content="Command exited with code 0",
        function="exec_command",
        tool_call_id="call_exec",
    )

    assert message.role == "tool"
    assert message.text == "Command exited with code 0"
    assert message.function == "exec_command"
    assert message.tool_call_id == "call_exec"


def test_model_output_from_content_protocol_is_available() -> None:
    output = ModelOutput.from_content("mock/model", "generated text")

    assert output.model == "mock/model"
    assert output.completion == "generated text"


def test_tool_schema_protocol_supports_required_properties() -> None:
    params = ToolParams(
        properties={"flight_number": {"type": "string"}},
        required=["flight_number"],
    )
    info = ToolInfo(name="lookup_flight", description="Look up a flight.", parameters=params)

    assert info.parameters.properties["flight_number"].type == "string"
    assert info.parameters.required == ["flight_number"]


def test_generate_config_supports_reliability_concurrency_options() -> None:
    config = GenerateConfig(max_connections=3, max_retries=2)

    assert config.max_connections == 3
    assert config.max_retries == 2


def test_tool_error_remains_exception_with_message() -> None:
    error = ToolError("synthetic tool failure")

    assert isinstance(error, Exception)
    assert str(error) == "synthetic tool failure"


def test_store_model_and_store_as_protocols_are_available() -> None:
    class SessionState(StoreModel):
        session_id: str | None = None

    assert SessionState(session_id="abc").session_id == "abc"
    assert callable(store_as)
