"""TauBench adapters for reliability runs."""

from __future__ import annotations

from typing import Any

from inspect_ai.agent import AgentState, BridgedToolsSpec
from inspect_ai.model import ChatMessageSystem, GenerateConfig, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store_as
from inspect_evals.tau2.airline.agents import get_domain_policy
from inspect_evals.tau2.airline.tools import all_airline_tools
from inspect_evals.tau2.common.agents import (
    DEFAULT_MESSAGE_LIMIT,
    StoredAgentState,
    TrajectoryMessage,
    flip_assistant_message_to_user,
    generate_loop,
)

from inspect_swe import codex_cli

from .baseline import RELIABILITY_CODEX_CLI_VERSION
from .structural_perturbations import StructuralEnvironment


def tau2_airline_bridged_tools(
    structural_env: StructuralEnvironment | None = None,
) -> list[BridgedToolsSpec]:
    """Return the real Tau2 Airline tools as one bridged MCP server."""
    tools = [tool_factory() for tool_factory in all_airline_tools]
    bridged_tools = [BridgedToolsSpec(name="tau2_airline", tools=tools)]
    if structural_env is None:
        return bridged_tools
    return structural_env.wrap_bridged_tools(bridged_tools) or bridged_tools


def tau2_airline_codex_solver(
    *,
    structural_env: StructuralEnvironment | None = None,
    message_limit: int | None = None,
    codex_kwargs: dict[str, Any] | None = None,
) -> Solver:
    """Run Tau2 Airline with Codex as the service agent.

    The Tau2 user simulator remains the task driver. Only the assistant/service
    agent side is replaced with Codex, and Codex sees the airline tools through
    the same bridged-tool surface used for structural API perturbations.
    """
    limit = message_limit or DEFAULT_MESSAGE_LIMIT
    kwargs = dict(codex_kwargs or {})
    kwargs.setdefault("retry_refusals", 3)
    kwargs.setdefault("version", RELIABILITY_CODEX_CLI_VERSION)
    kwargs.setdefault("disallowed_tools", ["web_search"])
    kwargs["bridged_tools"] = tau2_airline_bridged_tools(structural_env)
    codex_agent = codex_cli(**kwargs)

    @solver(name="tau2_airline_codex_solver")
    def tau2_codex_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            del generate
            user_model = get_model(role="user", config=GenerateConfig(temperature=0.0))
            user_state = AgentState(messages=state.messages)
            stored_agent_state = store_as(StoredAgentState, "AirlineAgent")
            stored_agent_state.messages = [_service_agent_system_message()]
            stored_agent_state.trajectory = []

            for _ in range(limit):
                user_output = await generate_loop(
                    "user",
                    user_model,
                    user_state.messages,
                    tools=[],
                    trajectory=stored_agent_state.trajectory,
                )
                if _is_tau2_stop(user_output.completion):
                    state.messages = user_state.messages
                    state.output = user_output
                    return state

                stored_agent_state.messages.append(
                    flip_assistant_message_to_user(user_output.message)
                )
                before_agent_turn = len(stored_agent_state.messages)
                codex_state = AgentState(messages=stored_agent_state.messages)
                codex_state = await codex_agent(codex_state)
                new_messages = codex_state.messages[before_agent_turn:]
                stored_agent_state.messages = codex_state.messages
                stored_agent_state.trajectory.extend(
                    TrajectoryMessage(role="assistant", message=message)
                    for message in new_messages
                )
                user_state.messages.append(
                    flip_assistant_message_to_user(codex_state.output.message)
                )

            state.messages = user_state.messages
            state.output = user_state.output
            return state

        return solve

    return tau2_codex_solver()


def _service_agent_system_message() -> ChatMessageSystem:
    return ChatMessageSystem(
        content=f"""<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
</instructions>
{get_domain_policy()}
"""
    )


def _is_tau2_stop(completion: str) -> bool:
    return any(
        marker in completion
        for marker in ("###STOP###", "###OUT-OF-SCOPE###", "###TRANSFER###")
    )
