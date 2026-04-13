"""Shared agent solver construction for reliability phases."""

from __future__ import annotations

from typing import Any

from .solver_signals import reliability_instrumented_solver


def default_solver_for_agent(
    agent: str,
    *,
    generate_filter: Any = None,
    bridged_tools: Any = None,
    message_transform: Any = None,
) -> Any | None:
    """Build the default inspect_swe solver for an agent."""
    kwargs: dict[str, Any] = {}
    if generate_filter is not None:
        kwargs["filter"] = generate_filter
    if bridged_tools is not None:
        kwargs["bridged_tools"] = bridged_tools

    if agent == "codex_cli":
        from inspect_swe._codex_cli.codex_cli import codex_cli

        return reliability_instrumented_solver(
            codex_cli(**kwargs),
            message_transform=message_transform,
        )
    if agent == "claude_code":
        from inspect_swe._claude_code.claude_code import claude_code

        return reliability_instrumented_solver(
            claude_code(**kwargs),
            message_transform=message_transform,
        )
    if agent == "gemini_cli":
        from inspect_swe._gemini_cli.gemini_cli import gemini_cli

        return reliability_instrumented_solver(
            gemini_cli(**kwargs),
            message_transform=message_transform,
        )
    if agent == "mini_swe_agent":
        from inspect_swe._mini_swe_agent.mini_swe_agent import mini_swe_agent

        return reliability_instrumented_solver(
            mini_swe_agent(**kwargs),
            message_transform=message_transform,
        )
    return None
