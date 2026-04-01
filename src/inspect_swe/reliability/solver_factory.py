"""Shared agent solver construction for reliability phases."""

from __future__ import annotations

from typing import Any


def default_solver_for_agent(
    agent: str,
    *,
    generate_filter: Any = None,
    bridged_tools: Any = None,
) -> Any | None:
    """Build the default inspect_swe solver for an agent."""
    kwargs: dict[str, Any] = {}
    if generate_filter is not None:
        kwargs["filter"] = generate_filter
    if bridged_tools is not None:
        kwargs["bridged_tools"] = bridged_tools

    if agent == "codex_cli":
        from inspect_swe._codex_cli.codex_cli import codex_cli

        return codex_cli(**kwargs)
    if agent == "claude_code":
        from inspect_swe._claude_code.claude_code import claude_code

        return claude_code(**kwargs)
    if agent == "gemini_cli":
        from inspect_swe._gemini_cli.gemini_cli import gemini_cli

        return gemini_cli(**kwargs)
    if agent == "mini_swe_agent":
        from inspect_swe._mini_swe_agent.mini_swe_agent import mini_swe_agent

        return mini_swe_agent(**kwargs)
    return None
