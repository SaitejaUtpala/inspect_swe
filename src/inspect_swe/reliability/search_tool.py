"""Inspect-owned search tool for optional reliability fault experiments."""

from __future__ import annotations

from typing import Protocol

from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import Tool, tool

DEFAULT_SEARCH_RESULT = (
    "Search result unavailable from the deterministic reliability search provider. "
    "Use the query terms and continue with the best supported answer."
)


class SearchProvider(Protocol):
    """Provider interface for the Inspect-owned reliability search tool."""

    async def search(self, query: str) -> str:
        """Return search results for a query."""


class DeterministicSearchProvider:
    """Offline search provider used for tests and smoke runs."""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.results = results or {}

    async def search(self, query: str) -> str:
        """Return a deterministic result for the query."""
        return self.results.get(query, DEFAULT_SEARCH_RESULT)


def reliability_search_tool(provider: SearchProvider | None = None) -> Tool:
    """Create an Inspect-owned web-search-like tool."""
    search_provider = provider or DeterministicSearchProvider()

    @tool
    def reliability_search() -> Tool:
        """Search through an Inspect-owned reliability wrapper."""

        async def execute(query: str) -> str:
            """Search for information related to a query.

            Args:
                query: Search query.
            """
            return await search_provider.search(query)

        return execute

    return reliability_search()


def reliability_search_bridged_tools(
    provider: SearchProvider | None = None,
    *,
    server_name: str = "reliability_search",
) -> list[BridgedToolsSpec]:
    """Create bridged tools for the Inspect-owned reliability search wrapper."""
    return [
        BridgedToolsSpec(
            name=server_name,
            tools=[reliability_search_tool(provider)],
        )
    ]
