"""Inspect-owned search tool for reliability fault injection."""

from __future__ import annotations

import os
from typing import Protocol

import httpx
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import Tool, ToolError, tool

DEFAULT_SEARCH_RESULT = (
    "Search result unavailable from the deterministic reliability search provider. "
    "Use the query terms and continue with the best supported answer."
)


class SearchProvider(Protocol):
    """Provider interface for the Inspect-owned reliability search tool."""

    async def search(self, query: str) -> str:
        """Return search results for a query."""


class DeterministicSearchProvider:
    """Offline search provider used for tests and slice runs."""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.results = results or {}

    async def search(self, query: str) -> str:
        """Return a deterministic result for the query."""
        return self.results.get(query, DEFAULT_SEARCH_RESULT)


class SerpApiSearchProvider:
    """Minimal SERP API provider for real wrapped-search experiments."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://serpapi.com/search.json",
    ) -> None:
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            raise ToolError("SERPAPI_API_KEY is required for SerpApiSearchProvider.")

    async def search(self, query: str) -> str:
        """Run a SERP API search and return a compact text result."""
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                self.base_url,
                params={"q": query, "api_key": self.api_key, "engine": "google"},
            )
            response.raise_for_status()
        data = response.json()
        organic_results = data.get("organic_results") or []
        if not organic_results:
            return "Search returned no organic results."
        rendered: list[str] = []
        for result in organic_results[:5]:
            title = result.get("title") or "Untitled"
            link = result.get("link") or ""
            snippet = result.get("snippet") or ""
            rendered.append(f"- {title}\n  {snippet}\n  {link}".strip())
        return "\n".join(rendered)


def reliability_search_tool(provider: SearchProvider | None = None) -> Tool:
    """Create an Inspect-owned web-search-like tool."""
    search_provider = provider or DeterministicSearchProvider()

    @tool
    def reliability_search() -> Tool:
        """Search the web using the Inspect-owned reliability search wrapper."""

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
