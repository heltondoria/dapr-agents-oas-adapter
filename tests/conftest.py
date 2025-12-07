"""Pytest configuration and fixtures."""

from collections.abc import Callable
from typing import Any

import pytest

from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader


@pytest.fixture
def sample_tool_registry() -> dict[str, Callable[..., Any]]:
    """Provide a sample tool registry for testing."""
    def search_tool(query: str) -> list[str]:
        """Search the web."""
        return [f"Result for: {query}"]

    def calculator_tool(expression: str) -> float:
        """Calculate an expression."""
        return eval(expression)  # noqa: S307

    return {
        "search": search_tool,
        "calculator": calculator_tool,
    }


@pytest.fixture
def loader(sample_tool_registry: dict[str, Callable[..., Any]]) -> DaprAgentSpecLoader:
    """Provide a configured loader instance."""
    return DaprAgentSpecLoader(tool_registry=sample_tool_registry)


@pytest.fixture
def exporter() -> DaprAgentSpecExporter:
    """Provide an exporter instance."""
    return DaprAgentSpecExporter()
