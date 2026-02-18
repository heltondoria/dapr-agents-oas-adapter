"""Tests for package-level functionality."""

import tomllib
from pathlib import Path

import dapr_agents_oas_adapter


def test_version_matches_pyproject() -> None:
    """Verify __version__ matches pyproject.toml version (single source of truth)."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    expected_version = pyproject["project"]["version"]
    assert dapr_agents_oas_adapter.__version__ == expected_version


def test_version_is_string() -> None:
    """Verify __version__ is a string."""
    assert isinstance(dapr_agents_oas_adapter.__version__, str)


def test_version_is_not_empty() -> None:
    """Verify __version__ is not empty."""
    assert len(dapr_agents_oas_adapter.__version__) > 0


def test_public_api_exports() -> None:
    """Verify all expected public API symbols are exported."""
    assert hasattr(dapr_agents_oas_adapter, "DaprAgentSpecLoader")
    assert hasattr(dapr_agents_oas_adapter, "DaprAgentSpecExporter")
    assert hasattr(dapr_agents_oas_adapter, "IDGenerator")
    assert hasattr(dapr_agents_oas_adapter, "WorkflowValidator")
    assert hasattr(dapr_agents_oas_adapter, "ValidationResult")
    assert hasattr(dapr_agents_oas_adapter, "WorkflowValidationError")
    assert hasattr(dapr_agents_oas_adapter, "validate_workflow")
    assert hasattr(dapr_agents_oas_adapter, "__version__")


def test_all_exports_match() -> None:
    """Verify __all__ contains expected exports."""
    expected = {
        # Core loaders and exporters
        "AsyncDaprAgentSpecLoader",
        "CachedLoader",
        "DaprAgentSpecExporter",
        "DaprAgentSpecLoader",
        "StrictLoader",
        # Caching
        "CacheBackend",
        "CacheStats",
        "InMemoryCache",
        # Exceptions
        "ConversionError",
        "DaprAgentsOasAdapterError",
        "ValidationError",
        # Validation
        "IDGenerator",
        "OASSchemaValidationError",
        "OASSchemaValidator",
        "ValidationResult",
        "WorkflowValidationError",
        "WorkflowValidator",
        "validate_oas_dict",
        "validate_workflow",
        # Logging
        "get_logger",
        "set_logger",
        # Utilities
        "run_sync",
    }
    assert set(dapr_agents_oas_adapter.__all__) == expected
