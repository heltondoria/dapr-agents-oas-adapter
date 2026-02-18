"""Dapr Agents Open Agent Spec Adapter.

This library provides bidirectional conversion between Open Agent Spec (OAS)
configurations and Dapr Agents components, enabling:
- Import OAS specifications to create Dapr Agents and Workflows
- Export Dapr Agents and Workflows to OAS format
"""

from importlib.metadata import version

from dapr_agents_oas_adapter.async_loader import AsyncDaprAgentSpecLoader, run_sync
from dapr_agents_oas_adapter.cache import (
    CacheBackend,
    CachedLoader,
    CacheStats,
    InMemoryCache,
)
from dapr_agents_oas_adapter.exceptions import (
    ConversionError,
    DaprAgentsOasAdapterError,
    ValidationError,
)
from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader, StrictLoader
from dapr_agents_oas_adapter.logging import get_logger, set_logger
from dapr_agents_oas_adapter.utils import IDGenerator
from dapr_agents_oas_adapter.validation import (
    OASSchemaValidationError,
    OASSchemaValidator,
    ValidationResult,
    WorkflowValidationError,
    WorkflowValidator,
    validate_oas_dict,
    validate_workflow,
)

__version__ = version("dapr-agents-oas-adapter")
__all__ = [
    "AsyncDaprAgentSpecLoader",
    "CacheBackend",
    "CacheStats",
    "CachedLoader",
    "ConversionError",
    "DaprAgentSpecExporter",
    "DaprAgentSpecLoader",
    "DaprAgentsOasAdapterError",
    "IDGenerator",
    "InMemoryCache",
    "OASSchemaValidationError",
    "OASSchemaValidator",
    "StrictLoader",
    "ValidationError",
    "ValidationResult",
    "WorkflowValidationError",
    "WorkflowValidator",
    "get_logger",
    "run_sync",
    "set_logger",
    "validate_oas_dict",
    "validate_workflow",
]
