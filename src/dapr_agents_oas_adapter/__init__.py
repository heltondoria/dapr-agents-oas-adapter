"""Dapr Agents Open Agent Spec Adapter.

This library provides bidirectional conversion between Open Agent Spec (OAS)
configurations and Dapr Agents components, enabling:
- Import OAS specifications to create Dapr Agents and Workflows
- Export Dapr Agents and Workflows to OAS format
"""

from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader

__version__ = "0.1.0"
__all__ = [
    "DaprAgentSpecExporter",
    "DaprAgentSpecLoader",
]
