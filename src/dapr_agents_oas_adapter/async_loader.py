"""Async support for dapr-agents-oas-adapter.

Provides async versions of loading operations for concurrent processing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dapr_agents_oas_adapter.logging import get_logger, log_operation
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    NamedCallable,
    ToolRegistry,
    WorkflowDefinition,
)


class AsyncDaprAgentSpecLoader:
    """Async loader for converting OAS specifications to Dapr Agents components.

    This class provides async methods to load OAS (Open Agent Spec) configurations
    from JSON or YAML format and convert them to Dapr Agents components.

    The async operations use a thread pool executor for file I/O operations
    while keeping the conversion logic synchronous (since it's CPU-bound).

    Example:
        ```python
        async def main():
            loader = AsyncDaprAgentSpecLoader(
                tool_registry={
                    "search_tool": search_function,
                }
            )

            # Load multiple files concurrently
            configs = await asyncio.gather(
                loader.load_yaml_file("agent1.yaml"),
                loader.load_yaml_file("agent2.yaml"),
                loader.load_json_file("workflow.json"),
            )
        ```
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        *,
        max_workers: int | None = None,
    ) -> None:
        """Initialize the async loader.

        Args:
            tool_registry: Dictionary mapping tool names to their callable
                          implementations.
            max_workers: Maximum number of worker threads for file I/O
                        (default: None, uses ThreadPoolExecutor default)
        """
        # Import here to avoid circular imports
        from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader

        self._sync_loader = DaprAgentSpecLoader(tool_registry)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._logger = get_logger("AsyncDaprAgentSpecLoader")

    @property
    def tool_registry(self) -> ToolRegistry:
        """Get the current tool registry."""
        return self._sync_loader.tool_registry

    @tool_registry.setter
    def tool_registry(self, registry: ToolRegistry) -> None:
        """Update the tool registry."""
        self._sync_loader.tool_registry = registry

    def register_tool(self, name: str, implementation: Callable[..., Any]) -> None:
        """Register a tool implementation.

        Args:
            name: The tool name as defined in the OAS specification
            implementation: The callable implementation
        """
        self._sync_loader.register_tool(name, implementation)

    async def load_json(self, json_content: str) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from JSON string.

        Args:
            json_content: JSON string containing the OAS specification

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the JSON cannot be parsed or converted
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_loader.load_json,
            json_content,
        )

    async def load_yaml(self, yaml_content: str) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from YAML string.

        Args:
            yaml_content: YAML string containing the OAS specification

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the YAML cannot be parsed or converted
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_loader.load_yaml,
            yaml_content,
        )

    async def load_json_file(self, file_path: str | Path) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the file cannot be read or converted
        """
        with log_operation(
            "async_load_json_file",
            self._logger,
            file_path=str(file_path),
        ):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._sync_loader.load_json_file,
                file_path,
            )

    async def load_yaml_file(self, file_path: str | Path) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the file cannot be read or converted
        """
        with log_operation(
            "async_load_yaml_file",
            self._logger,
            file_path=str(file_path),
        ):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._sync_loader.load_yaml_file,
                file_path,
            )

    async def load_dict(self, spec_dict: dict[str, Any]) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a dictionary.

        Args:
            spec_dict: Dictionary containing the OAS specification

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the component type is not supported
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_loader.load_dict,
            spec_dict,
        )

    async def load_component(self, component: Any) -> DaprAgentConfig | WorkflowDefinition:
        """Load a PyAgentSpec Component and convert to Dapr format.

        Args:
            component: The OAS Component (or object) to convert

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the component type is not supported
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_loader.load_component,
            component,
        )

    async def create_agent(
        self,
        config: DaprAgentConfig,
        additional_tools: dict[str, Callable[..., Any]] | None = None,
    ) -> Any:
        """Create an executable Dapr Agent from configuration.

        Args:
            config: The agent configuration
            additional_tools: Additional tool implementations to include

        Returns:
            A Dapr Agent instance (AssistantAgent or ReActAgent)

        Raises:
            ConversionError: If agent creation fails
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._sync_loader.create_agent(config, additional_tools),
        )

    async def create_workflow(
        self,
        workflow_def: WorkflowDefinition,
        task_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> NamedCallable:
        """Create an executable Dapr workflow from definition.

        Args:
            workflow_def: The workflow definition
            task_implementations: Task implementations for workflow activities

        Returns:
            A workflow function that can be registered with Dapr

        Raises:
            ConversionError: If workflow creation fails
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._sync_loader.create_workflow(workflow_def, task_implementations),
        )

    async def load_multiple_files(
        self,
        file_paths: list[str | Path],
    ) -> list[DaprAgentConfig | WorkflowDefinition]:
        """Load multiple OAS specification files concurrently.

        Args:
            file_paths: List of file paths (JSON or YAML)

        Returns:
            List of loaded configurations in the same order as input

        Raises:
            ConversionError: If any file cannot be loaded
        """

        async def load_file(path: str | Path) -> DaprAgentConfig | WorkflowDefinition:
            path_str = str(path)
            if path_str.endswith((".yaml", ".yml")):
                return await self.load_yaml_file(path)
            return await self.load_json_file(path)

        with log_operation(
            "async_load_multiple_files",
            self._logger,
            file_count=len(file_paths),
        ):
            return await asyncio.gather(*[load_file(p) for p in file_paths])

    def get_sync_loader(self) -> Any:
        """Get the underlying synchronous loader.

        Returns:
            The DaprAgentSpecLoader instance
        """
        return self._sync_loader

    async def close(self) -> None:
        """Close the async loader and release resources."""
        self._executor.shutdown(wait=False)
        self._logger.debug("async_loader_closed")

    async def __aenter__(self) -> AsyncDaprAgentSpecLoader:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        del exc_type, exc_val, exc_tb  # Unused but required by protocol
        await self.close()


def run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    This is a utility function for running async code from sync contexts.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Example:
        ```python
        from dapr_agents_oas_adapter.async_loader import AsyncDaprAgentSpecLoader, run_sync

        async def load_async():
            loader = AsyncDaprAgentSpecLoader()
            return await loader.load_yaml_file("agent.yaml")

        # Run from sync code
        result = run_sync(load_async())
        ```
    """
    try:
        asyncio.get_running_loop()
        # Already in an async context, create a new thread to run it
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running loop, create a new one
        return asyncio.run(coro)
