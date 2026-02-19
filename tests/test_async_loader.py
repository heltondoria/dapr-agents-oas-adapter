"""Tests for the async loader module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dapr_agents_oas_adapter.async_loader import AsyncDaprAgentSpecLoader, run_sync
from dapr_agents_oas_adapter.types import DaprAgentConfig, WorkflowDefinition


class TestAsyncDaprAgentSpecLoader:
    """Tests for AsyncDaprAgentSpecLoader class."""

    @pytest.fixture
    def loader(self) -> AsyncDaprAgentSpecLoader:
        """Create a loader instance for testing."""
        return AsyncDaprAgentSpecLoader()

    @pytest.fixture
    def sample_agent_dict(self) -> dict:
        """Create sample agent dict for load_dict tests."""
        return {
            "component_type": "Agent",
            "name": "test_agent",
            "description": "A test agent",
        }

    @pytest.fixture
    def sample_workflow_dict(self) -> dict:
        """Create sample workflow dict for load_dict tests."""
        return {
            "component_type": "Flow",
            "name": "test_workflow",
            "description": "A test workflow",
        }

    def test_init_default(self) -> None:
        """Test default initialization."""
        loader = AsyncDaprAgentSpecLoader()
        assert loader.tool_registry == {}

    def test_init_with_tool_registry(self) -> None:
        """Test initialization with tool registry."""
        registry = {"my_tool": lambda: None}
        loader = AsyncDaprAgentSpecLoader(tool_registry=registry)
        assert "my_tool" in loader.tool_registry

    def test_init_creates_sync_loader(self) -> None:
        """Test initialization creates the underlying sync loader."""
        loader = AsyncDaprAgentSpecLoader()
        assert loader.get_sync_loader() is not None

    def test_tool_registry_setter(self) -> None:
        """Test setting tool registry."""
        loader = AsyncDaprAgentSpecLoader()
        new_registry = {"new_tool": lambda: None}
        loader.tool_registry = new_registry
        assert "new_tool" in loader.tool_registry

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        loader = AsyncDaprAgentSpecLoader()
        tool_fn = lambda x: x  # noqa: E731
        loader.register_tool("my_tool", tool_fn)
        assert "my_tool" in loader.tool_registry

    @pytest.mark.asyncio
    async def test_load_json(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test loading from JSON string via mocked sync loader."""
        # Mock the sync loader to avoid needing a full OAS spec
        mock_config = DaprAgentConfig(name="test_agent")
        with patch.object(loader._sync_loader, "load_json", return_value=mock_config):
            result = await loader.load_json('{"test": "json"}')
            assert isinstance(result, DaprAgentConfig)
            assert result.name == "test_agent"

    @pytest.mark.asyncio
    async def test_load_yaml(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test loading from YAML string via mocked sync loader."""
        mock_config = DaprAgentConfig(name="test_agent")
        with patch.object(loader._sync_loader, "load_yaml", return_value=mock_config):
            result = await loader.load_yaml("test: yaml")
            assert isinstance(result, DaprAgentConfig)
            assert result.name == "test_agent"

    @pytest.mark.asyncio
    async def test_load_json_file(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test loading from JSON file via mocked sync loader."""
        mock_config = DaprAgentConfig(name="test_agent")
        with patch.object(loader._sync_loader, "load_json_file", return_value=mock_config):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write("{}")
                f.flush()
                file_path = f.name

            try:
                result = await loader.load_json_file(file_path)
                assert isinstance(result, DaprAgentConfig)
                assert result.name == "test_agent"
            finally:
                Path(file_path).unlink()

    @pytest.mark.asyncio
    async def test_load_yaml_file(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test loading from YAML file via mocked sync loader."""
        mock_config = DaprAgentConfig(name="test_agent")
        with patch.object(loader._sync_loader, "load_yaml_file", return_value=mock_config):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write("test: yaml")
                f.flush()
                file_path = f.name

            try:
                result = await loader.load_yaml_file(file_path)
                assert isinstance(result, DaprAgentConfig)
                assert result.name == "test_agent"
            finally:
                Path(file_path).unlink()

    @pytest.mark.asyncio
    async def test_load_dict(
        self, loader: AsyncDaprAgentSpecLoader, sample_agent_dict: dict
    ) -> None:
        """Test loading from dictionary."""
        result = await loader.load_dict(sample_agent_dict)
        assert isinstance(result, DaprAgentConfig)
        assert result.name == "test_agent"

    @pytest.mark.asyncio
    async def test_load_multiple_files(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test loading multiple files concurrently via mocked sync loader."""
        mock_config1 = DaprAgentConfig(name="agent1")
        mock_config2 = DaprAgentConfig(name="agent2")

        files = []

        # Create temp files
        for i in range(2):
            suffix = ".json" if i == 0 else ".yaml"
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
                f.write("{}" if suffix == ".json" else "test: yaml")
                f.flush()
                files.append(f.name)

        try:
            with (
                patch.object(loader._sync_loader, "load_json_file", return_value=mock_config1),
                patch.object(loader._sync_loader, "load_yaml_file", return_value=mock_config2),
            ):
                results = await loader.load_multiple_files(files)
                assert len(results) == 2
                assert all(isinstance(r, DaprAgentConfig) for r in results)
        finally:
            for file_path in files:
                Path(file_path).unlink()

    @pytest.mark.asyncio
    async def test_get_sync_loader(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test getting the underlying sync loader."""
        sync_loader = loader.get_sync_loader()
        assert sync_loader is not None
        assert hasattr(sync_loader, "load_json")

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager usage."""
        async with AsyncDaprAgentSpecLoader() as loader:
            assert loader is not None
        # Loader should be closed after exiting context

    @pytest.mark.asyncio
    async def test_close(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test closing the loader."""
        await loader.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_create_agent(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test creating an agent."""
        config = DaprAgentConfig(
            name="test_agent",
            role="assistant",
            goal="help users",
        )

        # Mock the sync loader's create_agent method
        with patch.object(loader._sync_loader, "create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            result = await loader.create_agent(config)
            assert result is mock_agent

    @pytest.mark.asyncio
    async def test_create_workflow(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test creating a workflow."""
        from dapr_agents_oas_adapter.types import (
            WorkflowEdgeDefinition,
            WorkflowTaskDefinition,
        )

        workflow_def = WorkflowDefinition(
            name="test_workflow",
            tasks=[
                WorkflowTaskDefinition(name="start", task_type="start"),
                WorkflowTaskDefinition(name="end", task_type="end"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="start", to_node="end"),
            ],
            start_node="start",
            end_nodes=["end"],
        )

        # Mock the sync loader's create_workflow method
        with patch.object(loader._sync_loader, "create_workflow") as mock_create:
            mock_workflow = MagicMock()
            mock_create.return_value = mock_workflow

            result = await loader.create_workflow(workflow_def)
            assert result is mock_workflow

    @pytest.mark.asyncio
    async def test_load_component(self, loader: AsyncDaprAgentSpecLoader) -> None:
        """Test loading a component."""
        # Mock the sync loader's load_component method
        with patch.object(loader._sync_loader, "load_component") as mock_load:
            mock_config = DaprAgentConfig(name="component_agent")
            mock_load.return_value = mock_config

            component = MagicMock()
            result = await loader.load_component(component)
            assert result is mock_config


class TestRunSync:
    """Tests for run_sync utility function."""

    def test_run_sync_no_loop(self) -> None:
        """Test run_sync when no event loop is running."""

        async def async_func() -> str:
            return "result"

        result = run_sync(async_func())
        assert result == "result"

    def test_run_sync_with_args(self) -> None:
        """Test run_sync with async function that takes arguments."""

        async def async_add(a: int, b: int) -> int:
            return a + b

        result = run_sync(async_add(1, 2))
        assert result == 3

    def test_run_sync_exception(self) -> None:
        """Test run_sync when async function raises exception."""

        async def async_fail() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_sync(async_fail())


class TestAsyncLoaderIntegration:
    """Integration tests for async loader."""

    @pytest.mark.asyncio
    async def test_full_agent_loading_pipeline(self) -> None:
        """Test complete agent loading pipeline."""
        loader = AsyncDaprAgentSpecLoader()

        agent_spec = {
            "component_type": "Agent",
            "name": "integration_agent",
            "description": "An integration test agent",
            "prompt": "You are a helpful assistant.",
        }

        # Load from dict
        config = await loader.load_dict(agent_spec)
        assert isinstance(config, DaprAgentConfig)
        assert config.name == "integration_agent"

        await loader.close()

    @pytest.mark.asyncio
    async def test_concurrent_loading(self) -> None:
        """Test that concurrent loading works correctly."""
        import asyncio

        loader = AsyncDaprAgentSpecLoader()

        # Create multiple specs
        specs = [{"component_type": "Agent", "name": f"agent_{i}"} for i in range(5)]

        # Load all concurrently
        results = await asyncio.gather(*[loader.load_dict(spec) for spec in specs])

        assert len(results) == 5
        for i, result in enumerate(results):
            assert isinstance(result, DaprAgentConfig)
            assert result.name == f"agent_{i}"

        await loader.close()
