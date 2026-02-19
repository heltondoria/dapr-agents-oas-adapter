"""Tests for the logging module."""

import logging

from dapr_agents_oas_adapter.logging import get_logger, set_logger


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_stdlib_logger(self) -> None:
        """Test that get_logger returns a stdlib Logger."""
        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    def test_get_logger_default_name_is_dapr_agents_oas_adapter(self) -> None:
        """Test that the default logger name is 'dapr_agents_oas_adapter'."""
        logger = get_logger()
        assert logger.name == "dapr_agents_oas_adapter"


class TestSetLogger:
    """Tests for set_logger function."""

    def teardown_method(self) -> None:
        """Reset the logger after each test."""
        set_logger(logging.getLogger("dapr_agents_oas_adapter"))

    def test_set_logger_replaces_default(self) -> None:
        """Test that set_logger replaces the default logger."""
        custom = logging.getLogger("custom_test_logger")
        set_logger(custom)
        assert get_logger() is custom

    def test_set_logger_custom_logger_used_by_get_logger(self) -> None:
        """Test that after set_logger, get_logger returns the injected logger."""
        custom = logging.getLogger("injected")
        set_logger(custom)
        result = get_logger()
        assert result is custom
        assert result.name == "injected"

    def test_get_logger_after_set_logger_returns_injected(self) -> None:
        """Test that subsequent get_logger calls return the injected logger."""
        custom = logging.getLogger("persistent")
        set_logger(custom)
        assert get_logger() is custom
        assert get_logger() is custom


class TestLoggingIntegration:
    """Integration tests for logging in loader/exporter."""

    def test_loader_logs_operations(self) -> None:
        """Test that DaprAgentSpecLoader logs operations."""
        from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader

        loader = DaprAgentSpecLoader()
        assert hasattr(loader, "_logger")
        assert isinstance(loader._logger, logging.Logger)

    def test_exporter_logs_operations(self) -> None:
        """Test that DaprAgentSpecExporter logs operations."""
        from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter

        exporter = DaprAgentSpecExporter()
        assert hasattr(exporter, "_logger")
        assert isinstance(exporter._logger, logging.Logger)
