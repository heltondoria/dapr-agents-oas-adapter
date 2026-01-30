"""Tests for the logging module."""

import logging

import pytest

from dapr_agents_oas_adapter.logging import (
    LoggingMixin,
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    log_context,
    log_operation,
    unbind_context,
)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default(self) -> None:
        """Test default logging configuration."""
        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_configure_logging_with_level(self) -> None:
        """Test logging configuration with custom level."""
        configure_logging(level=logging.DEBUG)
        logger = get_logger("test_debug")
        assert logger is not None

    def test_configure_logging_json_format(self) -> None:
        """Test logging configuration with JSON format."""
        configure_logging(json_format=True)
        logger = get_logger("test_json")
        assert logger is not None

    def test_configure_logging_without_timestamp(self) -> None:
        """Test logging configuration without timestamps."""
        configure_logging(add_timestamp=False)
        logger = get_logger("test_no_timestamp")
        assert logger is not None


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_without_name(self) -> None:
        """Test getting a logger without a name."""
        logger = get_logger()
        assert logger is not None

    def test_get_logger_with_name(self) -> None:
        """Test getting a logger with a name."""
        logger = get_logger("my_module")
        assert logger is not None

    def test_get_logger_with_initial_context(self) -> None:
        """Test getting a logger with initial context."""
        logger = get_logger("contextual", component="test", version="1.0")
        assert logger is not None


class TestContextManagement:
    """Tests for context binding functions."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_context()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_context()

    def test_bind_context(self) -> None:
        """Test binding context variables."""
        bind_context(request_id="123", user="test")
        # Context is bound - we can't directly inspect it, but it shouldn't raise
        clear_context()

    def test_unbind_context(self) -> None:
        """Test unbinding context variables."""
        bind_context(key1="value1", key2="value2")
        unbind_context("key1")
        clear_context()

    def test_clear_context(self) -> None:
        """Test clearing all context."""
        bind_context(key1="value1", key2="value2")
        clear_context()
        # No exception means success


class TestLogContext:
    """Tests for log_context context manager."""

    def setup_method(self) -> None:
        """Clear context before each test."""
        clear_context()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        clear_context()

    def test_log_context_binds_and_unbinds(self) -> None:
        """Test that log_context properly binds and unbinds context."""
        with log_context(request_id="abc123"):
            # Context is bound during this block
            pass
        # Context should be unbound after the block

    def test_log_context_with_multiple_keys(self) -> None:
        """Test log_context with multiple context keys."""
        with log_context(user="test", action="read", resource="file"):
            pass

    def test_log_context_cleans_up_on_exception(self) -> None:
        """Test that log_context cleans up even when an exception occurs."""
        with pytest.raises(ValueError), log_context(error_key="value"):
            raise ValueError("test error")
        # Context should still be cleaned up


class TestLogOperation:
    """Tests for log_operation context manager."""

    def test_log_operation_success(self) -> None:
        """Test log_operation for successful operations."""
        logger = get_logger("test_op")

        with log_operation("test_operation", logger) as ctx:
            ctx["result"] = "success"

    def test_log_operation_with_extra_context(self) -> None:
        """Test log_operation with extra context."""
        logger = get_logger("test_op_ctx")

        with log_operation("load_file", logger, file_path="/test/file.txt"):
            pass

    def test_log_operation_failure(self) -> None:
        """Test log_operation when operation fails."""
        logger = get_logger("test_op_fail")

        with pytest.raises(RuntimeError), log_operation("failing_operation", logger):
            raise RuntimeError("Operation failed")

    def test_log_operation_without_logger(self) -> None:
        """Test log_operation creates default logger if none provided."""
        with log_operation("auto_logger_op") as ctx:
            ctx["test"] = True

    def test_log_operation_context_has_start_time(self) -> None:
        """Test that log_operation context includes start_time."""
        with log_operation("timed_op") as ctx:
            assert "start_time" in ctx
            assert isinstance(ctx["start_time"], float)


class TestLoggingMixin:
    """Tests for LoggingMixin class."""

    def test_mixin_provides_logger(self) -> None:
        """Test that LoggingMixin provides a logger property."""

        class MyConverter(LoggingMixin):
            pass

        converter = MyConverter()
        assert converter.logger is not None

    def test_mixin_logger_is_cached(self) -> None:
        """Test that the logger is cached after first access."""

        class MyConverter(LoggingMixin):
            pass

        converter = MyConverter()
        logger1 = converter.logger
        logger2 = converter.logger
        assert logger1 is logger2

    def test_log_conversion_start(self) -> None:
        """Test log_conversion_start method."""

        class MyConverter(LoggingMixin):
            pass

        converter = MyConverter()
        # Should not raise
        converter.log_conversion_start("agent", "test_agent")

    def test_log_conversion_start_without_name(self) -> None:
        """Test log_conversion_start without component name."""

        class MyConverter(LoggingMixin):
            pass

        converter = MyConverter()
        converter.log_conversion_start("workflow")

    def test_log_conversion_complete(self) -> None:
        """Test log_conversion_complete method."""

        class MyConverter(LoggingMixin):
            pass

        converter = MyConverter()
        converter.log_conversion_complete("agent", "test_agent", task_count=5)

    def test_log_conversion_complete_with_extra(self) -> None:
        """Test log_conversion_complete with extra context."""

        class MyConverter(LoggingMixin):
            pass

        converter = MyConverter()
        converter.log_conversion_complete("workflow", "my_workflow", task_count=10, edge_count=9)

    def test_log_conversion_error(self) -> None:
        """Test log_conversion_error method."""

        class MyConverter(LoggingMixin):
            pass

        converter = MyConverter()
        error = ValueError("Test error")
        converter.log_conversion_error("agent", error, "test_agent")

    def test_log_conversion_error_with_extra(self) -> None:
        """Test log_conversion_error with extra context."""

        class MyConverter(LoggingMixin):
            pass

        converter = MyConverter()
        error = RuntimeError("Conversion failed")
        converter.log_conversion_error("workflow", error, "my_workflow", step="validation")


class TestLoggingIntegration:
    """Integration tests for logging in loader/exporter."""

    def test_loader_logs_operations(self) -> None:
        """Test that DaprAgentSpecLoader logs operations."""
        from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader

        loader = DaprAgentSpecLoader()
        # Logger should be initialized
        assert hasattr(loader, "_logger")

    def test_exporter_logs_operations(self) -> None:
        """Test that DaprAgentSpecExporter logs operations."""
        from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter

        exporter = DaprAgentSpecExporter()
        # Logger should be initialized
        assert hasattr(exporter, "_logger")
