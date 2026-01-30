"""Structured logging for dapr-agents-oas-adapter.

Provides a configured structlog logger with:
- Console output with rich formatting (dev mode)
- JSON output (production mode)
- Context binding for component/operation tracking
- Performance timing helpers
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog
from structlog.typing import FilteringBoundLogger

# Module-level logger instance
_logger: FilteringBoundLogger | None = None
_configured: bool = False


def configure_logging(
    *,
    level: int = logging.INFO,
    json_format: bool = False,
    add_timestamp: bool = True,
) -> None:
    """Configure structlog for the library.

    Args:
        level: Log level (default: INFO)
        json_format: Use JSON output format (default: False for console)
        add_timestamp: Add timestamps to log entries (default: True)
    """
    global _configured

    processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if add_timestamp:
        processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=sys.stdout.isatty(),
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None, **initial_context: Any) -> FilteringBoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically module name)
        **initial_context: Initial context to bind to the logger

    Returns:
        Configured structlog logger
    """
    global _logger, _configured

    if not _configured:
        configure_logging()

    logger = structlog.get_logger(name)

    if initial_context:
        logger = logger.bind(**initial_context)

    return logger


def bind_context(**context: Any) -> None:
    """Bind context variables for all loggers in current context.

    Args:
        **context: Context variables to bind
    """
    structlog.contextvars.bind_contextvars(**context)


def unbind_context(*keys: str) -> None:
    """Unbind context variables.

    Args:
        *keys: Keys to unbind
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


@contextmanager
def log_context(**context: Any) -> Generator[None, None, None]:
    """Context manager for temporary context binding.

    Args:
        **context: Context variables to bind temporarily

    Yields:
        None
    """
    bind_context(**context)
    try:
        yield
    finally:
        unbind_context(*context.keys())


@contextmanager
def log_operation(
    operation: str,
    logger: FilteringBoundLogger | None = None,
    **extra_context: Any,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for logging operation start/end with timing.

    Args:
        operation: Name of the operation
        logger: Logger to use (creates default if None)
        **extra_context: Additional context to log

    Yields:
        dict with 'start_time' key for custom timing

    Example:
        with log_operation("convert_agent", component="my_agent") as ctx:
            result = do_conversion()
            ctx["result_count"] = len(result)
    """
    log = logger if logger is not None else get_logger()

    ctx: dict[str, Any] = {"start_time": time.perf_counter()}

    log.info(f"{operation}_started", operation=operation, **extra_context)

    try:
        yield ctx
        duration_ms = (time.perf_counter() - ctx["start_time"]) * 1000
        # Include any additional context added during the operation
        result_context = {k: v for k, v in ctx.items() if k != "start_time"}
        log.info(
            f"{operation}_completed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **extra_context,
            **result_context,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - ctx["start_time"]) * 1000
        log.error(
            f"{operation}_failed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            error=str(e),
            error_type=type(e).__name__,
            **extra_context,
        )
        raise


class LoggingMixin:
    """Mixin class to add logging capability to converters.

    Add this mixin to converter classes to get a pre-configured logger.

    Example:
        class MyConverter(LoggingMixin):
            def convert(self, data):
                self.logger.info("converting", data_type=type(data).__name__)
    """

    _logger: FilteringBoundLogger | None = None

    @property
    def logger(self) -> FilteringBoundLogger:
        """Get the logger for this instance."""
        if self._logger is None:
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

    def log_conversion_start(self, component_type: str, component_name: str | None = None) -> None:
        """Log the start of a conversion operation.

        Args:
            component_type: Type of component being converted
            component_name: Name of the component (if available)
        """
        log = self.logger
        log.info(
            "conversion_started",
            component_type=component_type,
            component_name=component_name,
        )

    def log_conversion_complete(
        self, component_type: str, component_name: str | None = None, **extra: Any
    ) -> None:
        """Log the completion of a conversion operation.

        Args:
            component_type: Type of component converted
            component_name: Name of the component (if available)
            **extra: Additional context to log
        """
        log = self.logger
        log.info(
            "conversion_completed",
            component_type=component_type,
            component_name=component_name,
            **extra,
        )

    def log_conversion_error(
        self,
        component_type: str,
        error: Exception,
        component_name: str | None = None,
        **extra: Any,
    ) -> None:
        """Log a conversion error.

        Args:
            component_type: Type of component that failed
            error: The exception that occurred
            component_name: Name of the component (if available)
            **extra: Additional context to log
        """
        log = self.logger
        log.error(
            "conversion_failed",
            component_type=component_type,
            component_name=component_name,
            error=str(error),
            error_type=type(error).__name__,
            **extra,
        )
