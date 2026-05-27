"""Structured logging setup.

Every agent calls `setup_logger(agent_name)` once at startup. The returned
logger has the agent name bound to it, so every record in production traces
back to which agent emitted it.

Output format is controlled by `LOG_FORMAT` (json|pretty). JSON for prod
(grep-able, parseable by aggregators), pretty for local dev.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from .config import LogFormat, get_settings


def setup_logger(agent_name: str) -> structlog.stdlib.BoundLogger:
    """Configure structlog and return a logger bound to the given agent name.

    Calling this multiple times in the same process is safe: structlog's
    `configure()` replaces the previous configuration. In a multi-agent monorepo
    each agent process calls this once at startup.
    """
    settings = get_settings()
    level = logging.getLevelName(settings.log_level.upper())
    if not isinstance(level, int):
        raise ValueError(f"Invalid LOG_LEVEL: {settings.log_level!r}")

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    renderer: Any
    if settings.log_format is LogFormat.JSON:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )

    logger: structlog.stdlib.BoundLogger = structlog.get_logger().bind(agent=agent_name)
    return logger