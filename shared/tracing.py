"""OpenAI Agents SDK built-in tracing.

The SDK auto-traces every `Runner.run()` call when an OpenAI key is set; traces
appear in the OpenAI Traces dashboard. This module exists to:

  * disable tracing automatically in the test environment (so tests don't ship
    spans to OpenAI), and
  * give each agent a single place to opt in or out at startup.

There is nothing to wire up beyond calling `setup_tracing()` once per process.
"""

from __future__ import annotations

from agents import set_tracing_disabled

from .config import Environment, get_settings


def setup_tracing() -> bool:
    """Enable or disable Agents SDK tracing based on the current environment.

    Returns:
        True if tracing is enabled, False if it was disabled.
    """
    settings = get_settings()
    if settings.environment is Environment.TEST:
        set_tracing_disabled(True)
        return False
    set_tracing_disabled(False)
    return True