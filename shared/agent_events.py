"""Small in-process event envelope for observing agent execution."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    """One user-facing execution event placed on an in-process queue."""

    kind: str
    agent: str
    message: str
    status: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AgentEventQueue = asyncio.Queue[AgentEvent]


def emit_agent_event(
    queue: AgentEventQueue | None,
    *,
    kind: str,
    agent: str,
    message: str,
    status: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Publish without blocking the agent execution path."""
    if queue is None:
        return
    queue.put_nowait(
        AgentEvent(
            kind=kind,
            agent=agent,
            message=message,
            status=status,
            data=data or {},
        )
    )
