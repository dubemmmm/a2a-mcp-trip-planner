"""Task submission — send a typed Message to a discovered agent.

This module wraps a2a-sdk's `Client.send_message` with three sanity checks
applied *before* the request leaves the process:

  1. **Skill exists.** The requested `skill_id` must be declared on the
     discovered Agent Card. Catches typos like ``search_flght`` without
     burning a network round-trip.
  2. **Payload non-empty.** Blank prompts almost always indicate a caller bug
     (variable interpolation failed, an upstream parser dropped text). We
     refuse to send them.
  3. **Size cap.** Refuse to send payloads above `MAX_PAYLOAD_BYTES`. Protects
     downstream agents from accidental megabyte uploads from caller bugs.

After validation, the actual `send_message` call returns the async iterator
of `StreamResponse` events that `awaiting.await_task` consumes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import structlog
from a2a.client import Client
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
)

_LOGGER = structlog.get_logger(__name__)

# Refuse to send anything above 100 KB. Layer 4-6 traffic is short text;
# this guards against an upstream bug attaching a file or a huge dump.
MAX_PAYLOAD_BYTES = 100 * 1024


class SubmissionError(RuntimeError):
    """Base class for submission failures detected before the network call."""


class UnknownSkillError(SubmissionError):
    """`skill_id` is not declared on the discovered Agent Card."""


class EmptyPayloadError(SubmissionError):
    """The payload text is empty or whitespace-only."""


class PayloadTooLargeError(SubmissionError):
    """The payload exceeds `MAX_PAYLOAD_BYTES`."""


@dataclass(frozen=True)
class TaskHandle:
    """A reference to an in-flight A2A task plus its open event stream.

    The orchestrator passes this to `await_task` (Layer 5.3) to consume
    events until the task reaches a terminal state.
    """

    agent_card: AgentCard
    skill_id: str
    message_id: str
    events: AsyncIterator[StreamResponse]


def _validate_payload(skill_id: str, payload: str, agent_card: AgentCard) -> None:
    """Apply the three pre-flight checks. Raises on failure."""
    known_skills = {s.id for s in agent_card.skills}
    if skill_id not in known_skills:
        raise UnknownSkillError(
            f"agent {agent_card.name!r} has no skill {skill_id!r}; "
            f"available skills: {sorted(known_skills)}"
        )

    if not payload.strip():
        raise EmptyPayloadError(
            f"refusing to send empty payload to {agent_card.name!r}"
        )

    payload_bytes = len(payload.encode("utf-8"))
    if payload_bytes > MAX_PAYLOAD_BYTES:
        raise PayloadTooLargeError(
            f"payload is {payload_bytes} bytes; cap is {MAX_PAYLOAD_BYTES}"
        )


async def submit_task(
    client: Client,
    agent_card: AgentCard,
    *,
    skill_id: str,
    payload: str,
    message_id: str | None = None,
) -> TaskHandle:
    """Send a text task to the agent described by `agent_card`.

    Args:
        client: An a2a-sdk `Client` built for this agent (typically from
            `ClientFactory(config).create(card)`).
        agent_card: The discovered card — used for pre-flight skill validation.
        skill_id: The skill to invoke. Must appear in `agent_card.skills`.
        payload: The user-facing text to send. Skill name + payload together
            tell the remote agent what to do.
        message_id: Optional caller-supplied ID. Generated when omitted.

    Returns:
        A `TaskHandle` carrying the open event stream. Consume it with
        `await_task` (Layer 5.3).

    Raises:
        UnknownSkillError, EmptyPayloadError, PayloadTooLargeError: Validation.
        ValueError: For obviously bad arguments.
    """
    if not skill_id.strip():
        raise ValueError("skill_id must be non-empty")
    _validate_payload(skill_id, payload, agent_card)

    msg_id = message_id or str(uuid.uuid4())
    request = SendMessageRequest(
        message=Message(
            message_id=msg_id,
            role=Role.ROLE_USER,
            parts=[Part(text=payload)],
            metadata=_skill_metadata(skill_id),
        )
    )
    _LOGGER.info(
        "a2a_task_submit",
        agent=agent_card.name,
        skill=skill_id,
        message_id=msg_id,
        payload_bytes=len(payload.encode("utf-8")),
    )
    events = client.send_message(request)
    return TaskHandle(
        agent_card=agent_card,
        skill_id=skill_id,
        message_id=msg_id,
        events=events,
    )


def _skill_metadata(skill_id: str) -> object:
    """Attach the skill id as message metadata.

    The A2A spec leaves skill routing to the agent's discretion (the remote
    agent reads the message and decides which skill applies). Including the
    intended skill in metadata makes that easier for agents that route by
    explicit hint, and provides a paper trail in traces either way.
    """
    from google.protobuf import struct_pb2

    s = struct_pb2.Struct()
    s.update({"skill_id": skill_id})
    return s