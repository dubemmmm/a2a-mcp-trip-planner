"""Tests for `shared.a2a_client.submission`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from a2a.types import AgentCard, AgentSkill, SendMessageRequest, StreamResponse

from shared.a2a_client.submission import (
    MAX_PAYLOAD_BYTES,
    EmptyPayloadError,
    PayloadTooLargeError,
    SubmissionError,
    UnknownSkillError,
    submit_task,
)


def _card(*skill_ids: str) -> AgentCard:
    return AgentCard(
        name="test-agent",
        description="x",
        version="0.1.0",
        skills=[
            AgentSkill(id=sid, name=sid, description=sid, tags=[]) for sid in skill_ids
        ],
    )


class _FakeClient:
    """Records the SendMessageRequest it was given, returns an empty stream."""

    def __init__(self) -> None:
        self.last_request: SendMessageRequest | None = None

    def send_message(
        self, request: SendMessageRequest, *, context: Any = None
    ) -> AsyncIterator[StreamResponse]:
        self.last_request = request

        async def _empty() -> AsyncIterator[StreamResponse]:
            for _ in ():
                yield  # pragma: no cover

        return _empty()


# ---------- validation ----------

class TestValidation:
    async def test_unknown_skill_rejected(self) -> None:
        card = _card("search_flights")
        with pytest.raises(UnknownSkillError) as exc:
            await submit_task(
                _FakeClient(), card,  # type: ignore[arg-type]
                skill_id="search_flght", payload="ATL→LIS",
            )
        # Helpful diagnostics: known skills listed.
        assert "search_flights" in str(exc.value)

    async def test_empty_skill_id_rejected(self) -> None:
        card = _card("search_flights")
        with pytest.raises(ValueError):
            await submit_task(
                _FakeClient(), card,  # type: ignore[arg-type]
                skill_id="", payload="x",
            )

    async def test_empty_payload_rejected(self) -> None:
        card = _card("search_flights")
        with pytest.raises(EmptyPayloadError):
            await submit_task(
                _FakeClient(), card,  # type: ignore[arg-type]
                skill_id="search_flights", payload="",
            )

    async def test_whitespace_payload_rejected(self) -> None:
        card = _card("search_flights")
        with pytest.raises(EmptyPayloadError):
            await submit_task(
                _FakeClient(), card,  # type: ignore[arg-type]
                skill_id="search_flights", payload="   \n\t",
            )

    async def test_oversize_payload_rejected(self) -> None:
        card = _card("search_flights")
        with pytest.raises(PayloadTooLargeError):
            await submit_task(
                _FakeClient(), card,  # type: ignore[arg-type]
                skill_id="search_flights",
                payload="a" * (MAX_PAYLOAD_BYTES + 1),
            )

    async def test_at_cap_accepted(self) -> None:
        card = _card("search_flights")
        client = _FakeClient()
        handle = await submit_task(
            client, card,  # type: ignore[arg-type]
            skill_id="search_flights",
            payload="a" * MAX_PAYLOAD_BYTES,
        )
        assert handle.message_id

    async def test_all_validation_errors_share_base(self) -> None:
        """All three are catchable as a single SubmissionError."""
        card = _card("x")
        # Unknown skill
        for bad in [
            ("search_flghts", "ok", UnknownSkillError),
            ("x", "", EmptyPayloadError),
            ("x", "y" * (MAX_PAYLOAD_BYTES + 1), PayloadTooLargeError),
        ]:
            skill, payload, _expected = bad
            with pytest.raises(SubmissionError):
                await submit_task(
                    _FakeClient(), card,  # type: ignore[arg-type]
                    skill_id=skill, payload=payload,
                )


# ---------- happy path ----------

class TestSubmitSuccess:
    async def test_returns_handle(self) -> None:
        client = _FakeClient()
        card = _card("search_flights")
        handle = await submit_task(
            client, card,  # type: ignore[arg-type]
            skill_id="search_flights",
            payload="find ATL to LIS for next month",
        )
        assert handle.agent_card is card
        assert handle.skill_id == "search_flights"
        assert handle.message_id  # uuid generated
        assert handle.events is not None

    async def test_explicit_message_id_preserved(self) -> None:
        client = _FakeClient()
        card = _card("search_flights")
        handle = await submit_task(
            client, card,  # type: ignore[arg-type]
            skill_id="search_flights",
            payload="x",
            message_id="my-id-123",
        )
        assert handle.message_id == "my-id-123"

    async def test_sends_message_with_required_fields(self) -> None:
        client = _FakeClient()
        card = _card("search_hotels")
        handle = await submit_task(
            client, card,  # type: ignore[arg-type]
            skill_id="search_hotels",
            payload="Lisbon May 10-14",
            message_id="abc",
        )
        # Drain the empty stream so we exercise the client call.
        async for _ in handle.events:
            pass
        assert client.last_request is not None
        msg = client.last_request.message
        assert msg.message_id == "abc"
        # role == ROLE_USER (proto enum = 1)
        from a2a.types import Role
        assert msg.role == Role.ROLE_USER
        assert len(msg.parts) == 1
        assert msg.parts[0].text == "Lisbon May 10-14"

    async def test_skill_id_in_metadata(self) -> None:
        client = _FakeClient()
        card = _card("search_hotels")
        await submit_task(
            client, card,  # type: ignore[arg-type]
            skill_id="search_hotels", payload="x",
        )
        meta = client.last_request.message.metadata  # type: ignore[union-attr]
        # Struct proto stores fields under a `.fields` map.
        assert "skill_id" in meta.fields
        assert meta.fields["skill_id"].string_value == "search_hotels"