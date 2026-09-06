from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.dialectic import chat


@asynccontextmanager
async def _db_context(*_args, **_kwargs):
    yield object()


@pytest.mark.asyncio
async def test_agentic_chat_with_allowlist_omits_unscoped_peer_cards(monkeypatch):
    peer = SimpleNamespace(name="observer", internal_metadata={})
    get_peer_card = AsyncMock(return_value=["must not be exposed"])
    monkeypatch.setattr(chat, "tracked_db", _db_context)
    monkeypatch.setattr(chat.crud, "get_peer", AsyncMock(return_value=peer))
    monkeypatch.setattr(chat.crud, "get_workspace", AsyncMock(return_value=object()))
    monkeypatch.setattr(chat.crud, "get_peer_card", get_peer_card)
    monkeypatch.setattr(
        chat,
        "get_configuration",
        lambda *_args: SimpleNamespace(peer_card=SimpleNamespace(use=True)),
    )

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def answer(self, *_args, **_kwargs):
            return "ok"

    monkeypatch.setattr(chat, "DialecticAgent", FakeAgent)
    result = await chat.agentic_chat(
        "workspace",
        None,
        "query",
        "observer",
        "observer",
        session_allowlist=[],
    )
    assert result == "ok"
    get_peer_card.assert_not_awaited()
    assert captured["observer_peer_card"] is None
    assert captured["observed_peer_card"] is None


@pytest.mark.asyncio
async def test_streaming_chat_with_allowlist_omits_unscoped_peer_cards(monkeypatch):
    peer = SimpleNamespace(name="observer", internal_metadata={})
    get_peer_card = AsyncMock(return_value=["must not be exposed"])
    monkeypatch.setattr(chat, "tracked_db", _db_context)
    monkeypatch.setattr(chat.crud, "get_peer", AsyncMock(return_value=peer))
    monkeypatch.setattr(chat.crud, "get_workspace", AsyncMock(return_value=object()))
    monkeypatch.setattr(chat.crud, "get_peer_card", get_peer_card)
    monkeypatch.setattr(
        chat,
        "get_configuration",
        lambda *_args: SimpleNamespace(peer_card=SimpleNamespace(use=True)),
    )

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def answer_stream(self, *_args, **_kwargs):
            yield "ok"

    monkeypatch.setattr(chat, "DialecticAgent", FakeAgent)
    chunks = [
        chunk
        async for chunk in chat.agentic_chat_stream(
            "workspace",
            None,
            "query",
            "observer",
            "observer",
            session_allowlist=[],
        )
    ]
    assert chunks == ["ok"]
    get_peer_card.assert_not_awaited()
    assert captured["observer_peer_card"] is None
    assert captured["observed_peer_card"] is None
