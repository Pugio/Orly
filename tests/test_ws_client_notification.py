"""Tests for client-side WebSocket notification and unified tool_result dispatch.

Tests send_notification and receive_loop handling for consolidated tool names
(overlay, query, music) via the single on_tool_result callback.
"""

import json
from unittest.mock import AsyncMock

import pytest

from client.ws_client import TableLightClient


# ---------------------------------------------------------------------------
# Helper: mock async iterator for ws messages
# ---------------------------------------------------------------------------


class _MockAsyncIterator:
    """Mock a websocket that yields messages then stops."""

    def __init__(self, messages: list[str | bytes]):
        self._messages = messages
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._index]
        self._index += 1
        return msg


# ---------------------------------------------------------------------------
# send_notification
# ---------------------------------------------------------------------------


class TestSendNotification:
    @pytest.fixture
    def client(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        c.ws.closed = False
        return c

    async def test_send_notification_format(self, client):
        await client.send_notification("img_gen", "Image ready")
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert msg["type"] == "notification"
        assert msg["source"] == "img_gen"
        assert msg["text"] == "Image ready"

    async def test_send_notification_when_disconnected(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = None
        # Should not raise
        await c.send_notification("test", "hello")


# ---------------------------------------------------------------------------
# receive_loop — tool_result dispatch (consolidated names)
# ---------------------------------------------------------------------------


class TestReceiveToolResult:
    async def test_overlay_tool_result(self):
        c = TableLightClient("ws://localhost:8000/ws")
        calls = []

        async def on_tool(name, result):
            calls.append((name, result))

        c.on_tool_result(on_tool)
        msg = json.dumps({
            "type": "tool_result",
            "name": "overlay",
            "result": {"action": "create", "status": "displayed", "content_type": "graph"},
        })
        c.ws = _MockAsyncIterator([msg])
        await c.receive_loop()
        assert len(calls) == 1
        assert calls[0][0] == "overlay"
        assert calls[0][1]["action"] == "create"

    async def test_query_tool_result(self):
        c = TableLightClient("ws://localhost:8000/ws")
        calls = []

        async def on_tool(name, result):
            calls.append((name, result))

        c.on_tool_result(on_tool)
        msg = json.dumps({
            "type": "tool_result",
            "name": "query",
            "result": {"target": "fresh_view", "status": "refreshing"},
        })
        c.ws = _MockAsyncIterator([msg])
        await c.receive_loop()
        assert len(calls) == 1
        assert calls[0][0] == "query"

    async def test_music_tool_result(self):
        c = TableLightClient("ws://localhost:8000/ws")
        calls = []

        async def on_tool(name, result):
            calls.append((name, result))

        c.on_tool_result(on_tool)
        msg = json.dumps({
            "type": "tool_result",
            "name": "music",
            "result": {"action": "play", "status": "starting", "name": "chill"},
        })
        c.ws = _MockAsyncIterator([msg])
        await c.receive_loop()
        assert len(calls) == 1
        assert calls[0][0] == "music"


# ---------------------------------------------------------------------------
# No callbacks registered — should not crash
# ---------------------------------------------------------------------------


class TestCallbacksNotRegistered:
    async def test_callbacks_not_registered(self):
        """Receiving tool_result without callbacks should not crash."""
        c = TableLightClient("ws://localhost:8000/ws")
        msgs = [
            json.dumps({"type": "tool_result", "name": "overlay", "result": {"action": "create"}}),
            json.dumps({"type": "tool_result", "name": "music", "result": {"action": "play"}}),
            json.dumps({"type": "tool_result", "name": "query", "result": {"target": "fresh_view"}}),
        ]
        c.ws = _MockAsyncIterator(msgs)
        await c.receive_loop()  # should not raise
