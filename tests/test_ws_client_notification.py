"""Tests for client-side WebSocket notification extensions.

Tests send_notification and receive_loop handling for run_program,
stop_program, and get_overlay_state messages.
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
# receive_loop — run_program
# ---------------------------------------------------------------------------


class TestReceiveRunProgram:
    async def test_receive_run_program(self):
        c = TableLightClient("ws://localhost:8000/ws")
        calls = []

        async def on_run(name, code, description):
            calls.append((name, code, description))

        c.on_run_program(on_run)
        msg = json.dumps({
            "type": "run_program",
            "name": "tracker",
            "code": "print('hi')",
            "description": "A tracker",
        })
        c.ws = _MockAsyncIterator([msg])
        await c.receive_loop()
        assert len(calls) == 1
        assert calls[0] == ("tracker", "print('hi')", "A tracker")


# ---------------------------------------------------------------------------
# receive_loop — stop_program
# ---------------------------------------------------------------------------


class TestReceiveStopProgram:
    async def test_receive_stop_program(self):
        c = TableLightClient("ws://localhost:8000/ws")
        calls = []

        async def on_stop(name):
            calls.append(name)

        c.on_stop_program(on_stop)
        msg = json.dumps({"type": "stop_program", "name": "tracker"})
        c.ws = _MockAsyncIterator([msg])
        await c.receive_loop()
        assert calls == ["tracker"]


# ---------------------------------------------------------------------------
# receive_loop — get_overlay_state
# ---------------------------------------------------------------------------


class TestReceiveGetOverlayState:
    async def test_receive_get_overlay_state(self):
        c = TableLightClient("ws://localhost:8000/ws")
        calls = []

        async def on_get():
            calls.append(True)
            return {"overlays": [], "count": 0}

        c.on_get_overlay_state(on_get)
        msg = json.dumps({"type": "get_overlay_state"})
        c.ws = _MockAsyncIterator([msg])
        await c.receive_loop()
        assert calls == [True]


# ---------------------------------------------------------------------------
# No callbacks registered — should not crash
# ---------------------------------------------------------------------------


class TestCallbacksNotRegistered:
    async def test_callbacks_not_registered(self):
        """Receiving new message types without callbacks should not crash."""
        c = TableLightClient("ws://localhost:8000/ws")
        msgs = [
            json.dumps({"type": "run_program", "name": "x", "code": "pass", "description": ""}),
            json.dumps({"type": "stop_program", "name": "x"}),
            json.dumps({"type": "get_overlay_state"}),
        ]
        c.ws = _MockAsyncIterator(msgs)
        await c.receive_loop()  # should not raise
