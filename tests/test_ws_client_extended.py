"""Extended tests for client/ws_client.py — TableLightClient."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from client.ws_client import TableLightClient


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestTableLightClientInit:
    def test_stores_backend_url(self):
        c = TableLightClient("ws://example.com/ws")
        assert c.backend_url == "ws://example.com/ws"

    def test_ws_starts_none(self):
        c = TableLightClient("ws://localhost:8000/ws")
        assert c.ws is None

    def test_callbacks_start_none(self):
        c = TableLightClient("ws://localhost:8000/ws")
        assert c._on_audio is None
        assert c._on_tool_result is None
        assert c._on_transcript is None
        assert c._on_interrupted is None
        assert c._on_refresh_view is None


# ---------------------------------------------------------------------------
# connected property
# ---------------------------------------------------------------------------


class TestConnectedProperty:
    def test_false_when_ws_is_none(self):
        c = TableLightClient("ws://localhost:8000/ws")
        assert c.connected is False

    def test_true_when_ws_exists_and_not_closed(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = MagicMock()
        c.ws.closed = False
        assert c.connected is True

    def test_false_when_ws_closed(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = MagicMock()
        c.ws.closed = True
        assert c.connected is False

    def test_true_when_ws_has_no_closed_attr(self):
        """Some mock ws objects may not have 'closed'; should assume open."""
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = object()  # no 'closed' attribute
        assert c.connected is True


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


class TestCallbackRegistration:
    def test_on_audio_registers(self):
        c = TableLightClient("ws://localhost:8000/ws")
        cb = AsyncMock()
        c.on_audio(cb)
        assert c._on_audio is cb

    def test_on_tool_result_registers(self):
        c = TableLightClient("ws://localhost:8000/ws")
        cb = AsyncMock()
        c.on_tool_result(cb)
        assert c._on_tool_result is cb

    def test_on_transcript_registers(self):
        c = TableLightClient("ws://localhost:8000/ws")
        cb = AsyncMock()
        c.on_transcript(cb)
        assert c._on_transcript is cb

    def test_on_interrupted_registers(self):
        c = TableLightClient("ws://localhost:8000/ws")
        cb = AsyncMock()
        c.on_interrupted(cb)
        assert c._on_interrupted is cb

    def test_on_refresh_view_registers(self):
        c = TableLightClient("ws://localhost:8000/ws")
        cb = AsyncMock()
        c.on_refresh_view(cb)
        assert c._on_refresh_view is cb


# ---------------------------------------------------------------------------
# send methods — require connected ws
# ---------------------------------------------------------------------------


class TestSendAudio:
    @pytest.fixture
    def client(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        c.ws.closed = False
        return c

    async def test_sends_json_with_type_audio(self, client):
        await client.send_audio(b"\x00\x01")
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert msg["type"] == "audio"

    async def test_base64_roundtrip(self, client):
        pcm = bytes(range(256))
        await client.send_audio(pcm)
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert base64.b64decode(msg["data"]) == pcm

    async def test_does_not_send_when_disconnected(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = None
        # Should not raise
        await c.send_audio(b"\x00\x01")


class TestSendVideo:
    @pytest.fixture
    def client(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        c.ws.closed = False
        return c

    async def test_sends_json_with_type_video(self, client):
        await client.send_video(b"\xff\xd8")
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert msg["type"] == "video"

    async def test_base64_roundtrip(self, client):
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        await client.send_video(jpeg)
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert base64.b64decode(msg["data"]) == jpeg

    async def test_does_not_send_when_disconnected(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        c.ws.closed = True
        await c.send_video(b"\xff\xd8")
        c.ws.send.assert_not_called()


class TestSendText:
    @pytest.fixture
    def client(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        c.ws.closed = False
        return c

    async def test_sends_json_with_type_text(self, client):
        await client.send_text("hi there")
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert msg["type"] == "text"
        assert msg["text"] == "hi there"

    async def test_empty_text(self, client):
        await client.send_text("")
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert msg["text"] == ""

    async def test_does_not_send_when_disconnected(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = None
        await c.send_text("hi")  # should not raise


# ---------------------------------------------------------------------------
# receive_loop dispatch
# ---------------------------------------------------------------------------


class _MockAsyncIterator:
    """Mock a websocket that yields messages then stops."""

    def __init__(self, messages: list[str]):
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


class TestReceiveLoopRefreshView:
    async def test_refresh_view_dispatched(self):
        c = TableLightClient("ws://localhost:8000/ws")
        called = []

        async def on_refresh():
            called.append(True)

        c.on_refresh_view(on_refresh)
        msg = json.dumps({
            "type": "tool_result",
            "name": "refresh_view",
            "result": {"status": "refreshing"},
        })
        c.ws = _MockAsyncIterator([msg])
        await c.receive_loop()
        assert called == [True]

    async def test_tool_result_not_refresh_goes_to_on_tool(self):
        c = TableLightClient("ws://localhost:8000/ws")
        results = []

        async def on_tool(name, result):
            results.append((name, result))

        c.on_tool_result(on_tool)
        msg = json.dumps({
            "type": "tool_result",
            "name": "project_overlay",
            "result": {"content_type": "graph"},
        })
        c.ws = _MockAsyncIterator([msg])
        await c.receive_loop()
        assert len(results) == 1
        assert results[0][0] == "project_overlay"


class TestReceiveLoopMultipleMessages:
    async def test_dispatches_all_messages(self):
        c = TableLightClient("ws://localhost:8000/ws")
        audio_data = []
        transcripts = []

        async def on_audio(data):
            audio_data.append(data)

        async def on_transcript(direction, text):
            transcripts.append((direction, text))

        c.on_audio(on_audio)
        c.on_transcript(on_transcript)

        msgs = [
            json.dumps({"type": "audio", "data": base64.b64encode(b"\x01\x02").decode()}),
            json.dumps({"type": "transcript_in", "text": "hello"}),
            json.dumps({"type": "audio", "data": base64.b64encode(b"\x03\x04").decode()}),
            json.dumps({"type": "transcript_out", "text": "hi back"}),
        ]
        c.ws = _MockAsyncIterator(msgs)
        await c.receive_loop()

        assert len(audio_data) == 2
        assert audio_data[0] == b"\x01\x02"
        assert audio_data[1] == b"\x03\x04"
        assert transcripts == [("in", "hello"), ("out", "hi back")]


class TestReceiveLoopNoCallback:
    async def test_no_callback_does_not_crash(self):
        """Messages should be silently dropped if no callback is registered."""
        c = TableLightClient("ws://localhost:8000/ws")
        msgs = [
            json.dumps({"type": "audio", "data": base64.b64encode(b"\x01").decode()}),
            json.dumps({"type": "transcript_in", "text": "hello"}),
            json.dumps({"type": "interrupted"}),
        ]
        c.ws = _MockAsyncIterator(msgs)
        await c.receive_loop()  # should not raise


class TestClose:
    async def test_close_calls_ws_close(self):
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        await c.close()
        c.ws.close.assert_called_once()

    async def test_close_when_no_ws(self):
        c = TableLightClient("ws://localhost:8000/ws")
        await c.close()  # should not raise
