"""Extended tests for client/ws_client.py — TableLightClient."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from client.ws_client import (
    PREFIX_AUDIO_IN,
    PREFIX_AUDIO_OUT,
    PREFIX_VIDEO_IN,
    TableLightClient,
)


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

    def test_only_four_callbacks(self):
        """After refactor, only 4 callback slots remain."""
        c = TableLightClient("ws://localhost:8000/ws")
        assert hasattr(c, "_on_audio")
        assert hasattr(c, "_on_tool_result")
        assert hasattr(c, "_on_transcript")
        assert hasattr(c, "_on_interrupted")


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

    async def test_sends_binary_with_audio_prefix(self, client):
        await client.send_audio(b"\x00\x01")
        raw = client.ws.send.call_args[0][0]
        assert isinstance(raw, bytes)
        assert raw[0:1] == PREFIX_AUDIO_IN

    async def test_payload_roundtrip(self, client):
        pcm = bytes(range(256))
        await client.send_audio(pcm)
        raw = client.ws.send.call_args[0][0]
        assert raw[1:] == pcm

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

    async def test_sends_binary_with_video_prefix(self, client):
        await client.send_video(b"\xff\xd8")
        raw = client.ws.send.call_args[0][0]
        assert isinstance(raw, bytes)
        assert raw[0:1] == PREFIX_VIDEO_IN

    async def test_payload_roundtrip(self, client):
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        await client.send_video(jpeg)
        raw = client.ws.send.call_args[0][0]
        assert raw[1:] == jpeg

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
    """Mock a websocket that yields messages then stops.

    Accepts both str (JSON text) and bytes (binary) messages.
    """

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


class TestReceiveLoopToolResult:
    async def test_all_tool_results_go_to_on_tool_result(self):
        """All tool_result messages dispatch via on_tool_result."""
        c = TableLightClient("ws://localhost:8000/ws")
        results = []

        async def on_tool(name, result):
            results.append((name, result))

        c.on_tool_result(on_tool)
        msgs = [
            json.dumps({
                "type": "tool_result",
                "name": "query",
                "result": {"target": "fresh_view", "status": "refreshing"},
            }),
            json.dumps({
                "type": "tool_result",
                "name": "overlay",
                "result": {"action": "create", "content_type": "graph"},
            }),
        ]
        c.ws = _MockAsyncIterator(msgs)
        await c.receive_loop()
        assert len(results) == 2
        assert results[0][0] == "query"
        assert results[1][0] == "overlay"


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
            PREFIX_AUDIO_OUT + b"\x01\x02",  # binary audio
            json.dumps({"type": "transcript_in", "text": "hello"}),
            PREFIX_AUDIO_OUT + b"\x03\x04",  # binary audio
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
            PREFIX_AUDIO_OUT + b"\x01",  # binary audio
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
