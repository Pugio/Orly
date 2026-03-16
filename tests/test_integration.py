"""End-to-end integration tests for the TableLight backend WebSocket flow.

Uses a mock Gemini Live session so no real API calls are made.
Tests the full pipeline: client WS -> FastAPI -> mock session -> back to client.

Uses the binary WebSocket protocol:
  - Client sends binary frames: 0x01+PCM (audio), 0x02+JPEG (video)
  - Server sends binary frames: 0x03+PCM (audio responses)
  - JSON text frames for text, transcripts, tool results, interrupted
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from backend.main import PREFIX_AUDIO_IN, PREFIX_AUDIO_OUT, PREFIX_VIDEO_IN, app


# ---------------------------------------------------------------------------
# Mock Gemini types — lightweight stand-ins for google.genai.types
# ---------------------------------------------------------------------------


@dataclass
class FakeBlob:
    data: bytes = b""
    mime_type: str = ""


@dataclass
class FakeInlineData:
    data: bytes = b""
    mime_type: str = ""


@dataclass
class FakePart:
    text: str | None = None
    inline_data: FakeInlineData | None = None


@dataclass
class FakeContent:
    role: str = "model"
    parts: list[FakePart] = field(default_factory=list)


@dataclass
class FakeFunctionCall:
    name: str = ""
    args: dict = field(default_factory=dict)
    id: str = "call_1"


@dataclass
class FakeToolCall:
    function_calls: list[FakeFunctionCall] = field(default_factory=list)


@dataclass
class FakeInputTranscription:
    text: str = ""


@dataclass
class FakeOutputTranscription:
    text: str = ""


@dataclass
class FakeServerContent:
    interrupted: bool = False
    model_turn: FakeContent | None = None
    input_transcription: FakeInputTranscription | None = None
    output_transcription: FakeOutputTranscription | None = None


@dataclass
class FakeSessionResumptionUpdate:
    new_handle: str | None = None


@dataclass
class FakeGoAway:
    pass


@dataclass
class FakeToolCallCancellation:
    ids: list[str] = field(default_factory=list)


@dataclass
class FakeServerMessage:
    """A single message yielded by session.receive()."""
    server_content: FakeServerContent | None = None
    tool_call: FakeToolCall | None = None
    tool_call_cancellation: FakeToolCallCancellation | None = None
    session_resumption_update: FakeSessionResumptionUpdate | None = None
    go_away: Any = None


# ---------------------------------------------------------------------------
# Mock Gemini Live Session
# ---------------------------------------------------------------------------


class MockLiveSession:
    """Fake Gemini Live session that records calls and yields scripted messages."""

    def __init__(self, messages: list[FakeServerMessage] | None = None):
        self._messages = messages or []
        self.realtime_inputs: list[dict] = []
        self.client_contents: list[dict] = []
        self.tool_responses: list[dict] = []
        self._closed = False
        # Event to signal that all scripted messages have been consumed.
        self._all_sent = asyncio.Event()

    async def send_realtime_input(self, *, audio=None, video=None, text=None):
        self.realtime_inputs.append({"audio": audio, "video": video, "text": text})

    async def send_client_content(self, *, turns=None, turn_complete=False):
        self.client_contents.append(
            {"turns": turns, "turn_complete": turn_complete}
        )

    async def send_tool_response(self, *, function_responses=None):
        self.tool_responses.append({"function_responses": function_responses})

    async def receive(self):
        for msg in self._messages:
            # Small yield to let the event loop run other tasks.
            await asyncio.sleep(0.01)
            yield msg
        self._all_sent.set()
        # Keep alive until the session context manager exits.
        while not self._closed:
            await asyncio.sleep(0.05)

    async def close(self):
        self._closed = True

    async def __aenter__(self):
        self._closed = False  # reset on reconnect (same object reused)
        return self

    async def __aexit__(self, *args):
        self._closed = True


# ---------------------------------------------------------------------------
# Helper: build a fake genai module + client
# ---------------------------------------------------------------------------


def _make_mock_genai(session: MockLiveSession):
    """Create mock genai.Client and genai types that the backend imports lazily."""

    # --- Mock types module ---
    mock_types = MagicMock()

    # Blob: accept data= and mime_type= kwargs, store them.
    mock_types.Blob = FakeBlob
    mock_types.Content = FakeContent
    mock_types.Part = FakePart
    mock_types.FunctionResponse = MagicMock(
        side_effect=lambda name, response, id: {
            "name": name,
            "response": response,
            "id": id,
        }
    )
    # Config types — just accept any kwargs.
    mock_types.LiveConnectConfig = MagicMock(return_value=MagicMock())
    mock_types.Modality.AUDIO = "AUDIO"
    mock_types.SpeechConfig = MagicMock(return_value=MagicMock())
    mock_types.VoiceConfig = MagicMock(return_value=MagicMock())
    mock_types.PrebuiltVoiceConfig = MagicMock(return_value=MagicMock())
    mock_types.RealtimeInputConfig = MagicMock(return_value=MagicMock())
    mock_types.AutomaticActivityDetection = MagicMock(return_value=MagicMock())
    mock_types.StartSensitivity.START_SENSITIVITY_HIGH = "HIGH"
    mock_types.StartSensitivity.START_SENSITIVITY_LOW = "LOW"
    mock_types.EndSensitivity.END_SENSITIVITY_HIGH = "HIGH"
    mock_types.ProactivityConfig = MagicMock(return_value=MagicMock())
    mock_types.ContextWindowCompressionConfig = MagicMock(return_value=MagicMock())
    mock_types.SlidingWindow = MagicMock(return_value=MagicMock())
    mock_types.SessionResumptionConfig = MagicMock(return_value=MagicMock())

    # --- Mock client ---
    mock_client_instance = MagicMock()
    # client.aio.live.connect(...) returns an async context manager -> session
    mock_client_instance.aio.live.connect = MagicMock(return_value=session)

    mock_client_class = MagicMock(return_value=mock_client_instance)

    # --- Mock genai module ---
    mock_genai = MagicMock()
    mock_genai.Client = mock_client_class
    # Ensure `from google.genai import types` resolves to our mock_types.
    mock_genai.types = mock_types

    return mock_genai, mock_types, mock_client_instance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_ws():
    """Factory fixture: returns a function that sets up patched TestClient WS.

    Usage:
        session, ws = make_ws(messages=[...])
    """

    def _factory(
        messages: list[FakeServerMessage] | None = None,
    ):
        session = MockLiveSession(messages)
        mock_genai, mock_types, _ = _make_mock_genai(session)

        # Clear the Gemini client cache so our mock takes effect.
        from backend.main import _gemini_client_cache
        _gemini_client_cache.clear()

        # Patch the lazy imports inside session_endpoint.
        patcher_genai = patch.dict(
            "sys.modules",
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        )
        patcher_genai.start()

        client = TestClient(app)
        ws = client.websocket_connect("/ws/session")
        ws.__enter__()
        # Send init message (always JSON).
        ws.send_json({"text_only": False})

        class _Handle:
            def close(self):
                try:
                    ws.send_json({"type": "close"})
                except Exception:
                    pass
                try:
                    ws.__exit__(None, None, None)
                except Exception:
                    pass
                patcher_genai.stop()
                _gemini_client_cache.clear()

        return session, ws, _Handle()

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAudioForwarding:
    def test_audio_forwarded_to_gemini(self, make_ws):
        """Send audio via binary WS frame, verify mock session received it."""
        session, ws, handle = make_ws()
        try:
            pcm = b"\x00\x01" * 800  # 1600 bytes of fake PCM
            ws.send_bytes(PREFIX_AUDIO_IN + pcm)
            # Give time for the backend coroutine to process.
            import time
            time.sleep(0.5)

            assert len(session.realtime_inputs) >= 1
            sent = session.realtime_inputs[0]
            assert sent["audio"] is not None
            assert sent["audio"].data == pcm
            assert "audio" in sent["audio"].mime_type
        finally:
            handle.close()


class TestVideoForwarding:
    def test_video_forwarded_to_gemini(self, make_ws):
        """Send video via binary WS frame, verify mock session received it."""
        session, ws, handle = make_ws()
        try:
            jpeg = b"\xff\xd8fake-jpeg"
            ws.send_bytes(PREFIX_VIDEO_IN + jpeg)
            import time
            time.sleep(0.5)

            assert len(session.realtime_inputs) >= 1
            sent = session.realtime_inputs[0]
            assert sent["video"] is not None
            assert sent["video"].data == jpeg
            assert "image/jpeg" in sent["video"].mime_type
        finally:
            handle.close()


class TestTextForwarding:
    def test_text_forwarded_to_gemini(self, make_ws):
        """Send text via JSON, verify send_realtime_input(text=) was called."""
        session, ws, handle = make_ws()
        try:
            ws.send_json({"type": "text", "text": "What is 2+2?"})
            import time
            time.sleep(0.5)

            # Text now goes via send_realtime_input (not send_client_content)
            # to avoid the interleaving warning.
            text_inputs = [
                r for r in session.realtime_inputs if r.get("text")
            ]
            assert len(text_inputs) >= 1
            assert text_inputs[0]["text"] == "What is 2+2?"
        finally:
            handle.close()


class TestAudioResponse:
    def test_audio_response_forwarded_to_client(self, make_ws):
        """Mock yields audio, client receives it as binary WS frame."""
        audio_bytes = b"\x00\x01" * 100
        messages = [
            FakeServerMessage(
                server_content=FakeServerContent(
                    model_turn=FakeContent(
                        role="model",
                        parts=[
                            FakePart(
                                inline_data=FakeInlineData(
                                    data=audio_bytes,
                                    mime_type="audio/pcm;rate=24000",
                                )
                            )
                        ],
                    )
                )
            )
        ]
        session, ws, handle = make_ws(messages)
        try:
            # The server should push audio to us as binary frame.
            import time
            time.sleep(0.5)
            resp = ws.receive_bytes()
            assert resp[0:1] == PREFIX_AUDIO_OUT
            assert resp[1:] == audio_bytes
        finally:
            handle.close()


class TestTranscript:
    def test_transcript_forwarded_to_client(self, make_ws):
        """Mock yields output transcription, client receives transcript_out."""
        messages = [
            FakeServerMessage(
                server_content=FakeServerContent(
                    output_transcription=FakeOutputTranscription(
                        text="The answer is four."
                    )
                )
            )
        ]
        session, ws, handle = make_ws(messages)
        try:
            import time
            time.sleep(0.5)
            resp = ws.receive_json(mode="text")
            assert resp["type"] == "transcript_out"
            assert resp["text"] == "The answer is four."
        finally:
            handle.close()


class TestToolCallFlow:
    def test_tool_call_executed_and_responded(self, make_ws):
        """Mock yields a tool_call for overlay (create action).

        Verify: tool executed, result sent to client, FunctionResponse sent
        back to session.
        """
        messages = [
            FakeServerMessage(
                tool_call=FakeToolCall(
                    function_calls=[
                        FakeFunctionCall(
                            name="overlay",
                            args={
                                "action": "create",
                                "content_type": "annotation",
                                "placement": [100, 100, 500, 600],
                                "title": "Test label",
                                "data": {"text": "Hello!"},
                            },
                            id="call_42",
                        )
                    ]
                )
            )
        ]
        session, ws, handle = make_ws(messages)
        try:
            import time
            time.sleep(0.5)

            # Client should receive the tool_result message (JSON).
            resp = ws.receive_json(mode="text")
            assert resp["type"] == "tool_result"
            # Translated back to legacy name for client.
            assert resp["name"] == "project_overlay"
            assert resp["result"]["status"] == "displayed"
            assert resp["result"]["content_type"] == "annotation"
            # Verify args are merged into the result for client rendering.
            assert resp["result"]["data"] == {"text": "Hello!"}
            assert resp["result"]["title"] == "Test label"

            # Session should have received the FunctionResponse.
            assert len(session.tool_responses) >= 1
            fn_responses = session.tool_responses[0]["function_responses"]
            assert len(fn_responses) == 1
            assert fn_responses[0]["name"] == "overlay"
            assert fn_responses[0]["id"] == "call_42"
            assert fn_responses[0]["response"]["status"] == "displayed"
        finally:
            handle.close()


class TestInterruption:
    def test_interruption_forwarded(self, make_ws):
        """Mock yields interrupted, client receives interrupted message."""
        messages = [
            FakeServerMessage(
                server_content=FakeServerContent(interrupted=True)
            )
        ]
        session, ws, handle = make_ws(messages)
        try:
            import time
            time.sleep(0.5)
            resp = ws.receive_json(mode="text")
            assert resp["type"] == "interrupted"
        finally:
            handle.close()


class TestSessionReconnect:
    def test_session_reconnect_on_crash(self):
        """Mock raises on first receive(), succeeds on second — verify reconnect.

        Reads messages eagerly (no time.sleep) and sends close from a
        background thread as a safety timeout.
        """
        import threading

        call_count = {"n": 0}

        class CrashThenOkSession(MockLiveSession):
            """First receive raises, second yields a transcript then waits."""

            def __init__(self):
                super().__init__([])

            async def receive(self):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise RuntimeError("Simulated Gemini crash")
                yield FakeServerMessage(
                    server_content=FakeServerContent(
                        output_transcription=FakeOutputTranscription(
                            text="Recovered!"
                        )
                    )
                )
                while not self._closed:
                    await asyncio.sleep(0.05)

        session = CrashThenOkSession()
        mock_genai, mock_types, _ = _make_mock_genai(session)

        from backend.main import _gemini_client_cache
        _gemini_client_cache.clear()

        patcher = patch.dict(
            "sys.modules",
            {
                "google": MagicMock(genai=mock_genai),
                "google.genai": mock_genai,
                "google.genai.types": mock_types,
            },
        )
        patcher.start()
        try:
            client = TestClient(app)
            ws = client.websocket_connect("/ws/session")
            ws.__enter__()
            ws.send_json({"text_only": False})

            # After the crash the backend reconnects silently and the
            # second session yields "Recovered!" as a transcript.
            # Use a safety thread to send close if reading blocks.
            def _safety_close():
                import time
                time.sleep(8)
                try:
                    ws.send_json({"type": "close"})
                except Exception:
                    pass

            safety = threading.Thread(target=_safety_close, daemon=True)
            safety.start()

            msg1 = ws.receive_json(mode="text")
            assert msg1["type"] == "transcript_out"
            assert "Recovered" in msg1.get("text", "")
            assert call_count["n"] >= 2

            # Clean up
            try:
                ws.send_json({"type": "close"})
            except Exception:
                pass
            import time
            time.sleep(0.2)
            try:
                ws.__exit__(None, None, None)
            except Exception:
                pass
        finally:
            patcher.stop()
            _gemini_client_cache.clear()
