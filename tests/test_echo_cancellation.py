"""Tests for audio playback, interruption handling, and tool call cancellation.

AudioPlayer.clear() must instantly stop playback on interruption.
audio_send_loop must always forward mic audio (never gate it) so
Gemini's server-side VAD can detect user speech for interruptions.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# 1. AudioPlayer.is_playing property
# ---------------------------------------------------------------------------


class TestAudioPlayerIsPlaying:
    """AudioPlayer exposes is_playing so the send loop can gate mic input."""

    def test_not_playing_initially(self):
        from client.audio import AudioPlayer
        player = AudioPlayer()
        assert player.is_playing is False

    def test_not_playing_when_queue_empty(self):
        from client.audio import AudioPlayer
        player = AudioPlayer()
        # Even after start()-like setup, empty queue means not playing.
        assert player.is_playing is False

    def test_playing_when_queue_has_data(self):
        from client.audio import AudioPlayer
        player = AudioPlayer()
        player.play(b"\x00" * 100)
        assert player.is_playing is True

    def test_not_playing_after_queue_drained(self):
        from client.audio import AudioPlayer
        player = AudioPlayer()
        player.play(b"\x00" * 100)
        assert player.is_playing is True
        # Drain manually (simulating what _playback_loop does).
        player._queue.get_nowait()
        assert player.is_playing is False

    def test_clear_on_interrupt(self):
        """clear() empties the playback queue and resets is_playing."""
        from client.audio import AudioPlayer
        player = AudioPlayer()
        player.play(b"\x00" * 100)
        player.play(b"\x00" * 100)
        assert player.is_playing is True
        player.clear()
        assert player.is_playing is False
        assert player._queue.empty()

    def test_clear_restarts_stream(self):
        """clear() stops and restarts the PyAudio stream to flush OS buffer."""
        from client.audio import AudioPlayer
        from unittest.mock import MagicMock
        player = AudioPlayer()
        player._stream = MagicMock()
        player.play(b"\x00" * 100)
        player.clear()
        player._stream.stop_stream.assert_called_once()
        player._stream.start_stream.assert_called_once()

    def test_clear_sets_interrupted_flag(self):
        """clear() signals the playback thread to skip in-progress writes."""
        from client.audio import AudioPlayer
        player = AudioPlayer()
        # _interrupted should be False after clear finishes
        player.clear()
        assert player._interrupted is False


# ---------------------------------------------------------------------------
# 2. audio_send_loop always forwards mic (no gating)
# ---------------------------------------------------------------------------


class TestAudioSendLoopAlwaysForwards:
    """audio_send_loop must ALWAYS forward mic audio to Gemini.

    When no processor is provided, all audio is forwarded raw.
    When a processor is provided, audio is preprocessed but still forwarded
    (the processor may replace quiet chunks with silence, but chunks are
    never dropped entirely).
    """

    @pytest.mark.asyncio
    async def test_sends_all_audio_chunks(self):
        """All mic audio is forwarded — never gated (no processor)."""
        from client.audio import AudioCapture

        capture = AudioCapture.__new__(AudioCapture)
        capture._audio_queue = queue.Queue()
        capture.chunk_size = 320

        capture._audio_queue.put(b"\x01" * 640)
        capture._audio_queue.put(b"\x02" * 640)

        sent_chunks: list[bytes] = []

        class FakeClient:
            async def send_audio(self, data):
                sent_chunks.append(data)

        from client.main import audio_send_loop

        task = asyncio.create_task(audio_send_loop(capture, FakeClient()))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(sent_chunks) == 2

    @pytest.mark.asyncio
    async def test_sends_all_chunks_with_processor(self):
        """With a processor, chunks are still forwarded (possibly as silence)."""
        from client.audio import AudioCapture
        from client.audio_processor import AudioProcessor

        capture = AudioCapture.__new__(AudioCapture)
        capture._audio_queue = queue.Queue()
        capture.chunk_size = 320

        # Two chunks — one quiet (will be gated to silence), one loud
        import struct
        quiet = b"\x00" * 640  # silence
        loud = struct.pack("<320h", *([5000] * 320))  # speech-level

        capture._audio_queue.put(quiet)
        capture._audio_queue.put(loud)

        sent_chunks: list[bytes] = []

        class FakeClient:
            async def send_audio(self, data):
                sent_chunks.append(data)

        processor = AudioProcessor(noise_gate_rms=150)

        from client.main import audio_send_loop

        task = asyncio.create_task(
            audio_send_loop(capture, FakeClient(), processor=processor)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Both chunks forwarded (silence chunk becomes zero bytes, loud passes)
        assert len(sent_chunks) == 2
        # First chunk should be silence (gated)
        assert all(b == 0 for b in sent_chunks[0])
        # Second chunk should have audio content
        assert not all(b == 0 for b in sent_chunks[1])


# ---------------------------------------------------------------------------
# 3. Integration: tool_call_cancellation reopens the gate
# ---------------------------------------------------------------------------


class TestToolCallCancellationIntegration:
    """When Gemini cancels a tool call (user interrupted during execution),
    the backend should reopen the tool_call_pending gate."""

    def test_cancellation_reopens_gate(self, make_ws):
        """Send a tool call followed by cancellation.

        The gate should reopen so audio/video can flow again.
        """
        from tests.test_integration import (
            FakeFunctionCall,
            FakeServerContent,
            FakeServerMessage,
            FakeToolCall,
            FakeToolCallCancellation,
        )

        messages = [
            # Tool call arrives
            FakeServerMessage(
                tool_call=FakeToolCall(
                    function_calls=[
                        FakeFunctionCall(
                            name="get_overlay_state",
                            args={},
                            id="call_99",
                        )
                    ]
                )
            ),
            # Then cancellation (user interrupted)
            FakeServerMessage(
                tool_call_cancellation=FakeToolCallCancellation(
                    ids=["call_99"]
                )
            ),
        ]
        from backend.main import PREFIX_AUDIO_IN
        session, ws, handle = make_ws(messages)
        try:
            time.sleep(1.0)
            # After cancellation, audio should still flow (gate reopened).
            pcm = b"\x00\x01" * 400
            ws.send_bytes(PREFIX_AUDIO_IN + pcm)
            time.sleep(0.5)

            # If the gate was stuck closed, the sender would block and
            # never forward this audio. Check it arrived.
            assert len(session.realtime_inputs) >= 1
        finally:
            handle.close()


@pytest.fixture
def make_ws():
    """Re-export the make_ws fixture from test_integration."""
    from tests.test_integration import _make_mock_genai, MockLiveSession
    from tests.test_integration import (
        FakeServerMessage,
    )
    from unittest.mock import MagicMock, patch
    from starlette.testclient import TestClient
    from backend.main import app, _gemini_client_cache

    def _factory(messages=None):
        session = MockLiveSession(messages)
        mock_genai, mock_types, _ = _make_mock_genai(session)
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

        client = TestClient(app)
        ws = client.websocket_connect("/ws/session")
        ws.__enter__()
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
                patcher.stop()
                _gemini_client_cache.clear()

        return session, ws, _Handle()

    return _factory
