"""Live API smoke tests — connects to the REAL Gemini API.

These tests validate that the backend config actually works against the
Gemini server, catching issues that mock-only tests miss (unsupported
config params, 1008 errors, session drops, etc.).

Requires a Gemini API key (GOOGLE_API_KEY or via `llm keys get gemini`).
Skipped automatically in CI or when no API key is available.

Run explicitly:
    uv run python -m pytest tests/test_live_api.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

# ---------------------------------------------------------------------------
# Skip if no API key available
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return subprocess.check_output(
            ["llm", "keys", "get", "gemini"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


_API_KEY = _get_api_key()
_skip = pytest.mark.skipif(
    not _API_KEY or os.environ.get("CI") == "true",
    reason="No Gemini API key or running in CI",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    from google import genai
    return genai.Client(api_key=_API_KEY)


def _make_config(*, with_tools: bool = False):
    """Build the same LiveConnectConfig the backend uses for Google AI."""
    from google.genai import types

    tools_arg = []
    if with_tools:
        tools_arg = [{"function_declarations": [{
            "name": "overlay",
            "description": "Display an overlay on the table.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "content_type": {"type": "STRING"},
                    "title": {"type": "STRING"},
                },
                "required": ["content_type", "title"],
            },
        }]}]

    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction="You are a helpful assistant. Be very brief.",
        tools=tools_arg or None,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore"
                )
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=100,
                silence_duration_ms=500,
            )
        ),
        input_audio_transcription={},
        output_audio_transcription={},
    )


MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip
class TestLiveConnection:
    """Basic connection and single-turn response."""

    @pytest.mark.asyncio
    async def test_connect_and_get_response(self):
        """Connect, send text, get a transcription back."""
        from google.genai import types

        client = _make_client()
        config = _make_config()

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            await session.send_client_content(
                turns=types.Content(
                    role="user", parts=[types.Part(text="Say hello in one word")]
                ),
                turn_complete=True,
            )
            got_transcript = False
            async for msg in session.receive():
                sc = getattr(msg, "server_content", None)
                if sc:
                    ot = getattr(sc, "output_transcription", None)
                    if ot and getattr(ot, "text", ""):
                        got_transcript = True
                        break
            assert got_transcript, "Expected at least one transcription"


@_skip
class TestMultiTurn:
    """Survive multiple turns without 1008 or session death."""

    @pytest.mark.asyncio
    async def test_two_turns(self):
        """Send two questions, survive both turns — no 1008."""
        from google.genai import types

        client = _make_client()
        config = _make_config()

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            for q in ["What is 2+2?", "What is 3+3?"]:
                await session.send_client_content(
                    turns=types.Content(
                        role="user", parts=[types.Part(text=q)]
                    ),
                    turn_complete=True,
                )
                # Consume all messages for this turn. The model may
                # respond with audio-only (no transcription text) so
                # we just verify the turn completes without error.
                got_any = False
                async for msg in session.receive():
                    sc = getattr(msg, "server_content", None)
                    if sc:
                        got_any = True
                        # Check for audio or transcript
                        if sc.model_turn and sc.model_turn.parts:
                            break
                        ot = getattr(sc, "output_transcription", None)
                        if ot and getattr(ot, "text", ""):
                            break
                assert got_any, f"No server content for: {q}"


@_skip
class TestConcurrentAudioVideo:
    """Send audio+video while model speaks — the real failure scenario."""

    @pytest.mark.asyncio
    async def test_audio_video_during_response(self):
        """Stream silence+blank JPEG while model responds — must not 1008."""
        from google.genai import types
        import numpy as np

        client = _make_client()
        config = _make_config()

        # Generate a small valid JPEG (8x8 black image).
        import cv2
        _, jpeg_buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        jpeg_bytes = jpeg_buf.tobytes()
        silence = b"\x00" * 3200  # 100ms at 16kHz

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            stop = asyncio.Event()
            errors: list[str] = []

            async def send_av():
                count = 0
                while not stop.is_set():
                    try:
                        blob = types.Blob(
                            data=silence, mime_type="audio/pcm;rate=16000"
                        )
                        await session.send_realtime_input(audio=blob)
                        if count % 50 == 0:
                            vblob = types.Blob(
                                data=jpeg_bytes, mime_type="image/jpeg"
                            )
                            await session.send_realtime_input(video=vblob)
                    except Exception as e:
                        errors.append(str(e))
                        break
                    count += 1
                    await asyncio.sleep(0.02)

            sender = asyncio.create_task(send_av())

            # Ask a question that will generate a multi-sentence response.
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text="Tell me a very short joke")],
                ),
                turn_complete=True,
            )

            # Read the full response.
            got_transcript = False
            async for msg in session.receive():
                sc = getattr(msg, "server_content", None)
                if sc:
                    ot = getattr(sc, "output_transcription", None)
                    if ot and getattr(ot, "text", ""):
                        got_transcript = True

            stop.set()
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass

            assert not errors, f"Audio/video sender hit errors: {errors}"
            assert got_transcript, "Expected transcription"


@_skip
class TestMultiTurnWithAudioVideo:
    """The exact scenario that was failing: multi-turn with concurrent audio/video."""

    @pytest.mark.asyncio
    async def test_two_turns_with_av_streaming(self):
        """Two turns with continuous audio/video — no 1008 across turn boundary."""
        from google.genai import types
        import numpy as np
        import cv2

        client = _make_client()
        config = _make_config()

        _, jpeg_buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        jpeg_bytes = jpeg_buf.tobytes()
        silence = b"\x00" * 3200

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            stop = asyncio.Event()
            errors: list[str] = []

            async def send_av():
                count = 0
                while not stop.is_set():
                    try:
                        blob = types.Blob(
                            data=silence, mime_type="audio/pcm;rate=16000"
                        )
                        await session.send_realtime_input(audio=blob)
                        if count % 50 == 0:
                            vblob = types.Blob(
                                data=jpeg_bytes, mime_type="image/jpeg"
                            )
                            await session.send_realtime_input(video=vblob)
                    except Exception as e:
                        errors.append(str(e))
                        break
                    count += 1
                    await asyncio.sleep(0.02)

            sender = asyncio.create_task(send_av())

            for i, q in enumerate(["Say hi", "Say bye"]):
                await session.send_client_content(
                    turns=types.Content(
                        role="user", parts=[types.Part(text=q)]
                    ),
                    turn_complete=True,
                )
                async for msg in session.receive():
                    sc = getattr(msg, "server_content", None)
                    if sc:
                        ot = getattr(sc, "output_transcription", None)
                        if ot and getattr(ot, "text", ""):
                            break
                # Brief pause between turns (like real usage).
                await asyncio.sleep(0.3)

            stop.set()
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass

            assert not errors, f"Audio/video errors across turns: {errors}"


@_skip
class TestToolCallWithGating:
    """Tool call with audio/video gating — the 1008 fix."""

    @pytest.mark.asyncio
    async def test_tool_call_with_av_gating(self):
        """Trigger a tool call, gate audio/video, respond — no 1008."""
        from google.genai import types
        import numpy as np
        import cv2

        client = _make_client()
        config = _make_config(with_tools=True)

        _, jpeg_buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        jpeg_bytes = jpeg_buf.tobytes()
        silence = b"\x00" * 3200

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            tool_pending = asyncio.Event()
            tool_pending.set()  # start open
            stop = asyncio.Event()
            errors: list[str] = []

            async def send_av():
                count = 0
                while not stop.is_set():
                    try:
                        await tool_pending.wait()
                        blob = types.Blob(
                            data=silence, mime_type="audio/pcm;rate=16000"
                        )
                        await session.send_realtime_input(audio=blob)
                        if count % 50 == 0:
                            vblob = types.Blob(
                                data=jpeg_bytes, mime_type="image/jpeg"
                            )
                            await session.send_realtime_input(video=vblob)
                    except Exception as e:
                        errors.append(str(e))
                        break
                    count += 1
                    await asyncio.sleep(0.02)

            sender = asyncio.create_task(send_av())

            # Ask something that should trigger the tool.
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(
                        text="Show me an overlay with title 'test'. Use the project_overlay tool."
                    )],
                ),
                turn_complete=True,
            )

            # Listen for tool call or transcript.
            got_tool_call = False
            async for msg in session.receive():
                if hasattr(msg, "tool_call") and msg.tool_call:
                    got_tool_call = True
                    tool_pending.clear()
                    responses = []
                    for fc in msg.tool_call.function_calls:
                        responses.append(
                            types.FunctionResponse(
                                name=fc.name,
                                response={"status": "displayed"},
                                id=fc.id,
                            )
                        )
                    await session.send_tool_response(
                        function_responses=responses
                    )
                    tool_pending.set()

                sc = getattr(msg, "server_content", None)
                if sc:
                    ot = getattr(sc, "output_transcription", None)
                    if ot and getattr(ot, "text", ""):
                        break

            stop.set()
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass

            assert not errors, f"Errors during tool call flow: {errors}"
            # Tool call may or may not fire (model's choice), but no crash.
