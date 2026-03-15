"""FastAPI WebSocket server bridging edge client to ADK Runner + Gemini Live."""

from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TableLight Backend")


# ---------------------------------------------------------------------------
# Pure message helpers (testable without ADK)
# ---------------------------------------------------------------------------

_BINARY_TYPES = {"audio", "video"}
_ALL_TYPES = {"audio", "video", "text", "close"}


def parse_client_message(data: dict) -> tuple[str, bytes | str | None]:
    """Parse a message from the edge client.

    Returns (message_type, payload).
    message_type: "audio", "video", "text", "close"
    payload: decoded bytes for audio/video, string for text, None for close.
    Raises ValueError for invalid messages.
    """
    if "type" not in data:
        raise ValueError("Message missing required 'type' field.")

    msg_type = data["type"]
    if msg_type not in _ALL_TYPES:
        raise ValueError(f"Unknown message type: '{msg_type}'.")

    if msg_type in _BINARY_TYPES:
        if "data" not in data:
            raise ValueError(f"Message type '{msg_type}' requires a 'data' field.")
        return msg_type, base64.b64decode(data["data"])

    if msg_type == "text":
        return "text", data.get("text", "")

    # close
    return "close", None


def format_audio_response(audio_data: bytes) -> dict:
    """Format an audio response for the edge client.

    Returns {"type": "audio", "data": base64_encoded_string}
    """
    return {"type": "audio", "data": base64.b64encode(audio_data).decode()}


def format_transcript(direction: str, text: str) -> dict:
    """Format a transcription message.

    direction: "in" or "out"
    Returns {"type": "transcript_in" or "transcript_out", "text": text}
    """
    return {"type": f"transcript_{direction}", "text": text}


def format_tool_result(function_name: str, response: dict) -> dict:
    """Format a tool result message for the edge client.

    Returns {"type": "tool_result", "name": function_name, "result": response}
    """
    return {"type": "tool_result", "name": function_name, "result": response}


def format_interrupted() -> dict:
    """Format an interruption message.

    Returns {"type": "interrupted"}
    """
    return {"type": "interrupted"}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/session")
async def session_endpoint(websocket: WebSocket) -> None:
    """Bridge an edge client to an ADK Gemini Live session."""
    # Lazy-import ADK so tests that only exercise the pure helpers above
    # never need ADK credentials or heavy dependencies at import time.
    from google.adk.agents import LiveRequestQueue  # type: ignore[import-untyped]
    from google.adk.agents.run_config import RunConfig, StreamingMode  # type: ignore[import-untyped]
    from google.adk.runners import Runner  # type: ignore[import-untyped]
    from google.adk.sessions import InMemorySessionService  # type: ignore[import-untyped]
    from google.genai import types  # type: ignore[import-untyped]

    from backend.agent import root_agent

    # --- Monkey-patch ADK to use separate audio=/video= streams ---
    # ADK sends everything via media= (deprecated mediaChunks), which
    # serializes audio and video into a single FIFO stream.  The Gemini
    # API processes audio= and video= as concurrent independent streams,
    # so routing them separately eliminates audio-behind-video latency.
    from google.adk.models import gemini_llm_connection as _glc  # type: ignore

    _original_send_realtime = _glc.GeminiLlmConnection.send_realtime

    async def _patched_send_realtime(self, input):
        if isinstance(input, types.Blob):
            mime = getattr(input, "mime_type", "") or ""
            if mime.startswith("audio/"):
                await self._gemini_session.send_realtime_input(audio=input)
            elif mime.startswith("image/"):
                await self._gemini_session.send_realtime_input(video=input)
            else:
                await self._gemini_session.send_realtime_input(media=input)
        elif isinstance(input, types.ActivityStart):
            await self._gemini_session.send_realtime_input(activity_start=input)
        elif isinstance(input, types.ActivityEnd):
            await self._gemini_session.send_realtime_input(activity_end=input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    _glc.GeminiLlmConnection.send_realtime = _patched_send_realtime
    logger.info("Patched ADK to use separate audio/video streams")

    # Verify the patch took effect
    assert _glc.GeminiLlmConnection.send_realtime is _patched_send_realtime

    await websocket.accept()

    # Read init message from client.
    init_msg = await websocket.receive_json()
    text_only = init_msg.get("text_only", False)

    # Per-connection ADK plumbing.
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="tablelight",
        session_service=session_service,
    )
    live_request_queue = LiveRequestQueue()

    session = await session_service.create_session(
        app_name="tablelight", user_id=str(id(websocket))
    )

    # Always use AUDIO modality — the native-audio model requires it.
    # In --no-audio mode, the client sends silence to keep the stream alive
    # and uses send_content() for text input.
    run_config = RunConfig(
        streaming_mode=StreamingMode.BIDI,
        response_modalities=[types.Modality.AUDIO],
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
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=200,
                silence_duration_ms=1000,
            )
        ),
        output_audio_transcription={},
        input_audio_transcription={},
    )
    logger.info(
        "Session started (text_only=%s)", text_only
    )

    # Mutable container so receive_from_client always sees the current queue.
    # gemini_session is populated once run_live establishes the connection,
    # allowing audio to bypass ADK's FIFO queue and go directly to Gemini.
    # audio_queue is a dedicated high-priority queue drained by a separate
    # task so audio sends are serialized but never blocked by video.
    audio_queue: asyncio.Queue = asyncio.Queue()

    state = {
        "live_request_queue": live_request_queue,
        "gemini_session": None,  # set by monkey-patch
    }

    # Capture the session from send_realtime calls
    _prev_patched = _glc.GeminiLlmConnection.send_realtime

    async def _capturing_send_realtime(self_conn, input):
        if state["gemini_session"] is None and hasattr(self_conn, '_gemini_session'):
            state["gemini_session"] = self_conn._gemini_session
            logger.info("Captured Gemini session for direct audio bypass")
        return await _prev_patched(self_conn, input)

    _glc.GeminiLlmConnection.send_realtime = _capturing_send_realtime

    async def _audio_sender():
        """Dedicated task: drain audio_queue and send directly to Gemini."""
        count = 0
        while True:
            blob = await audio_queue.get()
            count += 1
            gs = state.get("gemini_session")
            if gs:
                try:
                    await gs.send_realtime_input(audio=blob)
                    if count % 40 == 0:
                        import time
                        qsize = audio_queue.qsize()
                        logger.info("Audio sent #%d (queue=%d, t=%.1f)",
                                    count, qsize, time.time() % 1000)
                except Exception as e:
                    logger.warning("Audio send failed: %s", e)
            else:
                state["live_request_queue"].send_realtime(blob)

    async def receive_from_client() -> None:
        """Receive audio/video/text from edge client, forward to ADK.

        Audio is sent directly to the Gemini session (bypassing ADK's
        FIFO queue) for lowest latency. Video and text go through the
        queue as before.
        """
        try:
            while True:
                raw = await websocket.receive_json()
                msg_type, payload = parse_client_message(raw)
                lrq = state["live_request_queue"]

                try:
                    if msg_type == "audio":
                        blob = types.Blob(
                            data=payload, mime_type="audio/pcm;rate=16000"
                        )
                        # Route to dedicated audio queue (bypasses ADK's
                        # FIFO queue, sent by _audio_sender task).
                        audio_queue.put_nowait(blob)
                    elif msg_type == "video":
                        lrq.send_realtime(
                            types.Blob(data=payload, mime_type="image/jpeg")
                        )
                    elif msg_type == "text":
                        lrq.send_content(
                            types.Content(
                                role="user", parts=[types.Part(text=payload)]
                            )
                        )
                    elif msg_type == "close":
                        lrq.close()
                        break
                except Exception:
                    # Queue may be closed during reconnection — drop the
                    # message and keep reading from the client.
                    pass
        except WebSocketDisconnect:
            state["live_request_queue"].close()
        except Exception:
            logger.exception("Error in receive_from_client")
            state["live_request_queue"].close()

    async def _drain_pending_tool_calls() -> None:
        """Forward any tool calls stashed by _before_tool to the client.

        _before_tool runs synchronously inside ADK before the event is
        yielded.  If the Live API crashes right after (1011), the event
        with part.function_call never reaches _process_events.  This
        drainer ensures the client always gets the tool call.
        """
        import queue as _q
        from backend.agent import pending_tool_calls
        while True:
            try:
                name, args = pending_tool_calls.get_nowait()
                await websocket.send_json(format_tool_result(name, args))
                logger.info("Forwarded pending tool call: %s", name)
            except _q.Empty:
                break

    async def _process_events(lrq: LiveRequestQueue) -> None:
        """Process events from a single run_live session."""
        event_count = 0
        # Track conversation phase to filter consolidated transcript replays.
        # The Live API sends word-by-word transcripts during speech, then
        # replays the full transcript as a single event after the turn ends.
        # Phase: "listening" → "responding" → "listening" ...
        phase = "listening"
        sent_out_text = ""  # cumulative output text for dedup

        async for event in runner.run_live(
            session=session,
            live_request_queue=lrq,
            run_config=run_config,
        ):
            event_count += 1

            # Drain any tool calls that _before_tool stashed.
            await _drain_pending_tool_calls()

            # Process content parts: audio, function calls/responses.
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if (
                        part.inline_data
                        and part.inline_data.mime_type
                        and part.inline_data.mime_type.startswith("audio/")
                    ):
                        await websocket.send_json(
                            format_audio_response(part.inline_data.data)
                        )

            # Transcriptions — use phase tracking to skip consolidated
            # replays. The Live API sends:
            #   1. Word-by-word input transcripts while user speaks
            #   2. Word-by-word output transcripts while model speaks
            #   3. A consolidated replay of both after the turn ends
            # We forward (1) and (2), skip (3).
            if event.input_transcription:
                t = event.input_transcription.text or ""
                if t.strip() and "<ctrl" not in t:
                    if phase == "responding":
                        # Input transcript during model response =
                        # consolidated replay. Skip.
                        pass
                    else:
                        phase = "listening"
                        sent_out_text = ""  # new turn, reset output dedup
                        await websocket.send_json(
                            format_transcript("in", t)
                        )
            if event.output_transcription:
                t = event.output_transcription.text or ""
                if t.strip() and "<ctrl" not in t:
                    # Skip if this text is already in what we've sent
                    # (= consolidated replay of the response).
                    if t.strip() in sent_out_text:
                        pass
                    else:
                        if phase == "listening":
                            phase = "responding"
                        sent_out_text += t
                        await websocket.send_json(
                            format_transcript("out", t)
                        )

            # Interruption — reset phase so next input is treated as new.
            if event.interrupted:
                phase = "listening"
                sent_out_text = ""
                logger.info("Event #%d: INTERRUPTED", event_count)
                await websocket.send_json(format_interrupted())

    # Event set when receive_from_client exits (client disconnected).
    client_done = asyncio.Event()

    async def send_to_client() -> None:
        """Run ADK agent with auto-reconnect on Live API crashes.

        The native-audio model's Live API is flaky after tool execution —
        it often drops the WebSocket with 1011.  We catch that, create a
        fresh LiveRequestQueue, and restart run_live so the client keeps
        working transparently.

        This function stays alive as long as the client is connected,
        even if the Gemini session crashes repeatedly.
        """
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            if client_done.is_set():
                return
            try:
                await _process_events(state["live_request_queue"])
                return  # clean exit
            except WebSocketDisconnect:
                return
            except Exception:
                # Drain any tool calls stashed before the crash.
                await _drain_pending_tool_calls()
                logger.exception(
                    "Live session crashed (attempt %d/%d) — reconnecting",
                    attempt, max_retries,
                )
                if client_done.is_set():
                    return
                if attempt < max_retries:
                    new_lrq = LiveRequestQueue()
                    state["live_request_queue"] = new_lrq
                    try:
                        await websocket.send_json(
                            format_transcript(
                                "out",
                                "(Reconnecting to Gemini...)",
                            )
                        )
                    except Exception:
                        return  # client gone
                    await asyncio.sleep(1)
                else:
                    logger.error("Max retries reached — giving up")
                    try:
                        await websocket.send_json(
                            format_transcript(
                                "out",
                                "(Session lost — please restart the client)",
                            )
                        )
                    except Exception:
                        pass

    recv_task = asyncio.create_task(receive_from_client())
    send_task = asyncio.create_task(send_to_client())
    audio_task = asyncio.create_task(_audio_sender())

    # Wait for the client to disconnect (receive_from_client exits).
    # send_to_client may exit/restart independently.
    try:
        await recv_task
    finally:
        client_done.set()
        send_task.cancel()
        audio_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        try:
            await audio_task
        except asyncio.CancelledError:
            pass
