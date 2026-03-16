"""FastAPI WebSocket server bridging edge client to Gemini Live API (raw SDK)."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Orly Backend")


# ---------------------------------------------------------------------------
# Binary WebSocket protocol constants
# ---------------------------------------------------------------------------

PREFIX_AUDIO_IN = b"\x01"   # client -> server: PCM audio
PREFIX_VIDEO_IN = b"\x02"   # client -> server: JPEG video
PREFIX_AUDIO_OUT = b"\x03"  # server -> client: PCM audio


# ---------------------------------------------------------------------------
# Pure message helpers (testable without Gemini SDK)
# ---------------------------------------------------------------------------


def parse_binary_message(data: bytes) -> tuple[str, bytes]:
    """Parse a binary WebSocket frame from the edge client.

    Returns (message_type, payload).
    message_type: "audio" or "video".
    payload: raw bytes (PCM or JPEG).
    Raises ValueError for unknown prefix.
    """
    if len(data) < 2:
        raise ValueError("Binary message too short.")
    prefix = data[0:1]
    payload = data[1:]
    if prefix == PREFIX_AUDIO_IN:
        return "audio", payload
    elif prefix == PREFIX_VIDEO_IN:
        return "video", payload
    else:
        raise ValueError(f"Unknown binary prefix: 0x{prefix[0]:02x}")


def parse_text_message(data: str) -> tuple[str, str | None]:
    """Parse a text (JSON) WebSocket frame from the edge client.

    Returns (message_type, payload).
    message_type: "text", "close", "notification", or "set_proactive".
    payload: string for text/notification/set_proactive, None for close.
    Raises ValueError for invalid messages.
    """
    msg = json.loads(data)
    msg_type = msg.get("type")
    if msg_type is None:
        raise ValueError("Message missing required 'type' field.")
    if msg_type == "text":
        return "text", msg.get("text", "")
    if msg_type == "close":
        return "close", None
    if msg_type == "notification":
        return "notification", json.dumps(
            {"source": msg.get("source", ""), "text": msg.get("text", "")}
        )
    if msg_type == "set_proactive":
        return "set_proactive", str(msg.get("enabled", True)).lower()
    raise ValueError(f"Unknown text message type: '{msg_type}'.")


def format_audio_response(audio_data: bytes) -> bytes:
    """Format an audio response as a binary frame for the edge client.

    Returns PREFIX_AUDIO_OUT + raw PCM bytes.
    """
    return PREFIX_AUDIO_OUT + audio_data


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
# Tool execution helper
# ---------------------------------------------------------------------------


def _clean_args(args: dict, func) -> dict:
    """Strip unexpected kwargs that Gemini sometimes sends (training artifacts)."""
    valid = set(inspect.signature(func).parameters.keys())
    return {k: v for k, v in args.items() if k in valid}


def execute_tool(function_name: str, args: dict, registry: dict,
                 latency_tracker=None) -> dict:
    """Look up and execute a tool function, returning its result dict.

    Strips unexpected kwargs before calling. Returns an error dict if the
    function is not found or raises an exception.
    """
    func = registry.get(function_name)
    if func is None:
        return {"status": "error", "message": f"Unknown tool: {function_name}"}
    try:
        clean = _clean_args(args, func)
        if latency_tracker:
            latency_tracker.begin("tool_exec")
        result = func(**clean)
        if latency_tracker:
            latency_tracker.end("tool_exec")
        return result
    except Exception as e:
        if latency_tracker:
            latency_tracker.end("tool_exec")
        logger.exception("Tool %s raised an exception", function_name)
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Gemini client singleton (shared across connections)
# ---------------------------------------------------------------------------

_gemini_client_cache: dict = {}


def _get_gemini_client(genai):
    """Get or create a shared Gemini client. Cached per process."""
    if "client" not in _gemini_client_cache:
        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not api_key:
            import subprocess
            try:
                api_key = subprocess.check_output(
                    ["llm", "keys", "get", "gemini"], text=True
                ).strip()
            except Exception:
                pass
        is_vertex = not bool(api_key)
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()
        _gemini_client_cache["client"] = client
        _gemini_client_cache["is_vertex"] = is_vertex
        logger.info("Created Gemini client (vertex=%s)", is_vertex)
    return _gemini_client_cache["client"], _gemini_client_cache["is_vertex"]


# ---------------------------------------------------------------------------
# WebSocket endpoint — raw google-genai Live API
# ---------------------------------------------------------------------------


@app.websocket("/ws/session")
async def session_endpoint(websocket: WebSocket) -> None:
    """Bridge an edge client to a Gemini Live session (raw SDK)."""
    # Lazy imports so pure-helper tests never need credentials.
    from google import genai
    from google.genai import types

    from backend.agent import (
        MODEL,
        SYSTEM_PROMPT,
        TOOL_DECLARATIONS,
        TOOL_REGISTRY,
    )

    await websocket.accept()

    # Read init message from client (always JSON text).
    init_msg = await websocket.receive_json()
    text_only = init_msg.get("text_only", False)
    logger.info("Session started (text_only=%s)", text_only)

    # Get or create the shared Gemini client (one per process).
    client, is_vertex = _get_gemini_client(genai)

    # Build config — some features are Vertex AI only.
    config_kwargs = dict(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=SYSTEM_PROMPT,
        tools=[{"function_declarations": TOOL_DECLARATIONS}],
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
                # Client-side audio processing (noise gate + echo suppression)
                # now cleans the signal before it reaches the server, so we
                # can use HIGH start sensitivity for better speech detection
                # without false triggers from speaker echo.
                # LOW end sensitivity = less eager to cut off mid-sentence.
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=100,
                silence_duration_ms=500,
            )
        ),
    )
    # Transcription, context compression, session resumption, and
    # proactive audio may be Vertex AI only depending on the model.
    # Google AI returns 1008 for unsupported features.
    if is_vertex:
        config_kwargs["input_audio_transcription"] = {}
        config_kwargs["output_audio_transcription"] = {}
        config_kwargs["context_window_compression"] = (
            types.ContextWindowCompressionConfig(
                trigger_tokens=20000,
                sliding_window=types.SlidingWindow(target_tokens=10000),
            )
        )
        config_kwargs["session_resumption"] = (
            types.SessionResumptionConfig(transparent=True)
        )
        config_kwargs["proactivity"] = (
            types.ProactivityConfig(proactive_audio=True)
        )
    config = types.LiveConnectConfig(**config_kwargs)
    logger.info(
        "LiveConnectConfig keys: %s (model=%s, vertex=%s)",
        sorted(config_kwargs.keys()), MODEL, is_vertex,
    )

    # --- Audio pipeline stats (logged periodically) ---
    class _SessionStats:
        def __init__(self):
            self.audio_in = 0       # chunks from client
            self.audio_to_gemini = 0  # chunks sent to Gemini
            self.video_to_gemini = 0
            self.audio_from_gemini = 0
            self.last_audio_in = 0.0
            self.last_audio_out = 0.0
            self.start = time.monotonic()

        def log_and_reset(self):
            now = time.monotonic()
            elapsed = now - self.start
            last_in = f"{now - self.last_audio_in:.0f}s ago" if self.last_audio_in else "never"
            last_out = f"{now - self.last_audio_out:.0f}s ago" if self.last_audio_out else "never"
            logger.info(
                "Session stats [%.0fs]: audio_in=%d sent_to_gemini=%d "
                "video_to_gemini=%d audio_from_gemini=%d | "
                "last_audio_in=%s last_audio_out=%s",
                elapsed, self.audio_in, self.audio_to_gemini,
                self.video_to_gemini, self.audio_from_gemini,
                last_in, last_out,
            )
            self.audio_in = 0
            self.audio_to_gemini = 0
            self.video_to_gemini = 0
            self.audio_from_gemini = 0
            self.start = now

    stats = _SessionStats()

    async def _stats_loop():
        while not client_done.is_set():
            await asyncio.sleep(15)
            stats.log_and_reset()

    # Shared mutable state across tasks.
    client_done = asyncio.Event()
    # Handle for session resumption after crash/go_away.
    resumption_handle: dict[str, str | None] = {"handle": None}

    async def _run_session() -> None:
        """Connect to Gemini and run the send/receive loops.

        Returns normally on clean close or go_away. Raises on crash.
        """
        resume_cfg = config
        if resumption_handle["handle"] and is_vertex:
            # Session resumption is Vertex AI only. Sending a handle on
            # Google AI triggers 1008 "Operation is not implemented".
            resume_cfg = types.LiveConnectConfig(
                response_modalities=config.response_modalities,
                system_instruction=config.system_instruction,
                tools=config.tools,
                speech_config=config.speech_config,
                realtime_input_config=config.realtime_input_config,
                input_audio_transcription=config.input_audio_transcription,
                output_audio_transcription=config.output_audio_transcription,
                context_window_compression=config.context_window_compression,
                session_resumption=types.SessionResumptionConfig(
                    handle=resumption_handle["handle"], transparent=True
                ),
            )

        async with client.aio.live.connect(
            model=MODEL, config=resume_cfg
        ) as session:

            # Gate: block realtime input while a tool call is pending.
            # The Gemini Live API enters a blocking state during tool calls
            # and will terminate the connection (1008 policy violation) if
            # audio/video frames arrive before the tool response is sent.
            tool_call_pending = asyncio.Event()
            tool_call_pending.set()  # start open (not pending)

            # --- Sender: read from WebSocket, forward to Gemini ---
            async def send_from_client() -> None:
                """Read from WebSocket, forward audio/video/text to Gemini.

                ALL sends use send_realtime_input to avoid interleaving
                with send_client_content (which the SDK warns against).
                """
                try:
                    while not client_done.is_set():
                        try:
                            ws_msg = await asyncio.wait_for(
                                websocket.receive(), timeout=0.5
                            )
                        except asyncio.TimeoutError:
                            continue
                        except WebSocketDisconnect:
                            client_done.set()
                            return

                        if ws_msg.get("type") == "websocket.disconnect":
                            client_done.set()
                            return

                        # Binary frame
                        if "bytes" in ws_msg and ws_msg["bytes"] is not None:
                            raw_bytes = ws_msg["bytes"]
                            msg_type, payload = parse_binary_message(raw_bytes)
                            if msg_type == "audio":
                                stats.audio_in += 1
                                stats.last_audio_in = time.monotonic()
                                # Wait for any pending tool call to complete
                                # before sending audio to avoid 1008 errors.
                                await tool_call_pending.wait()
                                # Yield so receiver can process a just-arrived
                                # tool_call and close the gate before we send.
                                await asyncio.sleep(0)
                                if not tool_call_pending.is_set():
                                    continue  # gate just closed, drop frame
                                blob = types.Blob(
                                    data=payload, mime_type="audio/pcm;rate=16000"
                                )
                                await session.send_realtime_input(audio=blob)
                                stats.audio_to_gemini += 1
                            elif msg_type == "video":
                                await tool_call_pending.wait()
                                await asyncio.sleep(0)
                                if not tool_call_pending.is_set():
                                    continue
                                blob = types.Blob(
                                    data=payload, mime_type="image/jpeg"
                                )
                                await session.send_realtime_input(video=blob)
                                stats.video_to_gemini += 1

                        # Text frame
                        elif "text" in ws_msg and ws_msg["text"] is not None:
                            raw_text = ws_msg["text"]
                            msg_type, payload = parse_text_message(raw_text)
                            if msg_type == "text":
                                # Use send_realtime_input for text to avoid
                                # interleaving with send_client_content (which
                                # the SDK warns causes unexpected behavior).
                                await tool_call_pending.wait()
                                await asyncio.sleep(0)
                                if not tool_call_pending.is_set():
                                    continue
                                await session.send_realtime_input(
                                    text=payload
                                )
                            elif msg_type == "notification":
                                notif = json.loads(payload)
                                source = notif.get("source", "unknown")
                                text = notif.get("text", "")
                                notification_text = (
                                    f"[NOTIFICATION from {source}]: {text}"
                                )
                                await tool_call_pending.wait()
                                await asyncio.sleep(0)
                                if not tool_call_pending.is_set():
                                    continue
                                await session.send_realtime_input(
                                    text=notification_text
                                )
                            elif msg_type == "close":
                                client_done.set()
                                return
                except WebSocketDisconnect:
                    # Client (edge device) disconnected — stop everything.
                    client_done.set()
                except Exception:
                    # Gemini session error (1008/1011/etc.) — do NOT set
                    # client_done.  Let _run_session exit and the retry
                    # loop will reconnect to Gemini while keeping the
                    # client WebSocket alive.
                    logger.exception("Error in send_from_client")

            # --- Receiver: process events from Gemini ---
            async def receive_from_gemini() -> None:
                """Receive events from Gemini, forward to edge client."""
                # Phase tracking for transcript dedup.
                phase = "listening"
                sent_out_text = ""

                try:
                  while not client_done.is_set():
                    # session.receive() yields messages for ONE model
                    # turn then stops (breaks on turn_complete).  We
                    # wrap it in an outer loop so we keep listening
                    # across multiple turns for the life of the session.
                    async for msg in session.receive():
                        if client_done.is_set():
                            return

                        # --- Session resumption updates ---
                        if (
                            hasattr(msg, "session_resumption_update")
                            and msg.session_resumption_update
                        ):
                            update = msg.session_resumption_update
                            if hasattr(update, "new_handle") and update.new_handle:
                                resumption_handle["handle"] = update.new_handle
                                logger.info("Updated session resumption handle")

                        # --- go_away: server wants us to reconnect ---
                        if hasattr(msg, "go_away") and msg.go_away:
                            logger.info("Received go_away — will reconnect")
                            return

                        # --- Tool calls ---
                        if hasattr(msg, "tool_call") and msg.tool_call:
                            # Gate: block audio/video sends while we
                            # process tool calls.  The Live API will
                            # send 1008 if realtime input arrives
                            # before the tool response.
                            tool_call_pending.clear()
                            try:
                                responses = []
                                for fc in msg.tool_call.function_calls:
                                    fn_name = fc.name
                                    fn_args = dict(fc.args) if fc.args else {}
                                    logger.info(
                                        "Tool call: %s(%s)", fn_name, fn_args
                                    )
                                    result = execute_tool(
                                        fn_name, fn_args, TOOL_REGISTRY
                                    )
                                    # Send consolidated tool name + args
                                    # directly to client (no translation).
                                    client_payload = {**fn_args, **result}
                                    try:
                                        await websocket.send_json(
                                            format_tool_result(
                                                fn_name, client_payload
                                            )
                                        )
                                    except Exception:
                                        pass
                                    responses.append(
                                        types.FunctionResponse(
                                            name=fn_name,
                                            response=result,
                                            id=fc.id,
                                        )
                                    )
                                # Send tool responses back to Gemini.
                                await session.send_tool_response(
                                    function_responses=responses
                                )
                            finally:
                                # Brief cooldown before reopening the gate.
                                # Without this, queued audio frames flood in
                                # simultaneously and Gemini's VAD interprets
                                # the burst as speech, causing INTERRUPTED.
                                await asyncio.sleep(0.3)
                                tool_call_pending.set()
                            continue

                        # --- Tool call cancellation (user interrupted during tool exec) ---
                        tc_cancel = getattr(msg, "tool_call_cancellation", None)
                        if tc_cancel:
                            cancelled_ids = getattr(tc_cancel, "ids", [])
                            logger.info("Tool call cancelled: %s", cancelled_ids)
                            # Re-open the gate in case it was stuck closed.
                            tool_call_pending.set()
                            continue

                        # --- Server content (audio, transcriptions, interruptions) ---
                        sc = getattr(msg, "server_content", None)
                        if sc is None:
                            continue

                        # Interruption
                        if sc.interrupted:
                            phase = "listening"
                            sent_out_text = ""
                            logger.info("INTERRUPTED")
                            try:
                                await websocket.send_json(format_interrupted())
                            except Exception:
                                pass
                            continue

                        # Audio and text parts
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if (
                                    part.inline_data
                                    and part.inline_data.mime_type
                                    and part.inline_data.mime_type.startswith(
                                        "audio/"
                                    )
                                ):
                                    stats.audio_from_gemini += 1
                                    stats.last_audio_out = time.monotonic()
                                    try:
                                        await websocket.send_bytes(
                                            format_audio_response(
                                                part.inline_data.data
                                            )
                                        )
                                    except Exception:
                                        pass

                        # Input transcription
                        it = getattr(sc, "input_transcription", None)
                        if it:
                            t = getattr(it, "text", "") or ""
                            if t.strip() and "<ctrl" not in t:
                                if phase == "responding":
                                    pass  # consolidated replay, skip
                                else:
                                    phase = "listening"
                                    sent_out_text = ""
                                    try:
                                        await websocket.send_json(
                                            format_transcript("in", t)
                                        )
                                    except Exception:
                                        pass

                        # Output transcription
                        ot = getattr(sc, "output_transcription", None)
                        if ot:
                            t = getattr(ot, "text", "") or ""
                            if t.strip() and "<ctrl" not in t:
                                if t.strip() in sent_out_text:
                                    pass  # consolidated replay, skip
                                else:
                                    if phase == "listening":
                                        phase = "responding"
                                    sent_out_text += t
                                    try:
                                        await websocket.send_json(
                                            format_transcript("out", t)
                                        )
                                    except Exception:
                                        pass

                    # async for ended — turn_complete or stream closed.
                    # If the underlying WS is closed, break out of
                    # the outer while loop too.
                    try:
                        ws = getattr(session, "_ws", None)
                        if ws and hasattr(ws, "close_code") and ws.close_code is not None:
                            logger.info(
                                "Gemini WS closed: code=%s reason=%s",
                                ws.close_code,
                                getattr(ws, "close_reason", ""),
                            )
                            break
                    except Exception:
                        pass

                except WebSocketDisconnect:
                    client_done.set()
                except Exception:
                    logger.exception("Unexpected error in receive_from_gemini")
                    raise  # let the exception propagate so _run_session crashes
                finally:
                    logger.info("receive_from_gemini exited (client_done=%s)", client_done.is_set())

            # Run sender and receiver concurrently.
            sender = asyncio.create_task(send_from_client(), name="sender")
            receiver = asyncio.create_task(receive_from_gemini(), name="receiver")
            stats_task = asyncio.create_task(_stats_loop(), name="stats")
            try:
                done, pending = await asyncio.wait(
                    {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
                )
                stats_task.cancel()
                done_names = [t.get_name() for t in done]
                pending_names = [t.get_name() for t in pending]
                logger.info(
                    "Task finished: %s (still running: %s)",
                    done_names, pending_names,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                # Re-raise exceptions from completed tasks.
                for task in done:
                    exc = task.exception()
                    if exc and not isinstance(exc, asyncio.CancelledError):
                        raise exc
            except WebSocketDisconnect:
                client_done.set()

    # --- Main loop with auto-reconnect ---
    import random

    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        if client_done.is_set():
            break
        attempt += 1
        try:
            await _run_session()
            # Successful session — reset retry counter.
            attempt = 0
            if client_done.is_set():
                break
            # Clean exit from _run_session (e.g. go_away) — reconnect.
            logger.info("Session ended cleanly — reconnecting")
            await asyncio.sleep(0.3)
        except WebSocketDisconnect:
            break
        except Exception as exc:
            exc_str = str(exc)
            # Broad transient classification: 1006 (abnormal close),
            # 1008 (policy / tool race), 1011 (internal), 500/503.
            _TRANSIENT = ("1006", "1008", "1011", "500", "503")
            is_transient = any(code in exc_str for code in _TRANSIENT)
            if is_transient:
                logger.warning(
                    "Gemini session dropped (attempt %d/%d) — reconnecting: %s",
                    attempt,
                    max_retries,
                    exc_str[:120],
                )
            else:
                logger.exception(
                    "Live session crashed (attempt %d/%d) — reconnecting",
                    attempt,
                    max_retries,
                )
            if client_done.is_set():
                break
            if attempt < max_retries:
                # Exponential backoff with jitter.
                base = 0.3 if is_transient else 1.0
                delay = min(base * (2 ** (attempt - 1)), 30)
                delay += random.uniform(0, 0.5)
                await asyncio.sleep(delay)
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
