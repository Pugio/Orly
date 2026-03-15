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
        output_audio_transcription={},
        input_audio_transcription={},
    )
    logger.info(
        "Session started (text_only=%s)", text_only
    )

    async def receive_from_client() -> None:
        """Receive audio/video/text from edge client, forward to ADK."""
        try:
            while True:
                raw = await websocket.receive_json()
                msg_type, payload = parse_client_message(raw)

                if msg_type == "audio":
                    live_request_queue.send_realtime(
                        types.Blob(
                            data=payload, mime_type="audio/pcm;rate=16000"
                        )
                    )
                elif msg_type == "video":
                    live_request_queue.send_realtime(
                        types.Blob(data=payload, mime_type="image/jpeg")
                    )
                elif msg_type == "text":
                    live_request_queue.send_content(
                        types.Content(
                            role="user", parts=[types.Part(text=payload)]
                        )
                    )
                elif msg_type == "close":
                    live_request_queue.close()
                    break
        except WebSocketDisconnect:
            live_request_queue.close()
        except Exception:
            logger.exception("Error in receive_from_client")
            live_request_queue.close()

    async def send_to_client() -> None:
        """Run ADK agent, forward events to edge client."""
        print("[BACKEND] send_to_client starting run_live...")
        try:
            async for event in runner.run_live(
                session=session,
                live_request_queue=live_request_queue,
                run_config=run_config,
            ):
                # Debug: log event summary
                parts_summary = []
                if event.content and event.content.parts:
                    for p in event.content.parts:
                        if p.inline_data:
                            parts_summary.append(f"inline_data({p.inline_data.mime_type})")
                        if p.text:
                            parts_summary.append(f"text({p.text[:50]})")
                        if p.function_call:
                            parts_summary.append(f"function_call({p.function_call.name})")
                        if p.function_response:
                            parts_summary.append(f"function_response({p.function_response.name})")
                if parts_summary:
                    print(f"[BACKEND] Event: {', '.join(parts_summary)}")
                if event.input_transcription:
                    print(f"[BACKEND] Input: {event.input_transcription.text[:80]}")
                if event.output_transcription:
                    print(f"[BACKEND] Output: {event.output_transcription.text[:80]}")

                # Process content parts: audio, function calls/responses.
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        # Audio response → forward to edge client.
                        if (
                            part.inline_data
                            and part.inline_data.mime_type
                            and part.inline_data.mime_type.startswith("audio/")
                        ):
                            await websocket.send_json(
                                format_audio_response(part.inline_data.data)
                            )

                        # Text response — filter out audio control tokens.
                        if part.text:
                            text = part.text.strip()
                            if text and "<ctrl" not in text:
                                await websocket.send_json(
                                    format_transcript("out", text)
                                )

                        # Function call → ADK executes automatically,
                        # but we forward the call info to the edge client
                        # so it can render the overlay.
                        if part.function_call:
                            fc = part.function_call
                            await websocket.send_json(
                                format_tool_result(
                                    fc.name,
                                    dict(fc.args) if fc.args else {},
                                )
                            )

                # Transcriptions (flat fields on Event).
                if event.input_transcription:
                    await websocket.send_json(
                        format_transcript("in", event.input_transcription.text)
                    )
                if event.output_transcription:
                    await websocket.send_json(
                        format_transcript("out", event.output_transcription.text)
                    )

                # Interruption (flat field on Event).
                if event.interrupted:
                    await websocket.send_json(format_interrupted())
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Error in send_to_client")

    await asyncio.gather(receive_from_client(), send_to_client())
