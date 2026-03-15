"""WebSocket client that connects to the Cloud Run backend."""

import asyncio
import base64
import json

import websockets


class TableLightClient:
    """WebSocket client that connects to the Cloud Run backend.

    Sends audio/video/text to the backend and dispatches received
    messages (audio, tool_result, transcript, interrupted) to
    registered callbacks.
    """

    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.ws = None
        self._on_audio = None       # async def(audio_bytes)
        self._on_tool_result = None  # async def(name, result)
        self._on_transcript = None   # async def(direction, text)
        self._on_interrupted = None  # async def()

    async def connect(self, text_only: bool = False):
        """Connect to backend WebSocket.

        Args:
            text_only: If True, request text-only mode (no audio I/O).
        """
        self.ws = await websockets.connect(self.backend_url)
        # Send init message to configure session mode.
        await self.ws.send(json.dumps({"text_only": text_only}))

    async def send_audio(self, pcm_bytes: bytes):
        """Send audio chunk to backend."""
        msg = {"type": "audio", "data": base64.b64encode(pcm_bytes).decode()}
        await self.ws.send(json.dumps(msg))

    async def send_video(self, jpeg_bytes: bytes):
        """Send video frame to backend."""
        msg = {"type": "video", "data": base64.b64encode(jpeg_bytes).decode()}
        await self.ws.send(json.dumps(msg))

    async def send_text(self, text: str):
        """Send text message to backend."""
        await self.ws.send(json.dumps({"type": "text", "text": text}))

    async def close(self):
        """Close connection."""
        if self.ws:
            await self.ws.close()

    def on_audio(self, callback):
        self._on_audio = callback

    def on_tool_result(self, callback):
        self._on_tool_result = callback

    def on_transcript(self, callback):
        self._on_transcript = callback

    def on_interrupted(self, callback):
        self._on_interrupted = callback

    async def receive_loop(self):
        """Receive and dispatch messages from backend."""
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "audio" and self._on_audio:
                    await self._on_audio(base64.b64decode(msg["data"]))
                elif msg_type == "tool_result" and self._on_tool_result:
                    await self._on_tool_result(msg["name"], msg["result"])
                elif msg_type in ("transcript_in", "transcript_out") and self._on_transcript:
                    direction = "in" if msg_type == "transcript_in" else "out"
                    await self._on_transcript(direction, msg["text"])
                elif msg_type == "interrupted" and self._on_interrupted:
                    await self._on_interrupted()
        except websockets.exceptions.ConnectionClosed:
            print("[TableLight] Backend connection closed.")
        except Exception as e:
            print(f"[TableLight] Receive error: {e}")
