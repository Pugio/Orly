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
        self._on_audio = None        # async def(audio_bytes)
        self._on_tool_result = None  # async def(name, result)
        self._on_transcript = None   # async def(direction, text)
        self._on_interrupted = None  # async def()
        self._on_refresh_view = None # async def()

    async def connect(self, text_only: bool = False):
        """Connect to backend WebSocket.

        Args:
            text_only: If True, request text-only mode (no audio I/O).
        """
        self.ws = await websockets.connect(self.backend_url)
        # Send init message to configure session mode.
        await self.ws.send(json.dumps({"text_only": text_only}))

    @property
    def connected(self) -> bool:
        """Check if the WebSocket connection is open."""
        if not self.ws:
            return False
        # websockets library uses .closed as a bool property
        closed = getattr(self.ws, "closed", None)
        if closed is True:
            return False
        return True

    async def send_audio(self, pcm_bytes: bytes):
        """Send audio chunk to backend."""
        if not self.connected:
            return
        msg = {"type": "audio", "data": base64.b64encode(pcm_bytes).decode()}
        await self.ws.send(json.dumps(msg))

    async def send_video(self, jpeg_bytes: bytes):
        """Send video frame to backend."""
        if not self.connected:
            return
        msg = {"type": "video", "data": base64.b64encode(jpeg_bytes).decode()}
        await self.ws.send(json.dumps(msg))

    async def send_text(self, text: str):
        """Send text message to backend."""
        if not self.connected:
            return
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

    def on_refresh_view(self, callback):
        self._on_refresh_view = callback

    async def receive_loop(self):
        """Receive and dispatch messages from backend."""
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "audio" and self._on_audio:
                    await self._on_audio(base64.b64decode(msg["data"]))
                elif msg_type == "tool_result":
                    name = msg["name"]
                    if name == "refresh_view" and self._on_refresh_view:
                        await self._on_refresh_view()
                    elif self._on_tool_result:
                        await self._on_tool_result(name, msg["result"])
                elif msg_type in ("transcript_in", "transcript_out") and self._on_transcript:
                    direction = "in" if msg_type == "transcript_in" else "out"
                    await self._on_transcript(direction, msg["text"])
                elif msg_type == "interrupted" and self._on_interrupted:
                    await self._on_interrupted()
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[TableLight] Backend connection closed: {e}")
        except Exception as e:
            import traceback
            print(f"[TableLight] Receive error: {type(e).__name__}: {e}")
            traceback.print_exc()
