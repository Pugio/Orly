"""Simulated edge client — connects to the real backend without hardware.

Drop-in replacement for the physical client. Sends synthetic audio/video,
records all responses with timestamps, and measures latency.

Uses binary WebSocket protocol for audio/video (1-byte prefix + raw payload)
and JSON text frames for text and control messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import websockets

from simulation.fake_audio import (
    CHUNK_SAMPLES,
    chunk_audio,
    generate_silence,
    generate_sine,
    tts_to_pcm,
)
from simulation.fake_camera import generate_test_jpeg
from simulation.scenarios import Scenario, ScenarioStep

logger = logging.getLogger(__name__)

# Binary protocol prefixes (must match backend/main.py).
PREFIX_AUDIO_IN = b"\x01"
PREFIX_VIDEO_IN = b"\x02"
PREFIX_AUDIO_OUT = b"\x03"


@dataclass
class ResponseEvent:
    """A single event received from the backend."""

    timestamp: float  # time.monotonic()
    event_type: str  # "audio", "transcript_in", "transcript_out", "tool_result", "interrupted"
    data: dict | str | bytes = ""


@dataclass
class ScenarioResult:
    """Results from running a single scenario."""

    scenario_name: str
    events: list[ResponseEvent] = field(default_factory=list)
    first_send_time: float = 0.0  # when the first meaningful payload was sent
    last_send_time: float = 0.0  # when the last audio/text chunk was sent
    first_transcript_in_time: float | None = None
    first_transcript_out_time: float | None = None
    first_audio_response_time: float | None = None
    first_tool_result_time: float | None = None
    error: str | None = None

    @property
    def send_to_transcript_in_ms(self) -> float | None:
        """Latency from last send to first input transcription."""
        if self.first_transcript_in_time and self.last_send_time:
            return (self.first_transcript_in_time - self.last_send_time) * 1000
        return None

    @property
    def send_to_transcript_out_ms(self) -> float | None:
        """Latency from last send to first output transcription."""
        if self.first_transcript_out_time and self.last_send_time:
            return (self.first_transcript_out_time - self.last_send_time) * 1000
        return None

    @property
    def send_to_audio_response_ms(self) -> float | None:
        """Latency from last send to first audio response."""
        if self.first_audio_response_time and self.last_send_time:
            return (self.first_audio_response_time - self.last_send_time) * 1000
        return None

    @property
    def send_to_tool_result_ms(self) -> float | None:
        """Latency from last send to first tool result."""
        if self.first_tool_result_time and self.last_send_time:
            return (self.first_tool_result_time - self.last_send_time) * 1000
        return None

    @property
    def total_round_trip_ms(self) -> float | None:
        """Total round-trip: send to first meaningful response (transcript_out or audio)."""
        candidates = [
            t
            for t in [self.first_transcript_out_time, self.first_audio_response_time]
            if t is not None
        ]
        if candidates and self.last_send_time:
            return (min(candidates) - self.last_send_time) * 1000
        return None


class SimClient:
    """Simulated edge client that talks to the real backend WebSocket."""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.ws = None

    async def connect(self):
        """Connect and send init message."""
        self.ws = await websockets.connect(
            self.backend_url,
            max_size=10 * 1024 * 1024,  # 10MB for large audio responses
        )
        # Send init message (always JSON text).
        await self.ws.send(json.dumps({"text_only": False}))
        logger.info("SimClient connected to %s", self.backend_url)

    async def close(self):
        """Close connection."""
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass
            await self.ws.close()
            self.ws = None

    async def send_audio(self, pcm_bytes: bytes):
        """Send a single audio chunk as binary frame."""
        await self.ws.send(PREFIX_AUDIO_IN + pcm_bytes)

    async def send_video(self, jpeg_bytes: bytes):
        """Send a video frame as binary frame."""
        await self.ws.send(PREFIX_VIDEO_IN + jpeg_bytes)

    async def send_text(self, text: str):
        """Send a text message as JSON text frame."""
        await self.ws.send(json.dumps({"type": "text", "text": text}))

    async def _receive_events(
        self,
        result: ScenarioResult,
        done_event: asyncio.Event,
        timeout: float,
    ):
        """Receive events from backend until done_event is set or timeout."""
        deadline = time.monotonic() + timeout
        try:
            while not done_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(self.ws.recv(), timeout=min(remaining, 1.0))
                except asyncio.TimeoutError:
                    continue

                now = time.monotonic()

                # Binary frame: audio response
                if isinstance(raw, bytes):
                    if len(raw) >= 1 and raw[0:1] == PREFIX_AUDIO_OUT:
                        audio_payload = raw[1:]
                        event = ResponseEvent(
                            timestamp=now,
                            event_type="audio",
                            data=f"{len(audio_payload)} bytes",
                        )
                        result.events.append(event)
                        if result.first_audio_response_time is None:
                            result.first_audio_response_time = now
                    continue

                # Text frame: JSON message
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "transcript_in":
                    text = msg.get("text", "")
                    event = ResponseEvent(timestamp=now, event_type="transcript_in", data=text)
                    result.events.append(event)
                    if result.first_transcript_in_time is None:
                        result.first_transcript_in_time = now
                    logger.info("  [IN] %s", text)

                elif msg_type == "transcript_out":
                    text = msg.get("text", "")
                    event = ResponseEvent(timestamp=now, event_type="transcript_out", data=text)
                    result.events.append(event)
                    if result.first_transcript_out_time is None:
                        result.first_transcript_out_time = now
                    logger.info("  [OUT] %s", text)

                elif msg_type == "tool_result":
                    event = ResponseEvent(
                        timestamp=now,
                        event_type="tool_result",
                        data=msg.get("result", {}),
                    )
                    result.events.append(event)
                    if result.first_tool_result_time is None:
                        result.first_tool_result_time = now
                    logger.info("  [TOOL] %s: %s", msg.get("name"), msg.get("result"))

                elif msg_type == "interrupted":
                    event = ResponseEvent(timestamp=now, event_type="interrupted")
                    result.events.append(event)
                    logger.info("  [INTERRUPTED]")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed during receive")
        except Exception as e:
            logger.exception("Receive error: %s", e)

    async def _execute_step(self, step: ScenarioStep, result: ScenarioResult):
        """Execute a single scenario step."""
        if step.action == "send_silence":
            pcm = generate_silence(step.duration_s)
            chunks = chunk_audio(pcm)
            interval = CHUNK_SAMPLES / 16000  # real-time pacing
            for chunk in chunks:
                await self.send_audio(chunk)
                await asyncio.sleep(interval)
            result.last_send_time = time.monotonic()

        elif step.action == "send_audio":
            # Use TTS if text is given, otherwise sine wave.
            if step.text:
                pcm = tts_to_pcm(step.text)
            else:
                pcm = generate_sine(440.0, step.duration_s)
            chunks = chunk_audio(pcm)
            interval = CHUNK_SAMPLES / 16000
            for chunk in chunks:
                await self.send_audio(chunk)
                await asyncio.sleep(interval)
            result.last_send_time = time.monotonic()
            if result.first_send_time == 0.0:
                result.first_send_time = result.last_send_time

        elif step.action == "send_video":
            if step.image_path:
                from simulation.fake_camera import load_image_as_jpeg

                jpeg = load_image_as_jpeg(step.image_path)
            else:
                jpeg = generate_test_jpeg(text_lines=step.video_text_lines or None)
            await self.send_video(jpeg)
            logger.info("  Sent video frame (%d bytes)", len(jpeg))

        elif step.action == "send_text":
            await self.send_text(step.text)
            result.last_send_time = time.monotonic()
            if result.first_send_time == 0.0:
                result.first_send_time = result.last_send_time
            logger.info("  Sent text: %s", step.text)

        elif step.action == "wait":
            await asyncio.sleep(step.duration_s)

    async def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        """Execute a scenario and collect results.

        The client must already be connected (call connect() first).
        """
        logger.info("Running scenario: %s", scenario.name)
        result = ScenarioResult(scenario_name=scenario.name)
        done_event = asyncio.Event()

        # Start receiver in background.
        receiver = asyncio.create_task(
            self._receive_events(result, done_event, scenario.max_wait_s + 60)
        )

        try:
            for step in scenario.steps:
                logger.info("  Step: %s (%.1fs)", step.action, step.duration_s)
                await self._execute_step(step, result)

            # Wait for responses up to max_wait_s after all steps.
            logger.info("  Waiting %.1fs for remaining responses...", scenario.max_wait_s)
            await asyncio.sleep(scenario.max_wait_s)

        except Exception as e:
            result.error = str(e)
            logger.exception("Scenario error: %s", e)
        finally:
            done_event.set()
            try:
                await asyncio.wait_for(receiver, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                receiver.cancel()

        return result


def format_latency(ms: float | None) -> str:
    """Format a latency value for display."""
    if ms is None:
        return "N/A"
    return f"{ms:.0f}ms"


def print_result(result: ScenarioResult):
    """Print a human-readable summary of a scenario result."""
    print(f"\n{'=' * 60}")
    print(f"Scenario: {result.scenario_name}")
    print(f"{'=' * 60}")

    if result.error:
        print(f"  ERROR: {result.error}")

    # Event summary
    type_counts: dict[str, int] = {}
    for ev in result.events:
        type_counts[ev.event_type] = type_counts.get(ev.event_type, 0) + 1
    print(f"  Events received: {sum(type_counts.values())}")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # Latency
    print(f"  Latencies:")
    print(f"    Send -> Input transcript:  {format_latency(result.send_to_transcript_in_ms)}")
    print(f"    Send -> Output transcript: {format_latency(result.send_to_transcript_out_ms)}")
    print(f"    Send -> Audio response:    {format_latency(result.send_to_audio_response_ms)}")
    print(f"    Send -> Tool result:       {format_latency(result.send_to_tool_result_ms)}")
    print(f"    Total round-trip:          {format_latency(result.total_round_trip_ms)}")

    # Transcripts
    transcripts_out = [ev.data for ev in result.events if ev.event_type == "transcript_out"]
    if transcripts_out:
        print(f"  Response text: {''.join(str(t) for t in transcripts_out[:5])}")
        if len(transcripts_out) > 5:
            print(f"    ... ({len(transcripts_out) - 5} more)")
