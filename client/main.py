"""
TableLight edge client — connects camera, microphone, projector, and backend.

Usage:
    uv run python -m client.main --backend ws://localhost:8080/ws/session --url http://192.168.0.114:8080

Options:
    --backend URL     Backend WebSocket URL
    --url URL         IP Webcam URL (or --webcam N for local camera)
    --webcam N        Local webcam index
    --h-proj FILE     Projector homography file (enables projector mode)
    --mode MODE       Output mode: "screen" or "projector" (default: screen)
    --fps FLOAT       Video frame rate to send to backend (default: 1.0)
    --no-audio        Disable audio capture/playback (useful for testing)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import struct
import sys
import time

import cv2
import numpy as np

from client.audio import AudioCapture
from client.audio_processor import AudioProcessor
from client.camera import CameraCapture
from client.display import show_on_projector, show_on_laptop, get_projector_resolution
from client.hardware import HardwareConfig, setup_hardware, verify_projector
from client.latency_overlay import composite_debug_overlay, render_latency_overlay
from client.program_runtime import ProgramRuntime
from client.latency_tracker import LatencyTracker
from client.overlay_manager import OverlayManager
from client.music_player import MusicPlayer
from client.ws_client import TableLightClient

logger = logging.getLogger(__name__)

# Threshold for "voice detected" — RMS above this means someone is talking.
# 16-bit PCM silence is ~0, quiet room ~50-200, speech ~500-5000+.
_VOICE_RMS_THRESHOLD = 300


def _ts() -> str:
    """Wall-clock timestamp for diagnostics output (HH:MM:SS.mmm)."""
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}"


class AudioDiagnostics:
    """Tracks audio pipeline health for diagnostics output."""

    def __init__(self):
        self.last_voice_time: float | None = None
        self.last_response_time: float | None = None
        self.last_transcript_in_time: float | None = None
        self.last_transcript_out_time: float | None = None
        self._was_speaking = False
        self._start_time = time.monotonic()
        # Per-interval counters (reset each reporting window)
        self._interval_mic = 0
        self._interval_voice = 0
        self._interval_out = 0
        self._interval_start = time.monotonic()

    def on_mic_chunk(self, chunk: bytes):
        """Called for each mic chunk sent to backend."""
        self._interval_mic += 1
        # Compute RMS to detect voice activity
        if len(chunk) >= 2:
            n_samples = len(chunk) // 2
            samples = struct.unpack(f"<{n_samples}h", chunk[:n_samples * 2])
            rms = (sum(s * s for s in samples) / n_samples) ** 0.5
            if rms > _VOICE_RMS_THRESHOLD:
                self._interval_voice += 1
                now = time.monotonic()
                if not self._was_speaking:
                    self._was_speaking = True
                    print(f"\n[{_ts()}] Voice detected (RMS={rms:.0f})", flush=True)
                self.last_voice_time = now
            else:
                if self._was_speaking:
                    self._was_speaking = False

    def on_audio_out(self):
        """Called for each audio chunk received from backend."""
        self._interval_out += 1
        self.last_response_time = time.monotonic()

    def on_transcript_in(self):
        """Called when input transcription arrives."""
        self.last_transcript_in_time = time.monotonic()

    def on_transcript_out(self):
        """Called when output transcription arrives."""
        self.last_transcript_out_time = time.monotonic()

    def format_status(self) -> str:
        """Format a one-line status summary.

        Shows per-interval rates (chunks/s) and all-time timestamps.
        """
        now = time.monotonic()
        uptime = now - self._start_time
        elapsed = now - self._interval_start if self._interval_start else 1.0
        elapsed = max(elapsed, 0.1)

        mic_rate = self._interval_mic / elapsed
        voice_rate = self._interval_voice / elapsed
        out_rate = self._interval_out / elapsed

        parts = [f"up={uptime:.0f}s"]
        parts.append(f"mic={mic_rate:.0f}/s (voice={voice_rate:.1f}/s)")
        parts.append(f"recv={out_rate:.0f}/s")

        if self.last_voice_time:
            ago = now - self.last_voice_time
            parts.append(f"last_voice={ago:.0f}s ago")
        else:
            parts.append("last_voice=never")

        if self.last_transcript_in_time:
            ago = now - self.last_transcript_in_time
            parts.append(f"last_heard={ago:.0f}s ago")

        if self.last_response_time:
            ago = now - self.last_response_time
            parts.append(f"last_reply={ago:.0f}s ago")

        return " | ".join(parts)

    def reset_interval(self):
        """Reset per-interval counters for the next reporting window."""
        self._interval_mic = 0
        self._interval_voice = 0
        self._interval_out = 0
        self._interval_start = time.monotonic()


async def speaker_state_loop(processor: AudioProcessor, audio_player,
                             poll_interval: float = 0.05):
    """Poll AudioPlayer.is_playing and update the processor's echo state.

    Runs at ~20Hz so echo suppression deactivates promptly when the speaker
    queue drains (rather than waiting for the 15s diagnostics interval).
    """
    was_playing = False
    while True:
        playing = audio_player.is_playing
        if playing != was_playing:
            processor.set_speaker_active(playing)
            was_playing = playing
        await asyncio.sleep(poll_interval)


async def audio_diagnostics_loop(diag: AudioDiagnostics, interval: float = 15.0,
                                  processor: AudioProcessor | None = None):
    """Print periodic audio pipeline status."""
    diag.reset_interval()
    while True:
        await asyncio.sleep(interval)
        status = diag.format_status()
        if processor:
            status += f" | {processor.format_stats()}"
            processor.reset_stats()
        print(f"\n[{_ts()}] {status}", flush=True)
        diag.reset_interval()


# ---------------------------------------------------------------------------
# Async loops (testable with mocks)
# ---------------------------------------------------------------------------


async def video_loop(camera: CameraCapture, client: TableLightClient, fps: float,
                     overlay_manager: OverlayManager = None,
                     program_runtime: ProgramRuntime = None,
                     latency_tracker=None,
                     save_frame: bool = False):
    """Capture and send video frames at the specified FPS.

    When overlays are active, sends the last clean frame instead of
    the live view (which would show the projected overlay on the paper).

    When a refresh is requested, waits one frame for the projector to
    go dark, captures a fresh clean frame, then restores overlays.

    When *save_frame* is True, saves the first sent frame to
    ``gemini_view.jpg`` so you can verify what the agent sees.
    """
    interval = 1.0 / fps
    last_clean_frame = None
    refresh_wait_frames = 0  # countdown after refresh request
    _saved_debug_frame = False
    _prev_frame = None  # previous raw frame for change detection
    _frames_since_send = 0  # force send every N frames even if static
    _DIFF_THRESHOLD = 5.0  # mean absolute pixel diff to count as "changed"
    _MAX_SKIP = 10  # send at least every 10 frames even if unchanged
    loop = asyncio.get_running_loop()
    while True:
        lt = latency_tracker
        if lt:
            lt.begin("video_frame")

        # Run blocking camera capture in executor so it doesn't stall
        # the event loop (audio send/receive must keep flowing).
        jpeg_bytes, raw_frame, _ = await loop.run_in_executor(
            None, camera.get_rectified_frame
        )

        if not jpeg_bytes:
            # No markers detected yet — save raw camera frame for debugging.
            if save_frame and not _saved_debug_frame:
                cap_frame = await loop.run_in_executor(
                    None, lambda: camera.source.read() if camera.source else None
                )
                if cap_frame is not None:
                    _saved_debug_frame = True
                    cv2.imwrite("gemini_view_raw.jpg", cap_frame)
                    print("[TableLight] Raw camera frame saved to gemini_view_raw.jpg (no markers detected)")
            if lt:
                lt.end("video_frame")
            await asyncio.sleep(interval)
            continue

        # Save the post-rotation JPEG — exactly what Gemini will see.
        if save_frame and not _saved_debug_frame:
            _saved_debug_frame = True
            with open("gemini_view.jpg", "wb") as f:
                f.write(jpeg_bytes)
            print(f"[TableLight] Debug frame saved to gemini_view.jpg (rotate={camera.rotate})")

        # Handle refresh_view cycle.
        if overlay_manager and overlay_manager._refresh_requested:
            if refresh_wait_frames == 0:
                # First frame after request — projector just went dark,
                # skip this frame (may still show overlay residue).
                refresh_wait_frames = 1
            elif refresh_wait_frames == 1:
                # Second frame — projector is dark, capture is clean.
                last_clean_frame = jpeg_bytes
                overlay_manager.last_clean_frame = jpeg_bytes
                await client.send_video(jpeg_bytes)
                overlay_manager.complete_refresh()
                refresh_wait_frames = 0
                if lt:
                    lt.end("video_frame")
                await asyncio.sleep(interval)
                continue

        # Check if overlays are active on the canvas (cached flag, no array scan).
        has_content = False
        if overlay_manager and not overlay_manager._refresh_requested:
            has_content = overlay_manager._has_content

        if has_content and last_clean_frame:
            # Overlays active — send the cached clean frame, but skip if
            # it's the same one we already sent (no table change).
            frame_to_send = last_clean_frame
        else:
            last_clean_frame = jpeg_bytes
            if overlay_manager:
                overlay_manager.last_clean_frame = jpeg_bytes
            frame_to_send = jpeg_bytes

        # Diff-based skip: only send when the frame has visually changed.
        # This saves ~258 tokens/frame when the table is static.
        _frames_since_send += 1
        should_send = _frames_since_send >= _MAX_SKIP  # force periodic send
        if not should_send and raw_frame is not None:
            if _prev_frame is None:
                should_send = True
            else:
                diff = np.mean(np.abs(
                    raw_frame.astype(np.int16) - _prev_frame.astype(np.int16)
                ))
                if diff > _DIFF_THRESHOLD:
                    should_send = True
        elif raw_frame is None:
            should_send = True  # no raw frame means marker detection issue

        if should_send:
            await client.send_video(frame_to_send)
            _prev_frame = raw_frame.copy() if raw_frame is not None else None
            _frames_since_send = 0

        # Offload tracker/program work to thread pool so it doesn't block
        # the async transport loop (slow tracking shouldn't delay frame sending).
        if program_runtime and raw_frame is not None:
            loop.run_in_executor(None, program_runtime.process_frame, raw_frame)

        if lt:
            lt.end("video_frame")

        await asyncio.sleep(interval)


async def silence_loop(client: TableLightClient):
    """Send silence chunks to keep the audio stream alive (for --no-audio mode)."""
    # 100ms of silence at 16kHz, 16-bit mono = 3200 bytes
    silence = b"\x00" * 3200
    while True:
        await client.send_audio(silence)
        await asyncio.sleep(0.1)


async def audio_send_loop(audio_capture: AudioCapture, client: TableLightClient,
                          diag: AudioDiagnostics | None = None,
                          processor: AudioProcessor | None = None):
    """Send audio chunks to backend continuously.

    When an AudioProcessor is provided, chunks are preprocessed with a
    noise gate and echo suppression before being forwarded.  This
    reduces false VAD triggers on the server while still allowing
    genuine user speech (including interruptions) through.

    Without a processor, all mic audio is forwarded raw — the server-side
    VAD handles everything (legacy behavior).
    """
    while True:
        chunk = audio_capture.get_chunk()
        if chunk:
            if processor is not None:
                chunk = processor.process(chunk)
            await client.send_audio(chunk)
            if diag:
                diag.on_mic_chunk(chunk)
        else:
            await asyncio.sleep(0.01)  # tight poll for low audio latency


async def text_input_loop(client: TableLightClient):
    """Read text from stdin and send to backend (for --no-audio testing)."""
    import threading
    import queue as queue_mod

    input_queue: queue_mod.Queue[str | None] = queue_mod.Queue()

    def _reader():
        try:
            while True:
                line = sys.stdin.readline()
                if not line:  # EOF
                    input_queue.put(None)
                    return
                input_queue.put(line.strip())
        except Exception:
            input_queue.put(None)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    print("[TableLight] Type a message and press Enter to send to Lumi.")
    while True:
        # Poll the queue so asyncio cancellation works
        try:
            line = input_queue.get_nowait()
        except queue_mod.Empty:
            await asyncio.sleep(0.1)
            continue
        if line is None:
            break
        if line:
            await client.send_text(line)
            print(f"[You] {line}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TableLight edge client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Backend WebSocket URL (e.g. ws://localhost:8080/ws/session)",
    )
    parser.add_argument("--url", type=str, default=None, help="IP Webcam URL")
    parser.add_argument(
        "--webcam", type=int, default=None, help="Local webcam index"
    )
    parser.add_argument(
        "--h-proj",
        type=str,
        default=None,
        help="Projector homography .npz file (enables projector mode)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["screen", "projector"],
        default="screen",
        help="Output mode (default: screen)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Video frame rate to send to backend (default: 1.0)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio capture/playback",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate rectified image CW before sending to Gemini (default: 0)",
    )
    parser.add_argument(
        "--white-bg",
        action="store_true",
        help="Use white background instead of black (illuminates paper)",
    )
    parser.add_argument(
        "--debug-latency",
        action="store_true",
        help="Show live latency debug overlay on display",
    )
    parser.add_argument(
        "--save-frame",
        action="store_true",
        help="Save the first sent frame to gemini_view.jpg for inspection",
    )
    return parser.parse_args(argv)


async def main(args: argparse.Namespace | None = None):
    """Run the TableLight edge client (async tasks only, no OpenCV)."""
    logging.basicConfig(level=logging.INFO)
    if args is None:
        args = parse_args()

    # --- WebSocket client (created early so notify_fn can reference it) ---
    latency_tracker = LatencyTracker() if args.debug_latency else None
    _shared_state["latency_tracker"] = latency_tracker

    client = TableLightClient(args.backend, latency_tracker=latency_tracker)

    _event_loop = asyncio.get_running_loop()

    def _send_notification(text: str):
        """Thread-safe notification sender (called from background threads)."""
        asyncio.run_coroutine_threadsafe(
            client.send_notification("system", text), _event_loop
        )

    # --- Hardware stack (camera, overlay, tracker, runtime) ---
    hw_config = HardwareConfig(
        url=args.url,
        webcam=args.webcam,
        rotate=args.rotate,
        h_proj_path=args.h_proj,
        mode=args.mode,
        white_bg=args.white_bg,
        no_audio=args.no_audio,
        debug_latency=args.debug_latency,
        save_frame=getattr(args, "save_frame", False),
    )
    stack = setup_hardware(hw_config, notify_fn=_send_notification,
                           latency_tracker=latency_tracker)

    camera = stack.camera
    overlay_manager = stack.overlay_manager
    overlay_state = stack.overlay_state
    object_tracker = stack.object_tracker
    program_runtime = stack.program_runtime
    session_store = stack.session_store

    _shared_state["overlay_manager"] = overlay_manager
    _shared_state["program_runtime"] = program_runtime

    # --- Music player ---
    music_player = MusicPlayer(
        session_store=session_store,
        notify_fn=_send_notification,
    )

    # --- Audio (mic capture is agent-only; player comes from stack) ---
    audio_capture = None
    audio_player = stack.audio_player
    audio_diag = AudioDiagnostics() if not args.no_audio else None
    audio_processor = None

    if not args.no_audio:
        audio_capture = AudioCapture()
        audio_capture.start()
        audio_processor = AudioProcessor()
        print(
            f"[TableLight] Mic capture started "
            f"(noise_gate={audio_processor.noise_gate_rms}, "
            f"echo_gate={audio_processor.echo_gate_rms})."
        )

    # Register callbacks
    async def on_audio(audio_bytes: bytes):
        if audio_player:
            audio_player.play(audio_bytes)
            if audio_processor:
                audio_processor.set_speaker_active(True)
        if audio_diag:
            audio_diag.on_audio_out()

    async def on_tool_result(name: str, result: dict):
        action = result.get("action", "") or result.get("target", "")

        if name == "overlay":
            overlay_manager.handle_tool_result(name, result)
            # Register in overlay_state for create/advance_step/flip_flashcard
            if action == "create" and result.get("status") == "displayed":
                content_type = result.get("content_type", "unknown")
                title = result.get("title", "")
                placement = list(result.get("placement", [0, 0, 1000, 1000]))
                data = result.get("data", {})
                if content_type != "image":  # images register async after generation
                    try:
                        placement = overlay_manager._unrotate_placement(placement)
                        adjusted = OverlayManager.adjust_text_placement(
                            content_type, placement)
                        overlay_img = overlay_manager._last_rendered_overlay
                        if overlay_img is not None:
                            overlay_state.add(title or content_type, content_type,
                                              adjusted, title, data, overlay_img,
                                              recomposite=False)
                    except Exception:
                        pass
                print(f"[TableLight] Overlay projected: {content_type} — {title}")
            elif action == "advance_step" and result.get("status") == "advancing":
                overlay_name = result.get("overlay_name", "")
                step_number = result.get("step_number", 1)
                entry = overlay_state.get(overlay_name)
                if entry and entry.content_type == "steps":
                    entry.data["visible_count"] = step_number
                    new_img = overlay_manager.render_overlay(
                        "steps", entry.placement, entry.title, entry.data)
                    new_img = overlay_manager.transform.orient_overlay(new_img)
                    overlay_state.add(
                        overlay_name, "steps", entry.placement,
                        entry.title, entry.data, new_img)
            elif action == "flip_flashcard" and result.get("status") == "flipping":
                overlay_name = result.get("overlay_name", "")
                entry = overlay_state.get(overlay_name)
                if entry and entry.content_type == "flashcard":
                    entry.data["show_back"] = not entry.data.get("show_back", False)
                    new_img = overlay_manager.render_overlay(
                        "flashcard", entry.placement, entry.title, entry.data)
                    new_img = overlay_manager.transform.orient_overlay(new_img)
                    overlay_state.add(
                        overlay_name, "flashcard", entry.placement,
                        entry.title, entry.data, new_img)

        elif name == "query":
            if action == "fresh_view":
                overlay_manager.request_refresh()
            elif action == "overlay_state":
                state = overlay_state.to_json()
                state["ascii_grid"] = overlay_state.to_ascii()
                state["programs"] = [
                    {"name": p.name, "state": p.state, "description": p.description}
                    for p in program_runtime.list_programs()
                ]
                await client.send_notification("overlay_state", json.dumps(state, default=str))
            elif action == "session_manifest":
                manifest = session_store.get_manifest()
                await client.send_notification(
                    "session_manifest", json.dumps(manifest, default=str)
                )

        elif name == "music":
            if action == "play":
                music_player.play(
                    result.get("name", ""), result.get("prompt", ""),
                    result.get("bpm", 120), result.get("temperature", 1.0),
                    result.get("guidance", 3.0))
                print(f"[TableLight] Music starting: '{result.get('name', '')}'")
            elif action == "stop":
                music_player.stop()
                print("[TableLight] Music stopped.")
            elif action == "pause":
                music_player.pause()
                print("[TableLight] Music paused.")
            elif action == "resume":
                music_player.resume()
                print("[TableLight] Music resumed.")
            elif action == "replay":
                music_player.replay(result.get("name", ""))
                print(f"[TableLight] Replaying music: '{result.get('name', '')}'")

    _last_direction = {"value": None}

    async def on_transcript(direction: str, text: str):
        if not text or not text.strip():
            return
        # Skip Gemini's internal thinking blocks (markdown bold headers)
        stripped = text.strip()
        if stripped.startswith("**") and stripped.endswith("**"):
            return
        # Track in diagnostics
        if audio_diag:
            if direction == "in":
                audio_diag.on_transcript_in()
            else:
                audio_diag.on_transcript_out()
        # Print label on direction change, then stream text inline
        if direction != _last_direction["value"]:
            label = "Student" if direction == "in" else "Lumi"
            print(f"\n[{_ts()} {label}] ", end="", flush=True)
            _last_direction["value"] = direction
        print(text, end="", flush=True)

    async def on_interrupted():
        # Interruptions are normal — the user spoke while the model was
        # talking.  Do NOT clear overlays; they should persist until
        # explicitly removed by a tool call.
        # Clear queued audio so mic gating ends immediately — the user
        # is speaking and we need to hear them, not play stale audio.
        print(f"\n[{_ts()}] INTERRUPTED", flush=True)
        if audio_player:
            audio_player.clear()
        if audio_processor:
            audio_processor.set_speaker_active(False)

    client.on_audio(on_audio)
    client.on_tool_result(on_tool_result)
    client.on_transcript(on_transcript)
    client.on_interrupted(on_interrupted)

    # --- Connect ---
    print(f"[TableLight] Connecting to backend at {args.backend} ...")
    await client.connect(text_only=args.no_audio)
    print(f"[TableLight] Connected ({'text-only' if args.no_audio else 'audio'} mode).")

    # --- Build task list (NO display_loop — OpenCV runs on main thread) ---
    tasks = [
        asyncio.create_task(video_loop(camera, client, args.fps, overlay_manager, program_runtime, latency_tracker=latency_tracker, save_frame=args.save_frame), name="video"),
        asyncio.create_task(client.receive_loop(), name="receive"),
    ]

    if audio_capture and not args.no_audio:
        tasks.append(
            asyncio.create_task(
                audio_send_loop(audio_capture, client, diag=audio_diag,
                                processor=audio_processor),
                name="audio_send",
            )
        )
        if audio_processor and audio_player:
            tasks.append(
                asyncio.create_task(
                    speaker_state_loop(audio_processor, audio_player),
                    name="speaker_state",
                )
            )
        if audio_diag:
            tasks.append(
                asyncio.create_task(
                    audio_diagnostics_loop(audio_diag, processor=audio_processor),
                    name="audio_diag",
                )
            )
    else:
        # In no-audio mode, send silence to keep the audio stream alive
        # and allow text input from terminal.
        tasks.append(
            asyncio.create_task(silence_loop(client), name="silence")
        )
        tasks.append(
            asyncio.create_task(text_input_loop(client), name="text_input")
        )

    # Let all tasks run; they'll be cancelled when the main thread stops
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        stack.stop()
        await client.close()
        if audio_capture:
            audio_capture.stop()
        print("[TableLight] Async tasks stopped.")


def run():
    """Entry point: runs asyncio in a background thread, OpenCV on main thread.

    macOS requires OpenCV highgui (window rendering) to run on the main thread.
    """
    args = parse_args()

    # --- Projector verification (must be on main thread) ---
    if args.mode == "projector":
        proj_width, proj_height = get_projector_resolution()
        if not verify_projector(proj_width, proj_height):
            print("[TableLight] Aborted.")
            return

    # Start the async event loop in a background thread
    loop = asyncio.new_event_loop()
    main_coro = main(args)

    import threading
    async_thread = threading.Thread(
        target=lambda: loop.run_until_complete(main_coro),
        daemon=True,
    )
    async_thread.start()

    # Main thread: OpenCV display loop
    # Wait briefly for async setup to initialize overlay_manager
    import time
    time.sleep(1)

    win_name = "TableLight Overlay"

    def _alive():
        return async_thread.is_alive()

    def _display_tick():
        """Called each display iteration — handles latency overlay."""
        om = _shared_state.get("overlay_manager")
        if om is None:
            cv2.waitKey(50)
            return
        canvas = om.canvas.copy() if args.debug_latency else om.canvas
        lt = _shared_state.get("latency_tracker")
        if lt and args.debug_latency:
            overlay = render_latency_overlay(lt, width=400, height=160)
            composite_debug_overlay(canvas, overlay, x=10, y=10)
            lt.log_stats_periodic(every_n=60)
        if args.mode == "projector":
            show_on_projector(win_name, canvas, fullscreen=True)
        else:
            show_on_laptop(win_name, canvas)

    # Install SIGINT handler: second Ctrl+C force-quits.
    _interrupt_count = {"n": 0}

    def _sigint_handler(signum, frame):
        _interrupt_count["n"] += 1
        if _interrupt_count["n"] >= 2:
            print("\n[TableLight] Force quit.")
            os._exit(1)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    print("[TableLight] Display loop running on main thread. Press Ctrl+C to quit.")
    try:
        while _alive():
            _display_tick()
            cv2.waitKey(50)
    except KeyboardInterrupt:
        print("\n[TableLight] Shutting down (Ctrl+C again to force quit)...")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        for task in asyncio.all_tasks(loop):
            loop.call_soon_threadsafe(task.cancel)
        async_thread.join(timeout=2)
        print("[TableLight] Shutdown complete.")
        os._exit(0)


# Shared state so the main thread can access overlay_manager created in async main()
_shared_state: dict = {}


if __name__ == "__main__":
    run()
