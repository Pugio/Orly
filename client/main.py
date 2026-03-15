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
import sys

import cv2
import numpy as np

from client.audio import AudioCapture, AudioPlayer
from client.camera import CameraCapture
from client.display import show_on_projector, show_on_laptop, get_projector_resolution
from client.object_tracker import ObjectTracker
from client.overlay_manager import OverlayManager
from client.overlay_state import OverlayStateManager
from client.program_runtime import ProgramRuntime, TableAPI
from client.session_store import SessionStore
from client.ws_client import TableLightClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async loops (testable with mocks)
# ---------------------------------------------------------------------------


async def video_loop(camera: CameraCapture, client: TableLightClient, fps: float,
                     overlay_manager: OverlayManager = None,
                     program_runtime: ProgramRuntime = None):
    """Capture and send video frames at the specified FPS.

    When overlays are active, sends the last clean frame instead of
    the live view (which would show the projected overlay on the paper).

    When a refresh is requested, waits one frame for the projector to
    go dark, captures a fresh clean frame, then restores overlays.
    """
    interval = 1.0 / fps
    last_clean_frame = None
    refresh_wait_frames = 0  # countdown after refresh request
    loop = asyncio.get_running_loop()
    while True:
        # Run blocking camera capture in executor so it doesn't stall
        # the event loop (audio send/receive must keep flowing).
        jpeg_bytes, raw_frame, _ = await loop.run_in_executor(
            None, camera.get_rectified_frame
        )
        if not jpeg_bytes:
            await asyncio.sleep(interval)
            continue

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
                await asyncio.sleep(interval)
                continue

        # Check if overlays are active on the canvas (cached flag, no array scan).
        has_content = False
        if overlay_manager and not overlay_manager._refresh_requested:
            has_content = overlay_manager._has_content

        if has_content and last_clean_frame:
            await client.send_video(last_clean_frame)
        else:
            last_clean_frame = jpeg_bytes
            if overlay_manager:
                overlay_manager.last_clean_frame = jpeg_bytes
            await client.send_video(jpeg_bytes)

        # Pass raw frame to programs and object tracking (no JPEG decode).
        if program_runtime and raw_frame is not None:
            program_runtime.process_frame(raw_frame)

        await asyncio.sleep(interval)


async def silence_loop(client: TableLightClient):
    """Send silence chunks to keep the audio stream alive (for --no-audio mode)."""
    # 100ms of silence at 16kHz, 16-bit mono = 3200 bytes
    silence = b"\x00" * 3200
    while True:
        await client.send_audio(silence)
        await asyncio.sleep(0.1)


async def audio_send_loop(audio_capture: AudioCapture, client: TableLightClient):
    """Send audio chunks to backend continuously."""
    while True:
        chunk = audio_capture.get_chunk()
        if chunk:
            await client.send_audio(chunk)
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
    return parser.parse_args(argv)


def load_projector_homography(path: str) -> tuple[np.ndarray, int, int]:
    """Load projector homography and resolution from .npz file.

    Returns (H_proj, proj_width, proj_height).
    """
    data = np.load(path)
    H_proj = data["H_proj"]
    proj_width = int(data["proj_width"])
    proj_height = int(data["proj_height"])
    return H_proj, proj_width, proj_height


async def main(args: argparse.Namespace | None = None):
    """Run the TableLight edge client (async tasks only, no OpenCV)."""
    if args is None:
        args = parse_args()

    # --- Camera ---
    camera = CameraCapture(url=args.url, webcam=args.webcam, rotate=args.rotate)
    camera.start()
    # Flush stale frames from the MJPEG buffer so first capture is fresh.
    for _ in range(10):
        camera.get_rectified_frame()
    print("[TableLight] Camera started (buffer flushed).")

    # --- Projector homography ---
    H_proj = None
    proj_width, proj_height = get_projector_resolution()

    if args.h_proj:
        H_proj, proj_width, proj_height = load_projector_homography(args.h_proj)
        print(f"[TableLight] Loaded projector homography from {args.h_proj}")

    # --- Session store ---
    session_store = SessionStore()
    print("[TableLight] Session store ready.")

    # --- WebSocket client (created early so notify_fn can reference it) ---
    client = TableLightClient(args.backend)

    # Notification function: sends async updates to backend (and thus to Gemini).
    _event_loop = asyncio.get_running_loop()

    def _send_notification(text: str):
        """Thread-safe notification sender (called from background threads)."""
        asyncio.run_coroutine_threadsafe(
            client.send_notification("system", text), _event_loop
        )

    # --- Overlay manager ---
    overlay_manager = OverlayManager(
        H_proj=H_proj,
        proj_width=proj_width,
        proj_height=proj_height,
        mode=args.mode,
        image_rotate=args.rotate,
        white_bg=args.white_bg,
        session_store=session_store,
        notify_fn=_send_notification,
    )
    _shared_state["overlay_manager"] = overlay_manager

    # --- Overlay state manager (named overlay tracking) ---
    overlay_state = OverlayStateManager(overlay_manager)
    overlay_manager.overlay_state = overlay_state

    # --- Object tracker ---
    object_tracker = ObjectTracker()

    # --- Program runtime ---
    def _make_table_api():
        """Factory: creates a fresh TableAPI for each program."""
        def _program_notify(msg):
            asyncio.run_coroutine_threadsafe(
                client.send_notification("program", msg), _event_loop
            )
        return TableAPI(
            overlay_state_manager=overlay_state,
            object_tracker=object_tracker,
            session_store=session_store,
            notify_fn=_program_notify,
            get_frame_fn=lambda: program_runtime._latest_frame,
        )

    program_runtime = ProgramRuntime(table_api_factory=_make_table_api)
    program_runtime._object_tracker = object_tracker
    _shared_state["program_runtime"] = program_runtime

    # --- Audio ---
    audio_capture = None
    audio_player = None

    if not args.no_audio:
        audio_capture = AudioCapture()
        audio_capture.start()
        print("[TableLight] Mic capture started.")

        audio_player = AudioPlayer()
        audio_player.start()
        print("[TableLight] Audio playback started.")

    # Register callbacks
    async def on_audio(audio_bytes: bytes):
        if audio_player:
            audio_player.play(audio_bytes)

    async def on_tool_result(name: str, result: dict):
        overlay_manager.handle_tool_result(name, result)
        content_type = result.get("content_type", "unknown")
        title = result.get("title", "")
        # Register in overlay_state so get_overlay_state() reflects tool-created
        # overlays. We pass recomposite=False because handle_tool_result already
        # placed the overlay on canvas — we just need to track the metadata.
        if name == "project_overlay" and result.get("status") == "displayed":
            placement = list(result.get("placement", [0, 0, 1000, 1000]))
            data = result.get("data", {})
            if content_type != "image":  # images register async after generation
                try:
                    adjusted = OverlayManager.adjust_text_placement(
                        content_type, placement)
                    # Use the cached rendered overlay instead of re-rendering.
                    overlay_img = overlay_manager._last_rendered_overlay
                    if overlay_img is not None:
                        overlay_state.add(title or content_type, content_type,
                                          adjusted, title, data, overlay_img,
                                          recomposite=False)
                except Exception:
                    pass
        elif name == "advance_step" and result.get("status") == "advancing":
            overlay_name = result.get("overlay_name", "")
            step_number = result.get("step_number", 1)
            entry = overlay_state.get(overlay_name)
            if entry and entry.content_type == "steps":
                entry.data["visible_count"] = step_number
                new_img = overlay_manager.render_overlay(
                    "steps", entry.placement, entry.title, entry.data)
                overlay_state.add(
                    overlay_name, "steps", entry.placement,
                    entry.title, entry.data, new_img)
        print(f"[TableLight] Overlay projected: {content_type} — {title}")

    _last_direction = {"value": None}

    async def on_transcript(direction: str, text: str):
        if not text or not text.strip():
            return
        # Skip Gemini's internal thinking blocks (markdown bold headers)
        stripped = text.strip()
        if stripped.startswith("**") and stripped.endswith("**"):
            return
        # Print label on direction change, then stream text inline
        if direction != _last_direction["value"]:
            label = "Student" if direction == "in" else "Lumi"
            print(f"\n[{label}] ", end="", flush=True)
            _last_direction["value"] = direction
        print(text, end="", flush=True)

    async def on_interrupted():
        overlay_state.clear()  # clears both state tracking and canvas
        print("[TableLight] Interrupted — overlays cleared.")

    async def on_refresh_view():
        overlay_manager.request_refresh()

    async def on_run_program(name: str, code: str, description: str):
        status = program_runtime.run(name, code, description)
        print(f"[TableLight] Program '{name}' → {status.state}")
        if status.error:
            await client.send_notification("program", f"Program '{name}' error: {status.error}")
        else:
            await client.send_notification("program", f"Program '{name}' started: {description}")

    async def on_stop_program(name: str):
        stopped = program_runtime.stop(name)
        print(f"[TableLight] Stop program '{name}' → {'ok' if stopped else 'not found'}")
        await client.send_notification("program", f"Program '{name}' stopped.")

    async def on_get_overlay_state():
        state = overlay_state.to_json()
        state["ascii_grid"] = overlay_state.to_ascii()
        state["programs"] = [
            {"name": p.name, "state": p.state, "description": p.description}
            for p in program_runtime.list_programs()
        ]
        # Send state back as a notification so Gemini can see it.
        await client.send_notification("overlay_state", json.dumps(state, default=str))

    async def on_list_programs():
        programs = [
            {"name": p.name, "state": p.state, "description": p.description}
            for p in program_runtime.list_programs()
        ]
        await client.send_notification("list_programs", json.dumps(programs, default=str))

    client.on_audio(on_audio)
    client.on_tool_result(on_tool_result)
    client.on_transcript(on_transcript)
    client.on_interrupted(on_interrupted)
    client.on_refresh_view(on_refresh_view)
    client.on_run_program(on_run_program)
    client.on_stop_program(on_stop_program)
    client.on_get_overlay_state(on_get_overlay_state)
    client.on_list_programs(on_list_programs)

    # --- Connect ---
    print(f"[TableLight] Connecting to backend at {args.backend} ...")
    await client.connect(text_only=args.no_audio)
    print(f"[TableLight] Connected ({'text-only' if args.no_audio else 'audio'} mode).")

    # --- Build task list (NO display_loop — OpenCV runs on main thread) ---
    tasks = [
        asyncio.create_task(video_loop(camera, client, args.fps, overlay_manager, program_runtime), name="video"),
        asyncio.create_task(client.receive_loop(), name="receive"),
    ]

    if audio_capture and not args.no_audio:
        tasks.append(
            asyncio.create_task(
                audio_send_loop(audio_capture, client), name="audio_send"
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
        program_runtime.stop_all()
        await client.close()
        camera.stop()
        if audio_capture:
            audio_capture.stop()
        if audio_player:
            audio_player.stop()
        print("[TableLight] Async tasks stopped.")


def run():
    """Entry point: runs asyncio in a background thread, OpenCV on main thread.

    macOS requires OpenCV highgui (window rendering) to run on the main thread.
    """
    args = parse_args()

    # --- Projector verification (must be on main thread) ---
    if args.mode == "projector":
        proj_width, proj_height = get_projector_resolution()
        print("[TableLight] Projector verification — showing test pattern...")
        test_canvas = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)
        cx, cy = proj_width // 2, proj_height // 2
        cv2.circle(test_canvas, (cx, cy), 80, (255, 255, 0), 3)
        cv2.putText(test_canvas, "TableLight", (cx - 100, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        for x, y in [(50, 50), (proj_width - 50, 50),
                      (proj_width - 50, proj_height - 50), (50, proj_height - 50)]:
            cv2.drawMarker(test_canvas, (x, y), (0, 0, 255),
                           cv2.MARKER_CROSS, 30, 2)
        show_on_projector("TableLight Overlay", test_canvas, fullscreen=True)
        cv2.waitKey(1)
        print("[TableLight] Do you see the test pattern on the projector? "
              "(Enter to continue, q to quit)")
        response = input().strip().lower()
        if response == "q":
            cv2.destroyAllWindows()
            print("[TableLight] Aborted.")
            return
        print("[TableLight] Projector verified.")
        # Clear the test pattern immediately so the camera doesn't capture it.
        black = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)
        show_on_projector("TableLight Overlay", black, fullscreen=True)
        cv2.waitKey(500)  # give projector time to go dark

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
    print("[TableLight] Display loop running on main thread. Press Ctrl+C to quit.")

    # Install SIGINT handler that forces exit on second Ctrl+C.
    _interrupt_count = {"n": 0}

    def _sigint_handler(signum, frame):
        _interrupt_count["n"] += 1
        if _interrupt_count["n"] >= 2:
            print("\n[TableLight] Force quit.")
            os._exit(1)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while async_thread.is_alive():
            om = _shared_state.get("overlay_manager")
            if om is not None:
                canvas = om.canvas
                if args.mode == "projector":
                    show_on_projector(win_name, canvas, fullscreen=True)
                else:
                    show_on_laptop(win_name, canvas)
            cv2.waitKey(50)
    except KeyboardInterrupt:
        print("\n[TableLight] Shutting down (Ctrl+C again to force quit)...")
    finally:
        # Best-effort cleanup, then force exit.
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # Cancel async tasks
        for task in asyncio.all_tasks(loop):
            loop.call_soon_threadsafe(task.cancel)
        async_thread.join(timeout=2)
        print("[TableLight] Shutdown complete.")
        # Force exit to avoid hanging on stuck threads (PyAudio, OpenCV).
        os._exit(0)


# Shared state so the main thread can access overlay_manager created in async main()
_shared_state: dict = {}


if __name__ == "__main__":
    run()
