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
import logging
import os
import signal
import sys

import cv2
import numpy as np

from client.audio import AudioCapture, AudioPlayer
from client.camera import CameraCapture
from client.display import show_on_projector, show_on_laptop, get_projector_resolution
from client.overlay_manager import OverlayManager
from client.ws_client import TableLightClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async loops (testable with mocks)
# ---------------------------------------------------------------------------


async def video_loop(camera: CameraCapture, client: TableLightClient, fps: float,
                     overlay_manager: OverlayManager = None):
    """Capture and send video frames at the specified FPS.

    When overlays are active, sends the last clean frame instead of
    the live view (which would show the projected overlay on the paper).

    When a refresh is requested, waits one frame for the projector to
    go dark, captures a fresh clean frame, then restores overlays.
    """
    interval = 1.0 / fps
    last_clean_frame = None
    refresh_wait_frames = 0  # countdown after refresh request
    loop = asyncio.get_event_loop()
    while True:
        # Run blocking camera capture in executor so it doesn't stall
        # the event loop (audio send/receive must keep flowing).
        jpeg_bytes, _ = await loop.run_in_executor(
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

        # Check if overlays are active on the canvas.
        has_content = False
        if overlay_manager and not overlay_manager._refresh_requested:
            if overlay_manager.white_bg:
                has_content = not np.all(overlay_manager.canvas == 255)
            else:
                has_content = np.any(overlay_manager.canvas > 0)

        if has_content and last_clean_frame:
            await client.send_video(last_clean_frame)
        else:
            last_clean_frame = jpeg_bytes
            if overlay_manager:
                overlay_manager.last_clean_frame = jpeg_bytes
            await client.send_video(jpeg_bytes)
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


async def display_loop(overlay_manager: OverlayManager, mode: str):
    """Update the projector/screen overlay display."""
    win_name = "TableLight Overlay"
    while True:
        canvas = overlay_manager.canvas
        if mode == "projector":
            show_on_projector(win_name, canvas, fullscreen=True)
        else:
            show_on_laptop(win_name, canvas)
        cv2.waitKey(1)
        await asyncio.sleep(0.05)  # ~20fps, yield to event loop


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

    # --- Overlay manager ---
    overlay_manager = OverlayManager(
        H_proj=H_proj,
        proj_width=proj_width,
        proj_height=proj_height,
        mode=args.mode,
        image_rotate=args.rotate,
        white_bg=args.white_bg,
    )
    _shared_state["overlay_manager"] = overlay_manager

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

    # --- WebSocket client ---
    client = TableLightClient(args.backend)

    # Register callbacks
    async def on_audio(audio_bytes: bytes):
        if audio_player:
            audio_player.play(audio_bytes)

    async def on_tool_result(name: str, result: dict):
        overlay_manager.handle_tool_result(name, result)
        content_type = result.get("content_type", "unknown")
        title = result.get("title", "")
        print(f"[TableLight] Overlay projected: {content_type} — {title}")
        # Canvas is updated — main thread display loop will pick it up

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
        overlay_manager.clear()
        print("[TableLight] Interrupted — overlays cleared.")

    async def on_refresh_view():
        overlay_manager.request_refresh()

    client.on_audio(on_audio)
    client.on_tool_result(on_tool_result)
    client.on_transcript(on_transcript)
    client.on_interrupted(on_interrupted)
    client.on_refresh_view(on_refresh_view)

    # --- Connect ---
    print(f"[TableLight] Connecting to backend at {args.backend} ...")
    await client.connect(text_only=args.no_audio)
    print(f"[TableLight] Connected ({'text-only' if args.no_audio else 'audio'} mode).")

    # --- Build task list (NO display_loop — OpenCV runs on main thread) ---
    tasks = [
        asyncio.create_task(video_loop(camera, client, args.fps, overlay_manager), name="video"),
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
    _original_sigint = signal.getsignal(signal.SIGINT)

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
