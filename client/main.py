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
    """
    interval = 1.0 / fps
    frame_count = 0
    last_clean_frame = None
    while True:
        jpeg_bytes, _ = camera.get_rectified_frame()
        if jpeg_bytes:
            has_overlay = overlay_manager and np.any(overlay_manager.canvas > 0)
            if has_overlay and last_clean_frame:
                # Send cached clean frame so Gemini doesn't see its own overlay
                await client.send_video(last_clean_frame)
            else:
                last_clean_frame = jpeg_bytes
                await client.send_video(jpeg_bytes)
            frame_count += 1
            # Save first few frames for debugging
            if frame_count <= 3:
                try:
                    import os
                    path = os.path.join(os.getcwd(), f"debug_sent_frame_{frame_count}.jpg")
                    with open(path, "wb") as f:
                        f.write(jpeg_bytes)
                    print(f"[TableLight] Saved frame to {path} ({len(jpeg_bytes)} bytes)")
                except Exception as e:
                    print(f"[TableLight] Failed to save frame: {e}")
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
            await asyncio.sleep(0.01)  # avoid busy-wait


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
    frame_count = 0
    while True:
        canvas = overlay_manager.canvas
        has_content = np.any(canvas > 0)
        if mode == "projector":
            show_on_projector(win_name, canvas, fullscreen=True)
        else:
            show_on_laptop(win_name, canvas)
        cv2.waitKey(1)
        frame_count += 1
        if has_content and frame_count % 50 == 0:
            print(f"[Display] Frame {frame_count}, canvas non-black: {np.count_nonzero(canvas)}")
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
    print("[TableLight] Camera started.")

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

    async def on_transcript(direction: str, text: str):
        if not text or not text.strip():
            return
        # Skip Gemini's internal thinking blocks (markdown bold headers)
        stripped = text.strip()
        if stripped.startswith("**") and stripped.endswith("**"):
            return
        if direction == "in":
            print(f"[Student] {text}")
        else:
            print(f"[Lumi] {text}")

    async def on_interrupted():
        overlay_manager.clear()
        print("[TableLight] Interrupted — overlays cleared.")

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

    frame_n = 0
    try:
        while async_thread.is_alive():
            om = _shared_state.get("overlay_manager")
            if om is not None:
                canvas = om.canvas
                if args.mode == "projector":
                    show_on_projector(win_name, canvas, fullscreen=True)
                else:
                    show_on_laptop(win_name, canvas)
                frame_n += 1
                if frame_n == 1:
                    print("[Display] First frame sent to projector.")
                has_content = np.any(canvas > 0)
                if has_content and frame_n % 20 == 0:
                    print(f"[Display] Frame {frame_n}, non-black: {np.count_nonzero(canvas)}, "
                          f"max: {canvas.max()}, shape: {canvas.shape}")
                    cv2.imwrite("debug_main_thread_canvas.png", canvas)
            else:
                if frame_n == 0:
                    print("[Display] Waiting for overlay_manager...")
            cv2.waitKey(50)
    except KeyboardInterrupt:
        print("\n[TableLight] Shutting down...")
    finally:
        cv2.destroyAllWindows()
        # Cancel async tasks
        for task in asyncio.all_tasks(loop):
            loop.call_soon_threadsafe(task.cancel)
        async_thread.join(timeout=3)
        print("[TableLight] Shutdown complete.")


# Shared state so the main thread can access overlay_manager created in async main()
_shared_state: dict = {}


if __name__ == "__main__":
    run()
