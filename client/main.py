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


async def video_loop(camera: CameraCapture, client: TableLightClient, fps: float):
    """Capture and send video frames at the specified FPS."""
    interval = 1.0 / fps
    while True:
        jpeg_bytes, _ = camera.get_rectified_frame()
        if jpeg_bytes:
            await client.send_video(jpeg_bytes)
        await asyncio.sleep(interval)


async def audio_send_loop(audio_capture: AudioCapture, client: TableLightClient):
    """Send audio chunks to backend continuously."""
    while True:
        chunk = audio_capture.get_chunk()
        if chunk:
            await client.send_audio(chunk)
        else:
            await asyncio.sleep(0.01)  # avoid busy-wait


async def display_loop(overlay_manager: OverlayManager, mode: str, fps: float = 30.0):
    """Update the projector/screen overlay display."""
    interval = 1.0 / fps
    win_name = "TableLight Overlay"
    while True:
        canvas = overlay_manager.canvas
        if mode == "projector":
            show_on_projector(win_name, canvas, fullscreen=True)
        else:
            show_on_laptop(win_name, canvas)
        cv2.waitKey(1)
        await asyncio.sleep(interval)


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


async def main(argv: list[str] | None = None):
    """Run the TableLight edge client."""
    args = parse_args(argv)

    # --- Camera ---
    camera = CameraCapture(url=args.url, webcam=args.webcam)
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
    )

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

    async def on_transcript(direction: str, text: str):
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
    await client.connect()
    print("[TableLight] Connected.")

    # --- Build task list ---
    tasks = [
        asyncio.create_task(video_loop(camera, client, args.fps), name="video"),
        asyncio.create_task(client.receive_loop(), name="receive"),
        asyncio.create_task(
            display_loop(overlay_manager, args.mode), name="display"
        ),
    ]

    if audio_capture and not args.no_audio:
        tasks.append(
            asyncio.create_task(
                audio_send_loop(audio_capture, client), name="audio_send"
            )
        )

    # --- Handle graceful shutdown ---
    stop_event = asyncio.Event()

    def _signal_handler():
        print("\n[TableLight] Shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Wait until stop signal or a task crashes
    done, pending = await asyncio.wait(
        [asyncio.create_task(stop_event.wait(), name="stop"), *tasks],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel remaining tasks
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    # --- Cleanup ---
    await client.close()
    camera.stop()
    if audio_capture:
        audio_capture.stop()
    if audio_player:
        audio_player.stop()
    cv2.destroyAllWindows()
    print("[TableLight] Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
