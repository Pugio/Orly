"""Shared hardware + overlay stack setup for both full agent and standalone modes.

Encapsulates camera, projector, audio, overlay manager, object tracker, and
program runtime — everything needed to run mini-programs on the physical table.

Used by:
  - client.main (full agent with Gemini backend)
  - client.run_program (standalone program runner, no backend)
"""

from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass, field

import cv2
import numpy as np

from client.camera import CameraCapture
from client.display import show_on_projector, show_on_laptop, get_projector_resolution
from client.object_tracker import ObjectTracker
from client.overlay_manager import OverlayManager
from client.overlay_state import OverlayStateManager
from client.program_runtime import ProgramRuntime, TableAPI
from client.session_store import SessionStore

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    """Configuration for the hardware stack."""
    # Camera
    url: str | None = None
    webcam: int | None = None
    rotate: int = 0

    # Projector
    h_proj_path: str | None = None
    mode: str = "screen"       # "screen" or "projector"
    white_bg: bool = False

    # Audio
    no_audio: bool = False

    # Debug
    debug_latency: bool = False
    save_frame: bool = False


@dataclass
class HardwareStack:
    """All initialised hardware and overlay components.

    Created by setup_hardware(). Callers access the components they need.
    """
    camera: CameraCapture
    overlay_manager: OverlayManager
    overlay_state: OverlayStateManager
    object_tracker: ObjectTracker
    program_runtime: ProgramRuntime
    session_store: SessionStore
    audio_player: object | None = None       # AudioPlayer or None
    latency_tracker: object | None = None    # LatencyTracker or None
    H_proj: np.ndarray | None = None
    proj_width: int = 1280
    proj_height: int = 720
    mode: str = "screen"

    def stop(self):
        """Clean up all hardware resources."""
        self.program_runtime.stop_all()
        self.camera.stop()
        if self.audio_player is not None:
            self.audio_player.stop()


def load_projector_homography(path: str) -> tuple[np.ndarray, int, int]:
    """Load projector homography and resolution from .npz file."""
    data = np.load(path)
    return data["H_proj"], int(data["proj_width"]), int(data["proj_height"])


def setup_hardware(
    config: HardwareConfig,
    notify_fn=None,
    latency_tracker=None,
) -> HardwareStack:
    """Set up the full hardware + overlay stack.

    Args:
        config: Hardware configuration.
        notify_fn: Callable[[str], None] for sending notifications back to agent.
                   If None, notifications are printed to stdout.
        latency_tracker: Optional LatencyTracker instance.

    Returns:
        HardwareStack with all components initialised and wired together.
    """
    if notify_fn is None:
        notify_fn = lambda msg: print(f"[Notify] {msg}")

    # --- Camera ---
    webcam = config.webcam
    if config.url is None and webcam is None:
        webcam = 0  # default to first webcam

    camera = CameraCapture(
        url=config.url, webcam=webcam, rotate=config.rotate,
        latency_tracker=latency_tracker,
        calibration_file="session/camera_homography.npy",
    )
    camera.start()
    # Flush stale frames
    for _ in range(10):
        camera.get_rectified_frame()
    print("[Hardware] Camera started (buffer flushed).")

    # --- Projector homography ---
    H_proj = None
    proj_width, proj_height = get_projector_resolution()

    if config.h_proj_path:
        H_proj, proj_width, proj_height = load_projector_homography(config.h_proj_path)
        print(f"[Hardware] Loaded projector homography from {config.h_proj_path}")

    # --- Session store ---
    session_store = SessionStore()
    print("[Hardware] Session store ready.")

    # --- Audio player ---
    audio_player = None
    if not config.no_audio:
        try:
            from client.audio import AudioPlayer
            audio_player = AudioPlayer()
            audio_player.start()
            print("[Hardware] Audio player started.")
        except Exception as e:
            print(f"[Hardware] Audio player not available: {e}")

    # --- Overlay manager ---
    overlay_manager = OverlayManager(
        H_proj=H_proj,
        proj_width=proj_width,
        proj_height=proj_height,
        mode=config.mode,
        image_rotate=config.rotate,
        white_bg=config.white_bg,
        session_store=session_store,
        notify_fn=notify_fn,
    )

    # --- Overlay state ---
    overlay_state = OverlayStateManager(overlay_manager)
    overlay_manager.overlay_state = overlay_state

    # --- Object tracker ---
    object_tracker = ObjectTracker()

    # --- Program runtime ---
    def _make_table_api():
        return TableAPI(
            overlay_state_manager=overlay_state,
            object_tracker=object_tracker,
            session_store=session_store,
            notify_fn=notify_fn,
            get_frame_fn=lambda: program_runtime._latest_frame,
            audio_player=audio_player,
        )

    program_runtime = ProgramRuntime(table_api_factory=_make_table_api)
    program_runtime._object_tracker = object_tracker

    return HardwareStack(
        camera=camera,
        overlay_manager=overlay_manager,
        overlay_state=overlay_state,
        object_tracker=object_tracker,
        program_runtime=program_runtime,
        session_store=session_store,
        audio_player=audio_player,
        latency_tracker=latency_tracker,
        H_proj=H_proj,
        proj_width=proj_width,
        proj_height=proj_height,
        mode=config.mode,
    )


def verify_projector(proj_width: int, proj_height: int) -> bool:
    """Show test pattern on projector. Returns True if user confirms.

    Must be called on the main thread (macOS OpenCV requirement).
    """
    print("[Hardware] Projector verification — showing test pattern...")
    test_canvas = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)
    cx, cy = proj_width // 2, proj_height // 2
    cv2.circle(test_canvas, (cx, cy), 80, (255, 255, 0), 3)
    cv2.putText(test_canvas, "Orly", (cx - 100, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    for x, y in [(50, 50), (proj_width - 50, 50),
                  (proj_width - 50, proj_height - 50), (50, proj_height - 50)]:
        cv2.drawMarker(test_canvas, (x, y), (0, 0, 255),
                       cv2.MARKER_CROSS, 30, 2)
    show_on_projector("Orly Overlay", test_canvas, fullscreen=True)
    cv2.waitKey(1)
    print("[Hardware] Do you see the test pattern on the projector? "
          "(Enter to continue, q to quit)")
    response = input().strip().lower()
    if response == "q":
        cv2.destroyAllWindows()
        return False
    print("[Hardware] Projector verified.")
    # Clear so the camera doesn't capture the test pattern
    black = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)
    show_on_projector("Orly Overlay", black, fullscreen=True)
    cv2.waitKey(500)
    return True


def display_loop(
    stack: HardwareStack,
    win_name: str = "Orly",
    feed_frames: bool = True,
    alive_check=None,
):
    """Run the main-thread OpenCV display loop.

    Args:
        stack: The initialised hardware stack.
        win_name: Window title.
        feed_frames: If True, capture camera frames and feed them to
                     program_runtime.process_frame() each iteration.
                     Set to False if frames are fed from another loop
                     (e.g. the async video_loop in main.py).
        alive_check: Optional callable returning bool. Loop exits when it
                     returns False. If None, loops until Ctrl+C or 'q'.

    Returns when the loop is terminated.
    """
    # Install SIGINT handler: second Ctrl+C force-quits.
    _interrupt_count = {"n": 0}

    def _sigint_handler(signum, frame):
        _interrupt_count["n"] += 1
        if _interrupt_count["n"] >= 2:
            print(f"\n[{win_name}] Force quit.")
            os._exit(1)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    print(f"[{win_name}] Display loop running. Press 'q' or Ctrl+C to quit.")

    try:
        while True:
            # Check external alive condition
            if alive_check is not None and not alive_check():
                break

            # Feed camera frames to runtime if requested
            if feed_frames:
                _, raw_frame, _ = stack.camera.get_rectified_frame()
                if raw_frame is not None:
                    stack.program_runtime.process_frame(raw_frame)

            # Render
            canvas = stack.overlay_manager.canvas
            if stack.latency_tracker and hasattr(stack, '_debug_latency') and stack._debug_latency:
                canvas = canvas.copy()
                from client.latency_overlay import render_latency_overlay, composite_debug_overlay
                overlay = render_latency_overlay(stack.latency_tracker, width=400, height=160)
                composite_debug_overlay(canvas, overlay, x=10, y=10)

            if stack.mode == "projector":
                show_on_projector(win_name, canvas, fullscreen=True)
            else:
                show_on_laptop(win_name, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                print(f"[{win_name}] Quit requested.")
                break

    except KeyboardInterrupt:
        print(f"\n[{win_name}] Shutting down...")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
