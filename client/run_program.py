"""Standalone program runner — runs a TableAPI mini-program without the backend.

Sets up the full hardware stack (camera, overlay, tracker, runtime) and executes
a program file directly. No Gemini API, no WebSocket, no backend needed.

Usage:
    uv run python -m client.run_program programs/paint.py [options]

Options:
    --url URL         IP Webcam URL (e.g. http://192.168.0.114:8080)
    --webcam N        Local webcam index (default: 0)
    --h-proj FILE     Projector homography .npz file (enables projector mode)
    --mode MODE       "screen" or "projector" (default: screen)
    --rotate N        Rotate rectified image CW: 0, 90, 180, 270
    --white-bg        Use white background instead of black
    --no-audio        Disable audio playback
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from client.hardware import HardwareConfig, setup_hardware, verify_projector, display_loop


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a TableAPI program standalone (no backend required)")
    parser.add_argument("program", help="Path to the Python program file")
    parser.add_argument("--url", help="IP Webcam URL")
    parser.add_argument("--webcam", type=int, default=None,
                        help="Local webcam index")
    parser.add_argument("--h-proj", help="Projector homography .npz file")
    parser.add_argument("--mode", default="screen",
                        choices=["screen", "projector"])
    parser.add_argument("--rotate", type=int, default=0,
                        choices=[0, 90, 180, 270])
    parser.add_argument("--white-bg", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    return parser.parse_args(argv)


def run(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    args = parse_args(argv)

    # --- Load program ---
    if not os.path.isfile(args.program):
        print(f"Error: program file not found: {args.program}")
        sys.exit(1)

    with open(args.program) as f:
        code = f.read()

    program_name = os.path.splitext(os.path.basename(args.program))[0]
    print(f"[RunProgram] Loading: {program_name} ({args.program})")

    # --- Projector verification (must happen before hardware setup, on main thread) ---
    if args.mode == "projector":
        from client.display import get_projector_resolution
        pw, ph = get_projector_resolution()
        if not verify_projector(pw, ph):
            print("[RunProgram] Aborted.")
            return

    # --- Hardware stack ---
    config = HardwareConfig(
        url=args.url,
        webcam=args.webcam,
        rotate=args.rotate,
        h_proj_path=args.h_proj,
        mode=args.mode,
        white_bg=args.white_bg,
        no_audio=args.no_audio,
    )
    stack = setup_hardware(config)

    # --- Prime the first frame so table.get_frame() works immediately ---
    for _ in range(5):
        _, raw, _ = stack.camera.get_rectified_frame()
        if raw is not None:
            stack.program_runtime.process_frame(raw)
            break

    # --- Run program ---
    print(f"[RunProgram] Starting '{program_name}'...")
    status = stack.program_runtime.run(program_name, code,
                                        description=f"Standalone: {args.program}")
    if status.state == "error":
        print(f"[RunProgram] Failed to start: {status.error}")
        stack.stop()
        sys.exit(1)

    # --- Display loop (exits when program stops or user quits) ---
    def _program_alive():
        s = stack.program_runtime.get_status(program_name)
        if s and s.state in ("stopped", "error"):
            if s.error:
                print(f"[RunProgram] Program error: {s.error}")
            else:
                print("[RunProgram] Program finished.")
            return False
        return True

    try:
        display_loop(stack, win_name="Orly Program",
                     feed_frames=True, alive_check=_program_alive)
    finally:
        stack.stop()
        print("[RunProgram] Done.")


if __name__ == "__main__":
    run()
