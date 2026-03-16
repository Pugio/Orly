"""
Projector verification — tests the real overlay pipeline end-to-end.

Uses OverlayManager with the same H_proj, mode, and rotation as the real
system, so every pattern goes through orient_overlay → place_on_canvas.
If it looks right here, it'll look right in production.

Usage:
    uv run python calibration/projector_verify.py --rotate 270
    uv run python calibration/projector_verify.py --homography projector_homography.npz --rotate 270
"""

import argparse
import os
import queue as queue_mod
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client.display import show_on_projector, get_projector_resolution
from client.overlay_manager import OverlayManager


# ---------------------------------------------------------------------------
# Pattern generators — each returns (description, content_type, placement, data)
# that go through the real OverlayManager pipeline.
# ---------------------------------------------------------------------------


def pattern_annotation():
    """Text annotation — readable orientation check."""
    return (
        "Annotation — 'Hello from Orly!' (should be human-readable)",
        "annotation",
        [200, 200, 800, 800],
        {"text": "Hello from Orly!"},
        "Hello",
    )


def pattern_graph():
    """Graph overlay rendered via matplotlib."""
    return (
        "Graph — y = x^2 - 3x + 2 (center of table)",
        "graph",
        [100, 100, 900, 900],
        {"expression": "x**2 - 3*x + 2", "x_range": [-5, 5], "y_range": [-5, 10]},
        "Graph",
    )


def pattern_markdown():
    """Markdown with multiple lines — tests text orientation thoroughly."""
    return (
        "Markdown — multi-line text (all should read left-to-right)",
        "markdown",
        [50, 50, 950, 950],
        {"text": "# Orientation Test\n\n- Line 1: **Human side** (bottom)\n- Line 2: Projector side (top)\n- If you can read this, orientation is correct!"},
        "Orientation",
    )


def pattern_corners():
    """Four annotations in each corner — tests placement + orientation."""
    return [
        ("Corner labels — TL=M0, TR=M3, BL=M1, BR=M2", [
            ("annotation", [0, 0, 200, 300], {"text": "TL (M0)"}, "TL"),
            ("annotation", [0, 700, 200, 1000], {"text": "TR (M3)"}, "TR"),
            ("annotation", [800, 0, 1000, 300], {"text": "BL (M1)"}, "BL"),
            ("annotation", [800, 700, 1000, 1000], {"text": "BR (M2)"}, "BR"),
        ]),
    ]


def pattern_number_line():
    """Number line — checks orientation of a different renderer."""
    return (
        "Number line — 0 to 10 (should read left-to-right)",
        "number_line",
        [300, 50, 700, 950],
        {"min_val": 0, "max_val": 10, "points": [{"value": 3}, {"value": 5}, {"value": 7}]},
        "NumLine",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Projector verification via real overlay pipeline")
    parser.add_argument(
        "--homography", type=str, default=None,
        help="Path to projector homography .npz file",
    )
    parser.add_argument(
        "--rotate", type=int, default=0, choices=[0, 90, 180, 270],
        help="Camera rotation (must match --rotate used with client.main)",
    )
    args = parser.parse_args()

    # Auto-detect homography file
    if args.homography is None:
        for candidate in ["projector_homography.npz",
                          os.path.join(os.path.dirname(__file__), "..", "projector_homography.npz")]:
            if os.path.exists(candidate):
                args.homography = candidate
                break

    # Load calibration
    H_proj = None
    if args.homography:
        data = np.load(args.homography)
        H_proj = data["H_proj"]
        proj_width = int(data.get("proj_width", 1280))
        proj_height = int(data.get("proj_height", 720))
        print(f"Loaded H_proj from {args.homography}")
    else:
        proj_width, proj_height = get_projector_resolution()
        print("(No homography — using screen fallback mode)")

    mode = "projector" if H_proj is not None else "screen"
    print(f"Resolution: {proj_width}x{proj_height}, mode: {mode}, rotate: {args.rotate}")

    # Build pattern list: each entry is (description, render_fn)
    # render_fn takes an OverlayManager and draws on its canvas.

    def _make_single_overlay(desc, content_type, placement, data, title):
        def render(om):
            overlay = om.render_overlay(content_type, placement, title, data)
            om._show_overlay(overlay, placement)
        return (desc, render)

    def _make_multi_overlay(desc, items):
        def render(om):
            for content_type, placement, data, title in items:
                overlay = om.render_overlay(content_type, placement, title, data)
                om._show_overlay(overlay, placement)
        return (desc, render)

    patterns = [
        _make_single_overlay(*pattern_annotation()),
        _make_single_overlay(*pattern_graph()),
        _make_single_overlay(*pattern_markdown()),
        _make_single_overlay(*pattern_number_line()),
    ]

    for desc, items in pattern_corners():
        patterns.append(_make_multi_overlay(desc, items))

    # --- Display loop ---
    current = 0
    win_name = "Projector Verify"

    def _render_current():
        om = OverlayManager(
            H_proj=H_proj,
            proj_width=proj_width,
            proj_height=proj_height,
            mode=mode,
            image_rotate=args.rotate,
        )
        patterns[current][1](om)
        return om.canvas

    print(f"\nPatterns ({len(patterns)}):")
    for i, (name, _) in enumerate(patterns):
        marker = " →" if i == current else "  "
        print(f"  {marker} {i + 1}. {name}")

    print(f"\nControls: n/ENTER = next, p = previous, q = quit")
    print(f"Showing: {patterns[current][0]}")

    show_on_projector(win_name, _render_current(), fullscreen=True)

    # Threaded stdin reader — cv2.waitKey() doesn't capture input on macOS.
    input_queue: queue_mod.Queue[str | None] = queue_mod.Queue()

    def _stdin_reader():
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    input_queue.put(None)
                    break
                input_queue.put(line.strip().lower())
        except Exception:
            input_queue.put(None)

    reader = threading.Thread(target=_stdin_reader, daemon=True)
    reader.start()

    while True:
        cv2.waitKey(50)
        try:
            cmd = input_queue.get_nowait()
        except queue_mod.Empty:
            continue

        if cmd is None or cmd == "q":
            break
        elif cmd in ("n", ""):
            current = (current + 1) % len(patterns)
        elif cmd == "p":
            current = (current - 1) % len(patterns)
        else:
            print(f"  Unknown command '{cmd}'. Use n/p/q.")
            continue

        print(f"Showing: {patterns[current][0]}")
        show_on_projector(win_name, _render_current())

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
