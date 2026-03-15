"""Test the full overlay pipeline: OverlayManager → projector display.

Simulates tool calls without needing the backend/Gemini.
Type commands to trigger overlays, see them on projector.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import threading
import queue

import cv2
import numpy as np

from client.display import show_on_projector, show_on_laptop, get_projector_resolution
from client.overlay_manager import OverlayManager


def main():
    mode = "projector" if "--projector" in sys.argv else "screen"

    proj_width, proj_height = get_projector_resolution()
    H_proj = None

    # Load H_proj if provided
    for i, arg in enumerate(sys.argv):
        if arg == "--h-proj" and i + 1 < len(sys.argv):
            data = np.load(sys.argv[i + 1])
            H_proj = data["H_proj"]
            proj_width = int(data["proj_width"])
            proj_height = int(data["proj_height"])
            print(f"Loaded H_proj from {sys.argv[i + 1]}")

    om = OverlayManager(
        H_proj=H_proj,
        proj_width=proj_width,
        proj_height=proj_height,
        mode=mode,
    )
    print(f"Mode: {mode}, Resolution: {proj_width}x{proj_height}")
    print()
    print("Commands (type + Enter):")
    print("  h          — highlight at [200, 200, 800, 800]")
    print("  hfull      — highlight full mat [0, 0, 1000, 1000]")
    print("  h2         — highlight at [100, 100, 400, 400]")
    print("  g          — graph y=x^2 at [100, 500, 500, 950]")
    print("  a          — annotation at [600, 100, 900, 500]")
    print("  clear      — clear all overlays")
    print("  q          — quit")
    print()

    # Input thread
    input_queue: queue.Queue[str] = queue.Queue()

    def _reader():
        try:
            while True:
                line = input().strip().lower()
                input_queue.put(line)
        except (EOFError, KeyboardInterrupt):
            input_queue.put("q")

    threading.Thread(target=_reader, daemon=True).start()

    win_name = "Overlay Pipeline Test"

    while True:
        # Show current canvas
        if mode == "projector":
            show_on_projector(win_name, om.canvas, fullscreen=True)
        else:
            show_on_laptop(win_name, om.canvas)
        cv2.waitKey(50)

        # Check for commands
        try:
            cmd = input_queue.get_nowait()
        except queue.Empty:
            continue

        if cmd == "q":
            break
        elif cmd == "clear":
            om.clear()
            print("Cleared.")
        elif cmd == "h":
            om.handle_tool_result("project_overlay", {
                "content_type": "highlight",
                "placement": [200, 200, 800, 800],
                "title": "Test Highlight",
                "data": {"color": "#00ffff"},
            })
            print(f"Highlight rendered. Canvas non-black: {np.count_nonzero(om.canvas)}")
        elif cmd == "hfull":
            om.handle_tool_result("project_overlay", {
                "content_type": "highlight",
                "placement": [0, 0, 1000, 1000],
                "title": "Full Mat",
                "data": {"color": "#00ffff"},
            })
            print(f"Full highlight rendered. Canvas non-black: {np.count_nonzero(om.canvas)}")
        elif cmd == "h2":
            om.handle_tool_result("project_overlay", {
                "content_type": "highlight",
                "placement": [100, 100, 400, 400],
                "title": "Small Highlight",
                "data": {"color": "#ff00ff"},
            })
            print(f"Highlight rendered. Canvas non-black: {np.count_nonzero(om.canvas)}")
        elif cmd == "g":
            om.handle_tool_result("project_overlay", {
                "content_type": "graph",
                "placement": [100, 500, 500, 950],
                "title": "y = x^2",
                "data": {"expression": "x**2", "x_range": [-5, 5], "y_range": [-5, 25]},
            })
            print(f"Graph rendered. Canvas non-black: {np.count_nonzero(om.canvas)}")
        elif cmd == "a":
            om.handle_tool_result("project_overlay", {
                "content_type": "annotation",
                "placement": [600, 100, 900, 500],
                "title": "Note",
                "data": {"text": "Check your work!"},
            })
            print(f"Annotation rendered. Canvas non-black: {np.count_nonzero(om.canvas)}")
        else:
            print(f"Unknown command: {cmd}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
