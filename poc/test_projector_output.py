"""Quick test: render overlays directly on the projector display.

Bypasses H_proj to verify the projector output pipeline works.
Press SPACE to cycle through test patterns, Q to quit.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from client.display import show_on_projector, get_projector_resolution
from client.renderer.graph import render_graph
from client.renderer.annotation import render_annotation
from client.renderer.highlight import render_highlight


def main():
    w, h = get_projector_resolution()
    print(f"Projector: {w}x{h}")

    patterns = []

    # 1. Cyan rectangle in center
    canvas1 = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(canvas1, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 255, 0), 3)
    cv2.putText(canvas1, "Test Pattern 1: Rectangle", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    patterns.append(("Rectangle", canvas1))

    # 2. Graph overlay
    graph = render_graph("x**2 - 3*x + 2", [-5, 5], [-5, 10], w // 2, h // 2)
    canvas2 = np.zeros((h, w, 3), dtype=np.uint8)
    canvas2[h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2] = graph
    patterns.append(("Graph: x^2 - 3x + 2", canvas2))

    # 3. Annotation
    anno = render_annotation("Hello from TableLight!", w // 2, h // 4,
                             font_scale=2.0, color=(0, 255, 255))
    canvas3 = np.zeros((h, w, 3), dtype=np.uint8)
    canvas3[h // 3:h // 3 + h // 4, w // 4:w // 4 + w // 2] = anno
    patterns.append(("Annotation", canvas3))

    # 4. Highlight (semi-transparent cyan)
    hl = render_highlight(w // 2, h // 2, "#00ffff", alpha=0.5)
    canvas4 = np.zeros((h, w, 3), dtype=np.uint8)
    canvas4[h // 4:h // 4 + h // 2, w // 4:w // 4 + w // 2] = hl[:, :, :3]
    patterns.append(("Highlight", canvas4))

    import threading
    import queue

    key_queue: queue.Queue[str] = queue.Queue()

    def _reader():
        while True:
            try:
                line = input().strip().lower()
                if line:
                    key_queue.put(line)
            except (EOFError, KeyboardInterrupt):
                key_queue.put("q")
                return

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    current = 0
    print(f"\nShowing: {patterns[current][0]}")
    print("Type 'n' + Enter = next, 'q' + Enter = quit")

    show_on_projector("Projector Test", patterns[current][1], fullscreen=True)

    while True:
        cv2.waitKey(100)

        try:
            cmd = key_queue.get_nowait()
        except queue.Empty:
            continue

        if cmd == "q":
            break
        elif cmd in ("n", ""):
            current = (current + 1) % len(patterns)
            print(f"Showing: {patterns[current][0]}")
            show_on_projector("Projector Test", patterns[current][1])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
