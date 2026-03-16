"""Debug overlay that renders live latency stats onto the display.

Renders a small panel showing per-stage pipeline latencies.
Dark background with bright text — projector-friendly.
"""

from __future__ import annotations

import cv2
import numpy as np

from client.latency_tracker import LatencyTracker


def render_latency_overlay(
    tracker: LatencyTracker,
    width: int = 400,
    height: int = 120,
) -> np.ndarray:
    """Render latency stats as a BGR image with dark background.

    Args:
        tracker: LatencyTracker with recorded data.
        width: Output image width.
        height: Output image height.

    Returns:
        BGR numpy array (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)

    avgs = tracker.averages()
    latest = tracker.summary()

    if not avgs and not latest:
        cv2.putText(
            img, "No latency data", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1,
        )
        return img

    y = 20
    line_height = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1

    # Header
    cv2.putText(img, "Pipeline Latency", (10, y), font, 0.5, (0, 200, 255), 1)
    y += line_height + 4

    total = 0.0
    for stage in sorted(avgs.keys()):
        if y + line_height > height - 5:
            break
        avg_ms = avgs[stage]
        cur_ms = latest.get(stage)
        total += avg_ms

        # Color: green < 10ms, yellow < 50ms, red >= 50ms
        if avg_ms < 10:
            color = (0, 220, 0)
        elif avg_ms < 50:
            color = (0, 220, 220)
        else:
            color = (0, 80, 255)

        cur_str = f"{cur_ms:.0f}" if cur_ms is not None else "?"
        text = f"{stage}: {avg_ms:.1f}ms (cur: {cur_str}ms)"
        cv2.putText(img, text, (10, y), font, font_scale, color, thickness)
        y += line_height

    # Total line
    if y + line_height <= height:
        color = (0, 220, 0) if total < 50 else (0, 220, 220) if total < 100 else (0, 80, 255)
        cv2.putText(img, f"TOTAL: {total:.1f}ms", (10, y), font, 0.5, color, 1)

    return img


def composite_debug_overlay(
    canvas: np.ndarray,
    overlay: np.ndarray,
    x: int = 10,
    y: int = 10,
) -> np.ndarray:
    """Composite a debug overlay onto a canvas, clipping to bounds.

    Args:
        canvas: Background image (modified in-place and returned).
        overlay: Debug overlay image.
        x: X position on canvas.
        y: Y position on canvas.

    Returns:
        The canvas with overlay composited.
    """
    oh, ow = overlay.shape[:2]
    ch, cw = canvas.shape[:2]

    # Clip to canvas bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2 = min(x + ow, cw)
    y2 = min(y + oh, ch)

    if x2 <= x1 or y2 <= y1:
        return canvas

    # Source region (handle negative x/y)
    sx = x1 - x
    sy = y1 - y
    sw = x2 - x1
    sh = y2 - y1

    canvas[y1:y2, x1:x2] = overlay[sy:sy + sh, sx:sx + sw]
    return canvas
