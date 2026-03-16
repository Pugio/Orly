"""Video loading placeholder for the projector overlay.

Actual video generation + playback is handled by VideoGenerator/VideoPlayer.
This module provides only the loading placeholder — it is NOT registered in
the renderer registry because video is handled by dedicated generate_video /
play_video tools, not by project_overlay.
"""

import cv2
import numpy as np


def render_video_loading(data: dict, width: int, height: int, title: str = "") -> np.ndarray:
    """Render a loading placeholder for video generation.

    Shows a film-strip border and "Generating video..." text on black background.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw magenta border (distinguishes from image loading which uses cyan)
    color = (255, 0, 255)  # magenta in BGR
    thickness = max(2, min(width, height) // 60)
    cv2.rectangle(img, (thickness, thickness),
                  (width - thickness, height - thickness), color, thickness)

    # "Generating video..." text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, min(width, height) / 400)
    text = "Generating video..."
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    tx = (width - tw) // 2
    ty = (height + th) // 2
    cv2.putText(img, text, (tx, ty), font, scale, color, 2, cv2.LINE_AA)

    # Show truncated prompt below
    prompt = data.get("prompt", title)
    prompt_display = prompt[:50] + "..." if len(prompt) > 50 else prompt
    scale_small = scale * 0.5
    (tw2, th2), _ = cv2.getTextSize(prompt_display, font, scale_small, 1)
    tx2 = (width - tw2) // 2
    ty2 = ty + th + int(th2 * 1.5)
    if ty2 < height - 10:
        cv2.putText(img, prompt_display, (tx2, ty2), font, scale_small,
                    (180, 0, 180), 1, cv2.LINE_AA)

    # Duration hint
    dur = data.get("duration", 5)
    dur_text = f"~{dur}s clip (may take 1-6 min)"
    (tw3, _), _ = cv2.getTextSize(dur_text, font, scale_small, 1)
    tx3 = (width - tw3) // 2
    ty3 = ty2 + int(th2 * 2)
    if ty3 < height - 10:
        cv2.putText(img, dur_text, (tx3, ty3), font, scale_small,
                    (120, 0, 120), 1, cv2.LINE_AA)

    return img
