"""Render semi-transparent highlight rectangles for projector overlay."""

import numpy as np


def render_highlight(
    width: int,
    height: int,
    color_hex: str = "#00ffff",
    alpha: float = 0.3,
) -> np.ndarray:
    """Render a semi-transparent colored rectangle on black background.

    Args:
        width: Output image width in pixels.
        height: Output image height in pixels.
        color_hex: Hex color string (e.g. "#00ffff" for cyan).
        alpha: Opacity value 0.0 (transparent) to 1.0 (opaque).

    Returns:
        BGRA numpy array (uint8) with shape (height, width, 4).
    """
    # Parse hex color to RGB
    hex_clean = color_hex.lstrip("#")
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)

    img = np.zeros((height, width, 4), dtype=np.uint8)
    img[:, :, 0] = b  # Blue channel
    img[:, :, 1] = g  # Green channel
    img[:, :, 2] = r  # Red channel
    img[:, :, 3] = int(alpha * 255)  # Alpha channel

    return img
