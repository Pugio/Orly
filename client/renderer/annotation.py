"""Render text annotations on black backgrounds for projector overlay."""

import cv2
import numpy as np


def render_annotation(
    text: str,
    width: int,
    height: int,
    font_scale: float = 2.0,
    color: tuple = (0, 255, 255),
) -> np.ndarray:
    """Render text on a black background with word wrapping.

    Args:
        text: Text to render. Empty string produces all-black image.
        width: Output image width in pixels.
        height: Output image height in pixels.
        font_scale: Font size multiplier.
        color: BGR color tuple for the text.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if not text:
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 2))
    margin = 10

    # Word-wrap: split text into lines that fit within width
    words = text.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip() if current_line else word
        (tw, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if tw > width - 2 * margin and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    # Calculate total text height and starting y position
    line_sizes = [cv2.getTextSize(line, font, font_scale, thickness) for line in lines]
    line_height = max(sz[0][1] + sz[1] for sz in line_sizes) if line_sizes else 0
    line_spacing = int(line_height * 0.6)
    total_height = len(lines) * line_height + (len(lines) - 1) * line_spacing

    y = max(line_height, (height - total_height) // 2 + line_height)

    for line in lines:
        cv2.putText(img, line, (margin, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height + line_spacing

    return img
