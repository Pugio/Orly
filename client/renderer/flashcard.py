"""Render flashcard overlays on black backgrounds for projector overlay."""

import cv2
import numpy as np


def _render_flashcard_impl(
    front: str,
    back: str,
    show_back: bool,
    width: int,
    height: int,
    title: str = "",
) -> np.ndarray:
    """Render a flashcard on a black background.

    Shows the front text (cyan, larger) or back text (green) depending
    on show_back. Draws a rounded rectangle border and a corner label.

    Args:
        front: Text for the front of the card.
        back: Text for the back of the card.
        show_back: If True, show the back text; otherwise show front.
        width: Output image width in pixels.
        height: Output image height in pixels.
        title: Optional title rendered at the top.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw rounded rectangle border in cyan
    border_color = (255, 255, 0)  # cyan in BGR
    margin = 8
    radius = 15
    thickness = 2

    # Draw the rounded rectangle using lines and ellipses
    x1, y1 = margin, margin
    x2, y2 = width - margin, height - margin
    # Top and bottom lines
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border_color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border_color, thickness)
    # Left and right lines
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border_color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border_color, thickness)
    # Corner arcs
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border_color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border_color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border_color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border_color, thickness)

    # Corner label: "FRONT" or "BACK"
    label = "BACK" if show_back else "FRONT"
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 0.4
    label_thickness = 1
    label_color = (128, 128, 128)  # grey
    cv2.putText(img, label, (margin + 10, margin + 20), label_font, label_scale, label_color, label_thickness, cv2.LINE_AA)

    # Determine text and color
    text = back if show_back else front
    text_color = (0, 255, 0) if show_back else (255, 255, 0)  # green for back, cyan for front

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font based on image size
    font_scale = max(0.5, min(width, height) / 200.0)
    text_thickness = max(1, int(font_scale * 2))

    # Title rendering
    y_offset = margin + 40
    if title:
        title_scale = font_scale * 0.6
        title_thickness = max(1, int(title_scale * 2))
        (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_thickness)
        tx = (width - tw) // 2
        cv2.putText(img, title, (tx, y_offset + th), font, title_scale, (200, 200, 200), title_thickness, cv2.LINE_AA)
        y_offset += th + 15

    if not text:
        return img

    # Word-wrap the main text
    inner_margin = margin + 20
    words = text.split()
    lines: list[str] = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip() if current_line else word
        (tw, _), _ = cv2.getTextSize(test_line, font, font_scale, text_thickness)
        if tw > width - 2 * inner_margin and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    # Calculate line metrics
    line_sizes = [cv2.getTextSize(line, font, font_scale, text_thickness) for line in lines]
    line_height = max(sz[0][1] + sz[1] for sz in line_sizes) if line_sizes else 0
    line_spacing = int(line_height * 0.6)
    total_height = len(lines) * line_height + (len(lines) - 1) * line_spacing

    # Center vertically in the remaining space
    available_top = y_offset
    available_height = height - available_top - margin
    y = available_top + max(0, (available_height - total_height) // 2) + line_height

    for line in lines:
        # Center horizontally
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, text_thickness)
        tx = (width - tw) // 2
        cv2.putText(img, line, (tx, y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)
        y += line_height + line_spacing

    return img


def render_flashcard(data: dict, width: int, height: int, title: str = "") -> np.ndarray:
    """Registry-compatible wrapper: render flashcard from data dict."""
    return _render_flashcard_impl(
        front=data.get("front", ""),
        back=data.get("back", ""),
        show_back=data.get("show_back", False),
        width=width,
        height=height,
        title=title,
    )


SPEC = {
    "name": "flashcard",
    "description": "Flashcard with front and back sides that can be flipped.",
    "data_format": '{"front": "Question text", "back": "Answer text", "show_back": false}.',
    "prompt_hint": "Use for vocabulary, definitions, Q&A review. Flip with flip_flashcard tool.",
    "render": render_flashcard,
}
