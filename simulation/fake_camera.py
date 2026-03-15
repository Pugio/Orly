"""Generate synthetic video frames for simulation (no camera required).

Produces JPEG bytes identical in format to CameraCapture.get_rectified_frame().
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_test_frame(
    width: int = 768,
    height: int = 768,
    text_lines: list[str] | None = None,
) -> np.ndarray:
    """Create a BGR test image simulating a homework sheet on a desk.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        text_lines: Lines of text to draw. Defaults to sample math problems.

    Returns:
        BGR numpy array (uint8).

    >>> frame = generate_test_frame()
    >>> frame.shape
    (768, 768, 3)
    >>> frame.dtype
    dtype('uint8')
    """
    # White paper on grey desk.
    frame = np.full((height, width, 3), 180, dtype=np.uint8)

    # Paper rectangle (slightly inset).
    margin = int(min(width, height) * 0.08)
    cv2.rectangle(
        frame,
        (margin, margin),
        (width - margin, height - margin),
        (255, 255, 255),
        cv2.FILLED,
    )

    if text_lines is None:
        text_lines = [
            "Math Homework",
            "",
            "1) 2 + 2 = ___",
            "2) 5 x 3 = ___",
            "3) 12 / 4 = ___",
            "4) 7 - 3 = ___",
        ]

    # Draw text.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    color = (30, 30, 30)
    thickness = 2
    y = margin + 50
    for line in text_lines:
        if line:
            cv2.putText(frame, line, (margin + 30, y), font, scale, color, thickness)
        y += 40

    return frame


def encode_frame_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame as JPEG bytes.

    >>> jpeg = encode_frame_jpeg(generate_test_frame())
    >>> jpeg[:2]
    b'\\xff\\xd8'
    """
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", frame, params)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def generate_test_jpeg(
    width: int = 768,
    height: int = 768,
    text_lines: list[str] | None = None,
) -> bytes:
    """Convenience: generate a test frame and return JPEG bytes.

    >>> jpeg = generate_test_jpeg()
    >>> isinstance(jpeg, bytes) and len(jpeg) > 100
    True
    """
    frame = generate_test_frame(width, height, text_lines)
    return encode_frame_jpeg(frame)


def load_image_as_jpeg(path: str, width: int = 768, height: int = 768) -> bytes:
    """Load an image file, resize it, and return JPEG bytes.

    Args:
        path: Path to an image file (PNG, JPG, etc.).
        width: Target width.
        height: Target height.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If OpenCV cannot read the file.
    """
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"OpenCV could not read image: {path}")
    frame = cv2.resize(frame, (width, height))
    return encode_frame_jpeg(frame)
