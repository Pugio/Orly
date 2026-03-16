"""Shared test helpers for camera and capture tests.

Provides synthetic ArUco marker image generation and a mock CaptureSource
for testing CameraCapture without hardware.
"""

from __future__ import annotations

import cv2
import numpy as np

from client.camera import ARUCO_DICT
from client.capture import CaptureSource


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------


def draw_marker(image: np.ndarray, marker_id: int, center: tuple, size: int):
    """Draw an ArUco marker at *center* (x, y) onto *image*.

    Handles markers partially outside the image bounds.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    cx, cy = center
    x1, y1 = cx - size // 2, cy - size // 2

    sx1, sy1 = max(0, -x1), max(0, -y1)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2 = min(image.shape[1], x1 + size)
    dy2 = min(image.shape[0], y1 + size)
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)

    if dx2 > dx1 and dy2 > dy1:
        image[dy1:dy2, dx1:dx2] = marker_bgr[sy1:sy2, sx1:sx2]


def make_table_image(
    size: int = 800,
    marker_size: int = 120,
    margin: int = 120,
    bg: int = 200,
) -> np.ndarray:
    """Synthetic table with 4 ArUco markers (IDs 0-3) at the corners."""
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    positions = {
        0: (margin, margin),
        1: (size - margin, margin),
        2: (size - margin, size - margin),
        3: (margin, size - margin),
    }
    for mid, pos in positions.items():
        draw_marker(img, mid, pos, marker_size)
    return img


def make_blank_image(size: int = 800) -> np.ndarray:
    """Plain grey image with no markers."""
    return np.full((size, size, 3), 200, dtype=np.uint8)


def make_detector() -> cv2.aruco.ArucoDetector:
    """Create a standard ArUco detector for tests."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


# ---------------------------------------------------------------------------
# Mock capture source
# ---------------------------------------------------------------------------


class MockCaptureSource(CaptureSource):
    """CaptureSource backed by a pre-loaded frame list for testing.

    Each ``read()`` call consumes one frame. None entries simulate failures.

    Attributes:
        open_count: Number of times open() was called.
        close_count: Number of times close() was called.
    """

    def __init__(self, frames: list[np.ndarray | None]):
        self._frames = frames
        self._idx = 0
        self.open_count = 0
        self.close_count = 0

    def open(self) -> None:
        self._idx = 0
        self.open_count += 1

    def read(self) -> np.ndarray | None:
        if self._idx >= len(self._frames):
            return None
        frame = self._frames[self._idx]
        self._idx += 1
        if frame is None:
            return None
        return frame.copy()

    def close(self) -> None:
        self.close_count += 1

    @property
    def name(self) -> str:
        return "MockCaptureSource"


def make_camera_with_mock(frames: list, rotate: int = 0):
    """Build a CameraCapture wired to a MockCaptureSource (already started)."""
    from client.camera import CameraCapture
    src = MockCaptureSource(frames)
    cam = CameraCapture(source=src, rotate=rotate)
    cam.start()
    return cam
