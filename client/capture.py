"""Pluggable capture sources for the camera pipeline.

Each source implements open/read/close for raw BGR frames.
CameraCapture (in camera.py) handles ArUco detection, homography,
rectification, and JPEG encoding — the source just provides frames.

Supported sources:
    USBWebcamSource  — local USB/built-in webcam via cv2.VideoCapture(index)
    IPWebcamSource   — IP Webcam (Android) MJPEG stream via HTTP
"""

from __future__ import annotations

import abc
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CaptureSource(abc.ABC):
    """Abstract base class for frame capture sources.

    Subclasses must implement open(), read(), and close().
    """

    @abc.abstractmethod
    def open(self) -> None:
        """Open the capture device. Raises RuntimeError on failure."""
        ...

    @abc.abstractmethod
    def read(self) -> np.ndarray | None:
        """Read a single BGR frame. Returns None on failure."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release the capture device."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for logging."""
        return self.__class__.__name__


class _VideoCaptureSource(CaptureSource):
    """Shared base for sources backed by cv2.VideoCapture.

    Handles open-guard, grab-and-discard for frame freshness,
    read delegation, and idempotent close. Subclasses only need
    to implement ``_create_capture()`` and ``name``.
    """

    def __init__(self, buffer_size: int = 1):
        self.buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None

    @abc.abstractmethod
    def _create_capture(self) -> cv2.VideoCapture:
        """Create and return a cv2.VideoCapture for this source.

        Called by open(). Should NOT call isOpened() — open() handles that.
        """
        ...

    def open(self) -> None:
        # Release any previous capture to avoid resource leaks on double-open.
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._cap = self._create_capture()
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(f"Failed to open capture source: {self.name}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        self._cap.grab()  # discard one stale buffered frame
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class USBWebcamSource(_VideoCaptureSource):
    """Captures frames from a local USB or built-in webcam.

    Uses cv2.VideoCapture with a device index. Grabs and discards one
    buffered frame before each read to reduce staleness.

    Args:
        index: Device index (0 = default camera).
        buffer_size: VideoCapture buffer size (default 1 for freshness).
    """

    def __init__(self, index: int = 0, buffer_size: int = 1):
        super().__init__(buffer_size)
        self.index = index

    def _create_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.index)

    @property
    def name(self) -> str:
        return f"USBWebcam({self.index})"


class IPWebcamSource(_VideoCaptureSource):
    """Captures frames from IP Webcam (Android app) MJPEG stream.

    Connects to ``{url}/video`` and reads via cv2.VideoCapture.
    Grabs and discards one buffered frame before each read.

    Args:
        url: Base URL of the IP Webcam server (e.g. ``http://192.168.1.100:8080``).
        buffer_size: VideoCapture buffer size (default 1).
    """

    def __init__(self, url: str, buffer_size: int = 1):
        super().__init__(buffer_size)
        self.url = url.rstrip("/")

    def _create_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(f"{self.url}/video")

    @property
    def name(self) -> str:
        return f"IPWebcam({self.url})"
