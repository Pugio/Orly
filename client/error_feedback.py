"""Error recovery UX — friendly feedback when markers/connection/camera/Gemini fail."""

import cv2
import numpy as np


class ErrorFeedback:
    """Tracks system health and provides user-facing error messages + overlays."""

    OVERLAY_NAME = "__system_error"

    # Grace periods before showing errors (seconds)
    _MARKER_GRACE = 3.0
    _CAMERA_GRACE = 5.0
    _GEMINI_GRACE = 10.0

    def __init__(self):
        self._marker_lost_at: float | None = None
        self._connection_lost: bool = False
        self._camera_lost_at: float | None = None
        self._gemini_timeout_at: float | None = None

    def update_marker_status(self, detected: bool, now: float = 0.0):
        if detected:
            self._marker_lost_at = None
        elif self._marker_lost_at is None:
            self._marker_lost_at = now

    def update_connection_status(self, connected: bool, now: float = 0.0):
        self._connection_lost = not connected

    def update_camera_status(self, received: bool, now: float = 0.0):
        if received:
            self._camera_lost_at = None
        elif self._camera_lost_at is None:
            self._camera_lost_at = now

    def update_gemini_status(self, timed_out: bool, now: float = 0.0):
        if not timed_out:
            self._gemini_timeout_at = None
        elif self._gemini_timeout_at is None:
            self._gemini_timeout_at = now

    def get_user_message(self, now: float = 0.0) -> str | None:
        """Return highest-priority error message, or None if all OK.

        Priority: connection > camera > marker > gemini.
        """
        if self._connection_lost:
            return "Connection lost — reconnecting to server..."

        if (self._camera_lost_at is not None
                and (now - self._camera_lost_at) > self._CAMERA_GRACE):
            return "Camera not responding — check USB/WiFi connection."

        if (self._marker_lost_at is not None
                and (now - self._marker_lost_at) > self._MARKER_GRACE):
            return "Markers not detected — make sure the table mat is visible."

        if (self._gemini_timeout_at is not None
                and (now - self._gemini_timeout_at) > self._GEMINI_GRACE):
            return "AI response timed out — Gemini may be overloaded."

        return None

    def get_overlay(self, now: float = 0.0) -> tuple[np.ndarray, list] | None:
        """Return (image, placement) for an error overlay, or None."""
        msg = self.get_user_message(now)
        if msg is None:
            return None

        # Render error message as a small bar at the bottom of the screen
        width, height = 800, 60
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Red background bar
        img[:, :] = (0, 0, 80)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        cv2.putText(img, msg, (10, 40), font, scale, (0, 200, 255), 2, cv2.LINE_AA)
        placement = [900, 100, 1000, 900]  # bottom strip
        return img, placement
