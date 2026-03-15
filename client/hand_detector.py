"""Hand/pointing gesture detection — pure functions + tracker.

MediaPipe is only imported lazily inside detect() to keep this module
testable without the dependency installed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class PointingResult:
    """Result of a pointing gesture detection."""

    fingertip: tuple[float, float]  # (y, x) in 0-1000 normalised
    confidence: float
    is_pointing: bool


def is_pointing_gesture(
    landmarks: list[tuple[float, float, float]],
    extension_threshold: float = 0.1,
) -> bool:
    """Check if hand is in a pointing gesture.

    landmarks: 21 (x, y, z) tuples from MediaPipe hand detection.
    Index finger must be extended (tip 8 far from MCP 5) and all other
    fingers must be curled (tips 12, 16, 20 close to their MCPs 9, 13, 17).

    Uses Euclidean distance. extension_threshold is the minimum distance
    for a finger to be considered "extended" (strictly greater than).
    """

    def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    index_dist = _dist(landmarks[8], landmarks[5])
    middle_dist = _dist(landmarks[12], landmarks[9])
    ring_dist = _dist(landmarks[16], landmarks[13])
    pinky_dist = _dist(landmarks[20], landmarks[17])

    index_extended = index_dist > extension_threshold
    others_curled = (
        middle_dist <= extension_threshold
        and ring_dist <= extension_threshold
        and pinky_dist <= extension_threshold
    )

    return index_extended and others_curled


def fingertip_to_table_coords(
    pixel_y: float,
    pixel_x: float,
    frame_h: int,
    frame_w: int,
) -> tuple[float, float]:
    """Convert pixel position to 0-1000 table coordinates.

    Returns (y, x) in normalised table space.
    """
    table_y = (pixel_y / frame_h) * 1000
    table_x = (pixel_x / frame_w) * 1000
    return (table_y, table_x)


@dataclass
class PointingTracker:
    """Debounced, stability-aware pointing tracker.

    Tracks whether a pointing gesture has been held steadily at roughly
    the same position for at least `stability_time` seconds. After a
    notification is consumed, enforces a `debounce` cooldown before
    allowing the next one.
    """

    stability_time: float = 1.0
    jitter_threshold: float = 50.0
    debounce: float = 2.0

    # Internal state
    _position: tuple[float, float] | None = field(default=None, repr=False)
    _first_seen: float | None = field(default=None, repr=False)
    _stable: bool = field(default=False, repr=False)
    _last_notify_time: float | None = field(default=None, repr=False)

    def update(self, result: PointingResult | None, now: float | None = None) -> None:
        """Feed a new detection (or None if no hand detected)."""
        if now is None:
            now = time.monotonic()

        if result is None:
            self._position = None
            self._first_seen = None
            self._stable = False
            return

        pos = result.fingertip

        if self._position is not None:
            dy = pos[0] - self._position[0]
            dx = pos[1] - self._position[1]
            dist = math.sqrt(dy * dy + dx * dx)
            if dist > self.jitter_threshold:
                # Position jumped — reset stability
                self._first_seen = now
                self._stable = False
        else:
            self._first_seen = now

        self._position = pos

        if self._first_seen is not None and (now - self._first_seen) >= self.stability_time:
            self._stable = True

    def current_point(self) -> tuple[float, float] | None:
        """Return the current tracked position, or None."""
        return self._position

    def is_stable(self) -> bool:
        """Whether the pointing gesture has been stable long enough."""
        return self._stable

    def should_notify(self, now: float | None = None) -> bool:
        """Check if a notification should fire (stable + debounce elapsed).

        Calling this when it returns True consumes the notification
        (sets the last-notify timestamp).
        """
        if now is None:
            now = time.monotonic()

        if not self._stable:
            return False

        if self._last_notify_time is not None:
            if (now - self._last_notify_time) < self.debounce:
                return False

        self._last_notify_time = now
        return True
