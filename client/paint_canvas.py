"""Per-program paint canvas for direct pixel drawing.

A PaintCanvas is a transparent (black) pixel buffer sized to the projector/screen
resolution. Programs draw on it using simple methods (circle, line, rect, etc.)
and it gets composited onto the main overlay canvas each frame.

Black pixels (0,0,0) are treated as transparent — this matches the projector
paradigm where black = no light = invisible.
"""

import threading

import cv2
import numpy as np


class PaintCanvas:
    """A drawable pixel buffer that composites onto the projection.

    Thread-safe: drawing methods acquire a lock so frame callbacks
    and the main program thread can both draw safely.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._visible = True

    # --- Coordinate helpers ---

    def _to_px(self, y_norm: float, x_norm: float) -> tuple[int, int]:
        """Convert 0-1000 normalised coords to pixel coords (x_px, y_px)."""
        x_px = int(x_norm / 1000.0 * self.width)
        y_px = int(y_norm / 1000.0 * self.height)
        return (x_px, y_px)

    def _size_to_px(self, size_norm: float) -> int:
        """Convert a 0-1000 size value to pixels (uses average of w/h)."""
        avg = (self.width + self.height) / 2.0
        return max(1, int(size_norm / 1000.0 * avg))

    # --- Drawing primitives (all coords in 0-1000 normalised space) ---

    def circle(self, center_y: float, center_x: float, radius: float,
               color: tuple[int, int, int], thickness: int = -1) -> None:
        """Draw a circle. thickness=-1 for filled."""
        pt = self._to_px(center_y, center_x)
        r = self._size_to_px(radius)
        with self._lock:
            cv2.circle(self._canvas, pt, r, color, thickness)

    def line(self, y1: float, x1: float, y2: float, x2: float,
             color: tuple[int, int, int], thickness: int = 2) -> None:
        """Draw a line between two points."""
        pt1 = self._to_px(y1, x1)
        pt2 = self._to_px(y2, x2)
        t = max(1, self._size_to_px(thickness))
        with self._lock:
            cv2.line(self._canvas, pt1, pt2, color, t)

    def rectangle(self, y1: float, x1: float, y2: float, x2: float,
                  color: tuple[int, int, int], thickness: int = -1) -> None:
        """Draw a rectangle. thickness=-1 for filled."""
        pt1 = self._to_px(y1, x1)
        pt2 = self._to_px(y2, x2)
        with self._lock:
            cv2.rectangle(self._canvas, pt1, pt2, color, thickness)

    def text(self, text: str, y: float, x: float,
             color: tuple[int, int, int], scale: float = 1.0,
             thickness: int = 2) -> None:
        """Draw text at a position."""
        pt = self._to_px(y, x)
        with self._lock:
            cv2.putText(self._canvas, text, pt, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness, cv2.LINE_AA)

    def stamp(self, y: float, x: float, radius: float,
              color: tuple[int, int, int]) -> None:
        """Stamp a filled circle (convenience for painting)."""
        self.circle(y, x, radius, color, thickness=-1)

    def clear(self) -> None:
        """Clear the canvas to black (transparent)."""
        with self._lock:
            self._canvas[:] = 0

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, val: bool) -> None:
        self._visible = val

    def get_image(self) -> np.ndarray:
        """Get a copy of the canvas image."""
        with self._lock:
            return self._canvas.copy()

    def has_content(self) -> bool:
        """Check if canvas has any non-black pixels."""
        with self._lock:
            return bool(np.any(self._canvas > 0))

    def composite_onto(self, target: np.ndarray) -> np.ndarray:
        """Composite this canvas onto a target image (additive on non-black pixels).

        Black pixels on the paint canvas are transparent.
        Returns the modified target.
        """
        if not self._visible:
            return target
        with self._lock:
            canvas = self._canvas
            # Resize if dimensions don't match
            th, tw = target.shape[:2]
            if canvas.shape[0] != th or canvas.shape[1] != tw:
                canvas = cv2.resize(canvas, (tw, th))
            # Additive blend: non-black paint pixels overwrite target
            mask = np.any(canvas > 0, axis=2)
            target[mask] = canvas[mask]
        return target
