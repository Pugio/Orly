"""Named overlay state tracker wrapping OverlayManager.

Tracks overlays by name, supports add/remove/query, and recomposites
the canvas whenever the overlay set changes.
"""

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class OverlayEntry:
    """A single named overlay on the canvas."""
    name: str
    content_type: str
    placement: list[float]    # [ymin, xmin, ymax, xmax] in 0-1000
    title: str
    data: dict
    image: np.ndarray         # The rendered BGR image (before placement)
    created_at: float = field(default_factory=time.time)


class OverlayStateManager:
    """Tracks named overlays and provides state queries."""

    def __init__(self, overlay_manager):
        """
        Args:
            overlay_manager: The OverlayManager instance to render on.
                             We call its place_on_canvas() method.
        """
        self._om = overlay_manager
        self._overlays: dict[str, OverlayEntry] = {}
        self._lock = threading.Lock()

    def add(self, name: str, content_type: str, placement: list[float],
            title: str, data: dict, image: np.ndarray,
            recomposite: bool = True) -> None:
        """Add or replace a named overlay.

        Args:
            recomposite: If True (default), re-render all overlays onto canvas.
                         Set to False when the caller has already placed the
                         overlay on canvas (e.g. overlay_manager.handle_tool_result).
        """
        with self._lock:
            self._overlays[name] = OverlayEntry(
                name=name, content_type=content_type, placement=placement,
                title=title, data=data, image=image.copy()
            )
            if recomposite:
                self._recomposite_locked()

    def remove(self, name: str) -> bool:
        """Remove overlay by name. Returns True if found. Recomposites."""
        with self._lock:
            if name in self._overlays:
                del self._overlays[name]
                self._recomposite_locked()
                return True
            return False

    def get(self, name: str) -> OverlayEntry | None:
        with self._lock:
            return self._overlays.get(name)

    def clear(self) -> None:
        with self._lock:
            self._overlays.clear()
            self._om.clear()

    def list_names(self) -> list[str]:
        with self._lock:
            return list(self._overlays.keys())

    def to_json(self) -> dict:
        """Serialize current state to JSON-safe dict."""
        with self._lock:
            overlays = []
            for entry in self._overlays.values():
                overlays.append({
                    "name": entry.name,
                    "content_type": entry.content_type,
                    "placement": entry.placement,
                    "title": entry.title,
                    "created_at": entry.created_at,
                })
            return {
                "overlays": overlays,
                "count": len(overlays),
                "dimensions": [1000, 1000],
            }

    def to_ascii(self, width: int = 40, height: int = 20) -> str:
        """Render ASCII grid showing overlay positions.

        '.' = empty space
        First character of overlay name = overlay occupies that cell
        '#' = multiple overlays overlap at that cell
        """
        grid = [['.' for _ in range(width)] for _ in range(height)]

        with self._lock:
            entries = list(self._overlays.values())

        for entry in entries:
            ymin, xmin, ymax, xmax = entry.placement
            # Map 0-1000 coords to grid cells
            r1 = max(0, int(ymin / 1000 * height))
            r2 = min(height, int(ymax / 1000 * height))
            c1 = max(0, int(xmin / 1000 * width))
            c2 = min(width, int(xmax / 1000 * width))

            char = entry.name[0] if entry.name else '?'
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if grid[r][c] == '.':
                        grid[r][c] = char
                    elif grid[r][c] != char:
                        grid[r][c] = '#'  # overlap

        return '\n'.join(''.join(row) for row in grid)

    def recomposite(self) -> None:
        """Re-render all overlays onto a fresh canvas (preserving order)."""
        with self._lock:
            self._recomposite_locked()

    def _recomposite_locked(self) -> None:
        """Re-render all overlays (caller must hold self._lock)."""
        self._om.canvas = self._om._make_bg()
        self._om._has_content = bool(self._overlays)
        for entry in self._overlays.values():
            overlay = entry.image.copy()
            if self._om.mode == "projector" and entry.content_type != "highlight":
                overlay = cv2.rotate(overlay, cv2.ROTATE_180)
            self._om.canvas = self._om.place_on_canvas(overlay, entry.placement)
