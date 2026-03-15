"""Frame-by-frame object tracking using color histograms (CamShift) and template matching."""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np


@dataclass
class TrackedObject:
    """State of a tracked object."""
    name: str
    bbox: tuple[int, int, int, int]  # (y, x, h, w) in pixel coords
    center: tuple[float, float]      # (y, x) in 0-1000 normalized
    visible: bool
    method: str                       # "color" or "template"
    last_seen: float = field(default_factory=time.time)


@dataclass
class Zone:
    """A named rectangular zone on the table."""
    name: str
    bbox: tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax) in 0-1000
    on_enter: Callable | None = None
    on_exit: Callable | None = None


class ObjectTracker:
    """Track objects across video frames using color or template matching."""

    def __init__(self, frame_size: tuple[int, int] = (768, 768)):
        """
        Args:
            frame_size: (height, width) of input frames in pixels.
        """
        self._lock = threading.Lock()
        self._color_trackers: dict[str, dict] = {}  # name -> {histogram, window}
        self._template_trackers: dict[str, dict] = {}  # name -> {template}
        self._positions: dict[str, TrackedObject] = {}
        self._zones: dict[str, Zone] = {}
        self._zone_occupancy: dict[str, set[str]] = {}  # zone_name -> set of object names inside
        self.frame_size = frame_size  # (height, width)

    def track_color(self, name: str, initial_frame: np.ndarray,
                    region: tuple[int, int, int, int],
                    hsv_range: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        """Start tracking an object by its color histogram (CamShift).

        Args:
            name: Unique name for this tracked object.
            initial_frame: BGR frame containing the object.
            region: (y, x, h, w) bounding box of the object in the initial frame.
            hsv_range: Optional (lower, upper) HSV bounds. If None, auto-computed.
        """
        y, x, h, w = region
        roi = initial_frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        if hsv_range is not None:
            lower, upper = hsv_range
            mask = cv2.inRange(hsv_roi, lower, upper)
        else:
            # Default: track hues in the range 0-180, saturation > 60
            mask = cv2.inRange(hsv_roi,
                               np.array([0, 60, 32]),
                               np.array([180, 255, 255]))

        histogram = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)

        with self._lock:
            self._color_trackers[name] = {
                "histogram": histogram,
                "window": (x, y, w, h),  # CamShift uses (x, y, w, h)
            }

            center_y = y + h / 2
            center_x = x + w / 2
            self._positions[name] = TrackedObject(
                name=name,
                bbox=(y, x, h, w),
                center=self._normalize_position(center_y, center_x),
                visible=True,
                method="color",
            )

    def track_template(self, name: str, template: np.ndarray) -> None:
        """Start tracking an object by template matching.

        Args:
            name: Unique name for this tracked object.
            template: BGR image of the object to track.
        """
        with self._lock:
            self._template_trackers[name] = {
                "template": template.copy(),
            }
            # Initial position unknown until first update
            th, tw = template.shape[:2]
            self._positions[name] = TrackedObject(
                name=name,
                bbox=(0, 0, th, tw),
                center=(500.0, 500.0),  # center until first match
                visible=False,
                method="template",
            )

    def remove(self, name: str) -> bool:
        """Stop tracking an object. Returns True if it was tracked."""
        with self._lock:
            found = False
            if name in self._color_trackers:
                del self._color_trackers[name]
                found = True
            if name in self._template_trackers:
                del self._template_trackers[name]
                found = True
            if name in self._positions:
                del self._positions[name]
                found = True
            # Remove from zone occupancy
            for zone_objects in self._zone_occupancy.values():
                zone_objects.discard(name)
            return found

    def update(self, frame: np.ndarray) -> dict[str, TrackedObject]:
        """Process a new frame and update all tracked object positions.

        Returns current state of all tracked objects.
        """
        # Phase 1: update positions under lock.
        with self._lock:
            self._update_positions(frame)
            events = self._collect_zone_events()
            result = dict(self._positions)

        # Phase 2: fire callbacks WITHOUT holding the lock, so callbacks
        # can safely call get_object/get_all/add_zone etc.
        for event_type, obj_name, zone_name, callback in events:
            try:
                callback(obj_name, zone_name)
            except Exception:
                pass  # don't let a bad callback break tracking

        return result

    def _update_positions(self, frame: np.ndarray) -> None:
        """Update all tracker positions (caller holds lock)."""
        # Update color trackers (CamShift)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for name, tracker in self._color_trackers.items():
            back_proj = cv2.calcBackProject([hsv], [0], tracker["histogram"],
                                            [0, 180], 1)
            window = tracker["window"]
            ret, _ = cv2.CamShift(back_proj, window,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

            if ret[1][0] > 0 and ret[1][1] > 0:
                cx, cy = ret[0]
                w, h = ret[1]
                tracker["window"] = (int(cx - w/2), int(cy - h/2), int(w), int(h))
                px, py, pw, ph = tracker["window"]
                self._positions[name] = TrackedObject(
                    name=name,
                    bbox=(py, px, ph, pw),
                    center=self._normalize_position(cy, cx),
                    visible=True,
                    method="color",
                )
            else:
                if name in self._positions:
                    self._positions[name].visible = False

        # Update template trackers
        for name, tracker in self._template_trackers.items():
            template = tracker["template"]
            th, tw = template.shape[:2]
            fh, fw = frame.shape[:2]

            if th > fh or tw > fw:
                if name in self._positions:
                    self._positions[name].visible = False
                continue

            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= 0.5:
                x, y = max_loc
                cx = x + tw / 2
                cy = y + th / 2
                self._positions[name] = TrackedObject(
                    name=name,
                    bbox=(y, x, th, tw),
                    center=self._normalize_position(cy, cx),
                    visible=True,
                    method="template",
                )
            else:
                if name in self._positions:
                    self._positions[name].visible = False

    def _collect_zone_events(self) -> list[tuple[str, str, str, Callable]]:
        """Collect zone events with their callbacks (caller holds lock).

        Returns list of (event_type, obj_name, zone_name, callback) tuples.
        Callbacks are NOT called here — caller fires them after releasing the lock.
        """
        events = []
        for zone_name, zone in self._zones.items():
            currently_inside = self._zone_occupancy.get(zone_name, set())
            zymin, zxmin, zymax, zxmax = zone.bbox

            for obj_name, obj in self._positions.items():
                if not obj.visible:
                    if obj_name in currently_inside:
                        currently_inside.discard(obj_name)
                        if zone.on_exit:
                            events.append(("exit", obj_name, zone_name, zone.on_exit))
                    continue

                oy, ox = obj.center
                inside = (zymin <= oy <= zymax) and (zxmin <= ox <= zxmax)

                if inside and obj_name not in currently_inside:
                    currently_inside.add(obj_name)
                    if zone.on_enter:
                        events.append(("enter", obj_name, zone_name, zone.on_enter))
                elif not inside and obj_name in currently_inside:
                    currently_inside.discard(obj_name)
                    if zone.on_exit:
                        events.append(("exit", obj_name, zone_name, zone.on_exit))

            self._zone_occupancy[zone_name] = currently_inside

        return events

    def get_object(self, name: str) -> TrackedObject | None:
        with self._lock:
            return self._positions.get(name)

    def get_all(self) -> dict[str, TrackedObject]:
        with self._lock:
            return dict(self._positions)

    def add_zone(self, zone: Zone) -> None:
        with self._lock:
            self._zones[zone.name] = zone
            self._zone_occupancy.setdefault(zone.name, set())

    def remove_zone(self, name: str) -> bool:
        with self._lock:
            if name in self._zones:
                del self._zones[name]
                self._zone_occupancy.pop(name, None)
                return True
            return False

    def _normalize_position(self, pixel_y: float, pixel_x: float) -> tuple[float, float]:
        """Convert pixel coordinates to 0-1000 normalized coords."""
        h, w = self.frame_size
        ny = (pixel_y / h) * 1000.0 if h > 0 else 0.0
        nx = (pixel_x / w) * 1000.0 if w > 0 else 0.0
        return (round(ny, 1), round(nx, 1))

    @staticmethod
    def compute_color_histogram(frame: np.ndarray,
                                 region: tuple[int, int, int, int]) -> np.ndarray:
        """Compute HSV hue histogram for a region."""
        y, x, h, w = region
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 60, 32]), np.array([180, 255, 255]))
        hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def match_template(frame: np.ndarray,
                       template: np.ndarray) -> tuple[int, int, float]:
        """Find best template match. Returns (y, x, confidence)."""
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return (max_loc[1], max_loc[0], max_val)
