"""End-to-end test for the paint program flow.

Simulates the full lifecycle: circle display → object detection →
hand removal → tracking init → painting → object disappearance → cleanup.
"""

import time
import threading

import cv2
import numpy as np
import pytest

from client.overlay_state import OverlayStateManager
from client.session_store import SessionStore
from client.object_tracker import ObjectTracker
from client.program_runtime import ProgramRuntime, TableAPI, validate_code
from client.paint_canvas import PaintCanvas


class MockOverlayManager:
    def __init__(self):
        self.proj_width = 640
        self.proj_height = 480
        self.mode = "screen"
        self._has_content = False
        self.canvas = self._make_bg()
        self._saved_canvas = None

    def _make_bg(self):
        return np.zeros((self.proj_height, self.proj_width, 3), dtype=np.uint8)

    def clear(self):
        self.canvas = self._make_bg()
        self._has_content = False

    def request_refresh(self):
        self._saved_canvas = self.canvas.copy()
        self.canvas = self._make_bg()

    def complete_refresh(self):
        if self._saved_canvas is not None:
            self.canvas = self._saved_canvas
            self._saved_canvas = None

    def render_overlay(self, content_type, placement, title, data):
        ymin, xmin, ymax, xmax = placement
        w = max(1, int((xmax - xmin) / 1000.0 * self.proj_width))
        h = max(1, int((ymax - ymin) / 1000.0 * self.proj_height))
        return np.full((h, w, 3), (0, 255, 255), dtype=np.uint8)

    class _MockTransform:
        def orient_overlay(self, overlay):
            return overlay

    transform = _MockTransform()

    def place_on_canvas(self, overlay, placement):
        canvas = self.canvas.copy()
        ymin, xmin, ymax, xmax = placement
        px1 = max(0, int(xmin / 1000 * self.proj_width))
        py1 = max(0, int(ymin / 1000 * self.proj_height))
        px2 = min(self.proj_width, int(xmax / 1000 * self.proj_width))
        py2 = min(self.proj_height, int(ymax / 1000 * self.proj_height))
        if px2 > px1 and py2 > py1:
            resized = cv2.resize(overlay, (px2 - px1, py2 - py1))
            canvas[py1:py2, px1:px2] = resized
        return canvas


@pytest.fixture
def paint_system(tmp_path):
    om = MockOverlayManager()
    osm = OverlayStateManager(om)
    tracker = ObjectTracker(frame_size=(768, 768))
    session = SessionStore(session_dir=str(tmp_path / "session"))

    # Frame sequence: empty -> object placed -> object stable -> object moves
    current_frame = [np.zeros((768, 768, 3), dtype=np.uint8)]
    notifications = []

    def make_api():
        return TableAPI(
            overlay_state_manager=osm,
            object_tracker=tracker,
            session_store=session,
            notify_fn=lambda msg: notifications.append(msg),
            get_frame_fn=lambda: current_frame[0].copy(),
        )

    runtime = ProgramRuntime(table_api_factory=make_api)
    runtime._object_tracker = tracker

    return {
        "om": om, "osm": osm, "tracker": tracker, "session": session,
        "runtime": runtime, "notifications": notifications,
        "set_frame": lambda f: current_frame.__setitem__(0, f),
        "make_api": make_api,
    }


class TestPaintProgramValidates:
    def test_paint_program_code_validates(self):
        """The paint.py program passes code validation."""
        with open("programs/paint.py") as f:
            code = f.read()
        valid, error = validate_code(code)
        assert valid, f"Validation failed: {error}"


class TestPaintProgramSimulatedFlow:
    def test_object_detection_and_tracking_init(self, paint_system):
        """Simulate: place bright object at center, hands clear, tracking starts."""
        runtime = paint_system["runtime"]
        notifications = paint_system["notifications"]
        set_frame = paint_system["set_frame"]

        # Simplified paint code that tests the detection + tracking init phase.
        # Uses an explicit baseline so we don't need the canvas-hide settle time.
        code = """
canvas = table.create_canvas()
frame = table.get_frame()
fh, fw = frame.shape[:2]
region_size = int(min(fh, fw) * 60 / 500)
ry = fh // 2 - region_size // 2
rx = fw // 2 - region_size // 2
region = (ry, rx, region_size, region_size)

canvas.circle(500, 500, 60, (0, 255, 255), thickness=4)
table.notify("circle_shown")

# Capture baseline from current (empty) frame
baseline = table.capture_baseline(region, settle_time=0.05)
table.notify("baseline_captured")

detected = table.wait_for_object_in_region(
    region, timeout=3.0, check_interval=0.05, baseline=baseline
)

if detected:
    table.notify("object_detected")
    hands_clear = table.wait_for_hands_clear(
        region, timeout=3.0, stable_time=0.2, check_interval=0.05
    )
    if hands_clear:
        table.notify("hands_clear")
        color = table.init_color_tracking("test_obj", region)
        if color:
            table.notify(f"tracking_started:color={color}")
        canvas.clear()
        table.notify("ready_to_paint")
else:
    table.notify("timeout")
"""
        # Start with empty frame (this becomes the baseline)
        set_frame(np.zeros((768, 768, 3), dtype=np.uint8))

        runtime.run("paint_detect", code)
        time.sleep(0.5)

        assert "circle_shown" in notifications
        assert "baseline_captured" in notifications

        # Now place a bright red object at center — triggers change detection
        frame_with_obj = np.zeros((768, 768, 3), dtype=np.uint8)
        cv2.circle(frame_with_obj, (384, 384), 40, (0, 0, 255), -1)  # red circle
        set_frame(frame_with_obj)

        # Wait for detection + hands clear + tracking
        time.sleep(3.0)

        assert "object_detected" in notifications
        assert "hands_clear" in notifications
        assert any("tracking_started" in n for n in notifications)
        assert "ready_to_paint" in notifications

        runtime.stop("paint_detect")
        time.sleep(0.2)

    def test_painting_stamps_on_canvas(self, paint_system):
        """After tracking is init'd, moving object stamps paint on canvas."""
        runtime = paint_system["runtime"]
        om = paint_system["om"]
        tracker = paint_system["tracker"]
        set_frame = paint_system["set_frame"]
        notifications = paint_system["notifications"]

        # Create a frame with a trackable object and init tracking manually
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        cv2.circle(frame, (384, 384), 30, (0, 200, 0), -1)  # green circle at center
        set_frame(frame)

        # Init tracking via the tracker directly
        tracker.track_color("brush", frame, (354, 354, 60, 60))

        code = """
canvas = table.create_canvas()
paint_color = (0, 200, 0)

# Simulate 5 frames of painting
for i in range(5):
    info = table.get_tracked("brush")
    if info and info["visible"]:
        cy, cx = info["center"]
        canvas.stamp(cy, cx, 15, paint_color)
        table.notify(f"stamp:{cy:.0f},{cx:.0f}")
    time.sleep(0.05)

table.notify("paint_done")

# Stay alive so canvas composites
while not table.stopped:
    time.sleep(0.05)
"""
        runtime.run("stamp_test", code)
        time.sleep(1.0)

        assert "paint_done" in notifications

        # Process frame to composite (program still running)
        runtime.process_frame(frame)

        stamp_msgs = [n for n in notifications if n.startswith("stamp:")]
        assert len(stamp_msgs) >= 1

        # Canvas should have green pixels
        assert np.any(om.canvas[:, :, 1] > 100)

        runtime.stop("stamp_test")
        time.sleep(0.2)

    def test_countdown_and_cleanup(self, paint_system):
        """When object disappears, countdown shows and canvas clears."""
        runtime = paint_system["runtime"]
        notifications = paint_system["notifications"]
        set_frame = paint_system["set_frame"]

        code = """
canvas = table.create_canvas()
canvas.stamp(500, 500, 30, (0, 255, 0))  # paint something

# Simulate: object immediately not visible, with short timeouts
MISSING_TIMEOUT = 0.3
COUNTDOWN_SECONDS = 0.5

last_visible_time = time.time() - MISSING_TIMEOUT - 0.1  # already past timeout
countdown_active = False
countdown_start = None

while not table.stopped:
    # Object is never visible in this test
    elapsed_missing = time.time() - last_visible_time

    if elapsed_missing >= MISSING_TIMEOUT and not countdown_active:
        countdown_active = True
        countdown_start = time.time()
        table.notify("countdown_started")

    if countdown_active:
        elapsed_countdown = time.time() - countdown_start
        remaining = COUNTDOWN_SECONDS - elapsed_countdown
        if remaining <= 0:
            canvas.clear()
            table.notify("canvas_cleared")
            table.stop()
            break

    time.sleep(0.05)
"""
        runtime.run("cleanup_test", code)
        time.sleep(2.0)

        assert "countdown_started" in notifications
        assert "canvas_cleared" in notifications

        status = runtime.get_status("cleanup_test")
        assert status.state == "stopped"
