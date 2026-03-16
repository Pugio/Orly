"""Tests for the paint program integration — TableAPI canvas, color detection,
object tracking, and the full paint workflow."""

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


# ---------------------------------------------------------------------------
# Mock overlay manager (same as test_integration_programs)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def system(tmp_path):
    om = MockOverlayManager()
    osm = OverlayStateManager(om)
    tracker = ObjectTracker(frame_size=(768, 768))
    session = SessionStore(session_dir=str(tmp_path / "session"))

    # Default frame: dark background with a bright red object at center
    frame = np.zeros((768, 768, 3), dtype=np.uint8)
    _current_frame = [frame]
    notifications = []

    def make_api():
        return TableAPI(
            overlay_state_manager=osm,
            object_tracker=tracker,
            session_store=session,
            notify_fn=lambda msg: notifications.append(msg),
            get_frame_fn=lambda: _current_frame[0].copy(),
        )

    runtime = ProgramRuntime(table_api_factory=make_api)
    runtime._object_tracker = tracker

    return {
        "om": om,
        "osm": osm,
        "tracker": tracker,
        "session": session,
        "runtime": runtime,
        "notifications": notifications,
        "frame": frame,
        "set_frame": lambda f: _current_frame.__setitem__(0, f),
        "make_api": make_api,
    }


# ---------------------------------------------------------------------------
# TableAPI.create_canvas tests
# ---------------------------------------------------------------------------


class TestCreateCanvas:
    def test_create_canvas_returns_paint_canvas(self, system):
        api = system["make_api"]()
        canvas = api.create_canvas()
        assert isinstance(canvas, PaintCanvas)
        assert canvas.width == 640
        assert canvas.height == 480

    def test_canvas_composites_onto_overlay_manager(self, system):
        """Paint canvas content appears on the overlay manager canvas after process_frame."""
        runtime = system["runtime"]
        om = system["om"]

        code = """
canvas = table.create_canvas()
canvas.rectangle(0, 0, 1000, 1000, (0, 0, 255))  # fill red
import time
time.sleep(0.5)
"""
        runtime.run("paint_test", code)
        time.sleep(0.1)

        # Process a frame to trigger compositing
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        runtime.process_frame(frame)

        # OM canvas should have red pixels
        assert om.canvas[240, 320, 2] > 200

    def test_canvas_cleared_when_program_stops(self, system):
        """After program stops, its canvas no longer composites."""
        runtime = system["runtime"]
        om = system["om"]

        code = """
canvas = table.create_canvas()
canvas.rectangle(0, 0, 1000, 1000, (0, 0, 255))
# Exit immediately
"""
        runtime.run("short_paint", code)
        time.sleep(0.2)

        # Program should have stopped
        status = runtime.get_status("short_paint")
        assert status.state == "stopped"

        # Reset canvas to black
        om.canvas = om._make_bg()

        # Process frame — stopped program's canvas should NOT composite
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        runtime.process_frame(frame)

        # Canvas should still be black
        assert not np.any(om.canvas > 0)


# ---------------------------------------------------------------------------
# TableAPI.get_dominant_color tests
# ---------------------------------------------------------------------------


class TestGetDominantColor:
    def test_dominant_color_red_object(self, system):
        """Detects dominant color from a region with a red object."""
        api = system["make_api"]()

        # Create frame with red square at center
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        frame[300:500, 300:500] = (0, 0, 255)  # red in BGR
        system["set_frame"](frame)

        color = api.get_dominant_color(region=(300, 300, 200, 200))
        assert color is not None
        b, g, r = color
        assert r > 200  # should be predominantly red
        assert g < 50
        assert b < 50

    def test_dominant_color_blue_object(self, system):
        api = system["make_api"]()

        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        frame[100:300, 100:300] = (255, 0, 0)  # blue in BGR
        system["set_frame"](frame)

        color = api.get_dominant_color(region=(100, 100, 200, 200))
        assert color is not None
        b, g, r = color
        assert b > 200

    def test_dominant_color_full_frame(self, system):
        api = system["make_api"]()

        frame = np.full((768, 768, 3), (0, 255, 0), dtype=np.uint8)  # green
        system["set_frame"](frame)

        color = api.get_dominant_color()
        assert color is not None
        b, g, r = color
        assert g > 200

    def test_dominant_color_no_frame(self, system):
        """Returns None when no frame is available."""
        api = system["make_api"]()
        system["set_frame"](None)

        # Need to override get_frame to return None
        api._get_frame_fn = lambda: None
        color = api.get_dominant_color()
        assert color is None


# ---------------------------------------------------------------------------
# TableAPI.get_object_size tests
# ---------------------------------------------------------------------------


class TestGetObjectSize:
    def test_get_object_size_from_tracker(self, system):
        tracker = system["tracker"]
        api = system["make_api"]()

        # Create template and track it
        template = np.zeros((40, 60, 3), dtype=np.uint8)
        cv2.circle(template, (30, 20), 15, (255, 255, 255), -1)
        tracker.track_template("obj1", template)

        # Place template in a frame and update
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        frame[200:240, 300:360] = template
        tracker.update(frame)

        size = api.get_object_size("obj1")
        assert size is not None
        h, w = size
        assert h > 0
        assert w > 0
        # 40px / 768px * 1000 ≈ 52
        assert 40 < h < 70
        # 60px / 768px * 1000 ≈ 78
        assert 60 < w < 100

    def test_get_object_size_not_found(self, system):
        api = system["make_api"]()
        assert api.get_object_size("nonexistent") is None


# ---------------------------------------------------------------------------
# TableAPI.init_color_tracking tests
# ---------------------------------------------------------------------------


class TestInitColorTracking:
    def test_init_color_tracking_starts_tracker_and_returns_color(self, system):
        api = system["make_api"]()

        # Frame with a green object
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        frame[200:300, 200:300] = (0, 200, 0)  # green
        system["set_frame"](frame)

        color = api.init_color_tracking("green_obj", (200, 200, 100, 100))
        assert color is not None
        b, g, r = color
        assert g > 150

        # Tracker should now have the object
        obj = api.get_tracked("green_obj")
        assert obj is not None
        assert obj["visible"] is True


# ---------------------------------------------------------------------------
# TableAPI.wait_for_object_in_region tests
# ---------------------------------------------------------------------------


class TestCaptureBaseline:
    def test_capture_baseline_returns_grayscale_roi(self, system):
        api = system["make_api"]()
        frame = np.full((768, 768, 3), 100, dtype=np.uint8)
        system["set_frame"](frame)

        baseline = api.capture_baseline(region=(300, 300, 100, 100), settle_time=0.05)
        assert baseline is not None
        assert baseline.shape == (100, 100)
        assert baseline.dtype == np.uint8

    def test_capture_baseline_full_frame(self, system):
        api = system["make_api"]()
        frame = np.full((768, 768, 3), 100, dtype=np.uint8)
        system["set_frame"](frame)

        baseline = api.capture_baseline(settle_time=0.05)
        assert baseline is not None
        assert baseline.shape == (768, 768)

    def test_capture_baseline_hides_canvases(self, system):
        api = system["make_api"]()
        canvas = api.create_canvas()
        canvas.rectangle(0, 0, 1000, 1000, (255, 255, 255))
        assert canvas.visible is True

        frame = np.full((768, 768, 3), 100, dtype=np.uint8)
        system["set_frame"](frame)

        # During capture, canvases should be hidden (we can't easily test
        # the exact moment, but after capture they should be visible again)
        baseline = api.capture_baseline(settle_time=0.05)
        assert canvas.visible is True
        assert baseline is not None


class TestWaitForObjectInRegion:
    def test_detects_object_via_change_from_baseline(self, system):
        api = system["make_api"]()

        # Baseline: empty region
        baseline = np.zeros((100, 100), dtype=np.uint8)

        # Frame with bright object placed in region
        frame_with_obj = np.zeros((768, 768, 3), dtype=np.uint8)
        frame_with_obj[350:420, 350:420] = (255, 255, 255)

        frames = [np.zeros((768, 768, 3), dtype=np.uint8)]

        def delayed_place():
            time.sleep(0.2)
            frames[0] = frame_with_obj

        api._get_frame_fn = lambda: frames[0].copy()
        threading.Thread(target=delayed_place, daemon=True).start()

        result = api.wait_for_object_in_region((340, 340, 100, 100),
                                                timeout=2.0,
                                                check_interval=0.1,
                                                baseline=baseline)
        assert result is True

    def test_no_false_trigger_on_identical_scene(self, system):
        """If scene hasn't changed from baseline, should not trigger."""
        api = system["make_api"]()

        frame = np.full((768, 768, 3), 80, dtype=np.uint8)
        api._get_frame_fn = lambda: frame.copy()

        # Baseline matches the frame
        baseline = np.full((100, 100), 80, dtype=np.uint8)

        result = api.wait_for_object_in_region((300, 300, 100, 100),
                                                timeout=0.5,
                                                check_interval=0.1,
                                                baseline=baseline)
        assert result is False

    def test_timeout_if_no_object(self, system):
        api = system["make_api"]()
        frame_empty = np.zeros((768, 768, 3), dtype=np.uint8)
        api._get_frame_fn = lambda: frame_empty.copy()

        baseline = np.zeros((100, 100), dtype=np.uint8)
        result = api.wait_for_object_in_region((300, 300, 100, 100),
                                                timeout=0.5,
                                                check_interval=0.1,
                                                baseline=baseline)
        assert result is False


# ---------------------------------------------------------------------------
# TableAPI.wait_for_hands_clear tests
# ---------------------------------------------------------------------------


class TestWaitForHandsClear:
    def test_detects_stable_region(self, system):
        api = system["make_api"]()

        # Consistent frame = stable
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        frame[300:400, 300:400] = (0, 0, 200)
        api._get_frame_fn = lambda: frame.copy()

        result = api.wait_for_hands_clear((300, 300, 100, 100),
                                           timeout=3.0,
                                           stable_time=0.3,
                                           check_interval=0.05)
        assert result is True

    def test_timeout_if_always_moving(self, system):
        api = system["make_api"]()
        counter = [0]

        def moving_frame():
            counter[0] += 1
            frame = np.zeros((768, 768, 3), dtype=np.uint8)
            # Different content each time
            val = (counter[0] * 50) % 256
            frame[300:400, 300:400] = (val, val, val)
            return frame

        api._get_frame_fn = moving_frame

        result = api.wait_for_hands_clear((300, 300, 100, 100),
                                           timeout=0.5,
                                           stable_time=0.3,
                                           check_interval=0.05)
        assert result is False


# ---------------------------------------------------------------------------
# TableAPI.play_tone tests
# ---------------------------------------------------------------------------


class TestPlayTone:
    def test_play_tone_without_audio_player_logs(self, system):
        api = system["make_api"]()
        # No audio player set — should log, not crash
        api.play_tone(440, 0.1)
        assert any("play_tone" in msg for msg in api._log_messages)

    def test_play_tone_with_mock_player(self, system):
        api = system["make_api"]()
        played_data = []

        class MockPlayer:
            def play(self, data):
                played_data.append(data)

        api._audio_player = MockPlayer()
        api.play_tone(440, 0.1)

        assert len(played_data) == 1
        # 16kHz * 0.1s = 1600 samples * 2 bytes = 3200 bytes
        assert len(played_data[0]) == 3200


# ---------------------------------------------------------------------------
# Full paint program integration test
# ---------------------------------------------------------------------------


class TestPaintProgramIntegration:
    def test_paint_program_draws_on_canvas(self, system):
        """A mini-program that creates a canvas and stamps circles paints onto OM canvas."""
        runtime = system["runtime"]
        om = system["om"]

        code = """
canvas = table.create_canvas()
# Simulate painting: stamp along a line
for i in range(5):
    y = 500
    x = 200 + i * 150
    canvas.stamp(y, x, 20, (0, 255, 0))  # green dots

table.notify("painted")
import time
time.sleep(0.3)
"""
        runtime.run("painter", code)
        time.sleep(0.1)

        # Process frame to trigger compositing
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        runtime.process_frame(frame)

        assert "painted" in system["notifications"]
        # Check OM canvas has green pixels
        assert np.any(om.canvas[:, :, 1] > 200)

    def test_paint_program_validates(self):
        """Paint program code passes validation."""
        code = """
canvas = table.create_canvas()
canvas.circle(500, 500, 50, (255, 0, 0))
canvas.stamp(300, 300, 20, (0, 255, 0))
canvas.clear()
table.notify("done")
"""
        valid, error = validate_code(code)
        assert valid, f"Validation failed: {error}"


class TestPaintCanvasCompositing:
    def test_multiple_programs_multiple_canvases(self, system):
        """Two programs with canvases both composite onto OM."""
        runtime = system["runtime"]
        om = system["om"]

        code_a = """
canvas = table.create_canvas()
canvas.rectangle(0, 0, 500, 500, (255, 0, 0))  # blue top-left
import time
time.sleep(0.5)
"""
        code_b = """
canvas = table.create_canvas()
canvas.rectangle(500, 500, 1000, 1000, (0, 0, 255))  # red bottom-right
import time
time.sleep(0.5)
"""
        runtime.run("prog_a", code_a)
        runtime.run("prog_b", code_b)
        time.sleep(0.1)

        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        runtime.process_frame(frame)

        # Top-left should be blue
        assert om.canvas[100, 100, 0] > 200
        # Bottom-right should be red
        assert om.canvas[400, 500, 2] > 200

        runtime.stop_all()
        time.sleep(0.2)
