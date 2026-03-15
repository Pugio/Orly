"""Integration tests for the program runtime + overlay state + session + tracker pipeline.

Tests verify components working together end-to-end, without real Gemini
or WebSocket connections.
"""

import time

import cv2
import numpy as np
import pytest

from client.overlay_state import OverlayStateManager
from client.session_store import SessionStore
from client.object_tracker import ObjectTracker
from client.program_runtime import ProgramRuntime, TableAPI, validate_code


# ---------------------------------------------------------------------------
# Mock overlay manager (minimal shim for integration tests)
# ---------------------------------------------------------------------------


class MockOverlayManager:
    """Minimal OverlayManager for integration tests."""

    def __init__(self):
        self.proj_width = 640
        self.proj_height = 480
        self.mode = "screen"
        self._has_content = False
        self.canvas = self._make_bg()

    def _make_bg(self):
        return np.zeros((self.proj_height, self.proj_width, 3), dtype=np.uint8)

    def clear(self):
        self.canvas = self._make_bg()
        self._has_content = False

    def render_overlay(self, content_type, placement, title, data):
        ymin, xmin, ymax, xmax = placement
        w = max(1, int((xmax - xmin) / 1000.0 * self.proj_width))
        h = max(1, int((ymax - ymin) / 1000.0 * self.proj_height))
        return np.full((h, w, 3), (0, 255, 255), dtype=np.uint8)

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
    """Complete system with all components wired together."""
    om = MockOverlayManager()
    osm = OverlayStateManager(om)
    tracker = ObjectTracker(frame_size=(768, 768))
    session = SessionStore(session_dir=str(tmp_path / "session"))

    frame = np.zeros((768, 768, 3), dtype=np.uint8)
    notifications = []

    def make_api():
        return TableAPI(
            overlay_state_manager=osm,
            object_tracker=tracker,
            session_store=session,
            notify_fn=lambda msg: notifications.append(msg),
            get_frame_fn=lambda: frame.copy(),
        )

    runtime = ProgramRuntime(table_api_factory=make_api)

    return {
        "om": om,
        "osm": osm,
        "tracker": tracker,
        "session": session,
        "runtime": runtime,
        "notifications": notifications,
        "frame": frame,
        "make_api": make_api,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProgramPlacesOverlay:
    def test_program_places_overlay_via_api(self, system):
        """Run a program that calls table.place_overlay(), verify it appears."""
        runtime = system["runtime"]
        osm = system["osm"]
        om = system["om"]

        code = """
table.place_overlay("test_box", "annotation", [100, 100, 400, 400], {"text": "hello"})
"""
        status = runtime.run("placer", code)
        time.sleep(0.2)

        assert osm.get("test_box") is not None
        assert osm.get("test_box").content_type == "annotation"
        assert osm.get("test_box").placement == [100, 100, 400, 400]
        # Canvas should have non-zero pixels in the overlay region
        assert om._has_content is True


class TestProgramRemovesOverlay:
    def test_program_removes_overlay_via_api(self, system):
        """Place overlay, run program that removes it, verify gone."""
        runtime = system["runtime"]
        osm = system["osm"]

        # Place overlay first via direct API
        api = system["make_api"]()
        api.place_overlay("to_remove", "annotation", [0, 0, 200, 200], {"text": "bye"})
        assert osm.get("to_remove") is not None

        code = """
table.remove_overlay("to_remove")
"""
        runtime.run("remover", code)
        time.sleep(0.2)

        assert osm.get("to_remove") is None
        assert "to_remove" not in osm.list_names()


class TestProgramGetsOverlayState:
    def test_program_gets_overlay_state(self, system):
        """Place overlays, run program that queries state."""
        runtime = system["runtime"]
        osm = system["osm"]

        api = system["make_api"]()
        api.place_overlay("box_a", "graph", [0, 0, 300, 300], {"series": [1, 2, 3]})
        api.place_overlay("box_b", "annotation", [500, 500, 800, 800], {"text": "hi"})

        code = """
state = table.get_overlay_state()
table.notify(str(state["count"]))
for ov in state["overlays"]:
    table.notify(ov["name"])
"""
        runtime.run("state_reader", code)
        time.sleep(0.2)

        notes = system["notifications"]
        assert "2" in notes
        assert "box_a" in notes
        assert "box_b" in notes


class TestProgramSavesImage:
    def test_program_saves_image_to_session(self, system):
        """Run program that saves an image, verify in SessionStore."""
        runtime = system["runtime"]
        session = system["session"]

        code = """
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:, :] = (0, 0, 255)  # red
table.save_image("red_square", img)
"""
        runtime.run("saver", code)
        time.sleep(0.2)

        loaded = session.load_image("red_square")
        assert loaded is not None
        assert loaded.shape == (100, 100, 3)
        # Check red channel dominates
        assert loaded[50, 50, 2] > 200  # Red channel


class TestProgramLoadsImage:
    def test_program_loads_image_from_session(self, system):
        """Save image to session, run program that loads it."""
        runtime = system["runtime"]
        session = system["session"]

        # Save image directly
        original = np.full((64, 64, 3), (0, 200, 0), dtype=np.uint8)
        session.save_image("green_tile", original)

        code = """
img = table.load_image("green_tile")
if img is not None:
    table.notify(f"loaded:{img.shape[0]}x{img.shape[1]}")
else:
    table.notify("not_found")
"""
        runtime.run("loader", code)
        time.sleep(0.2)

        assert "loaded:64x64" in system["notifications"]


class TestProgramTracksTemplate:
    def test_program_tracks_template_and_gets_position(self, system):
        """Track a template pattern, update tracker, query from program."""
        tracker = system["tracker"]
        runtime = system["runtime"]

        # Create a distinctive template with pattern (uniform won't match)
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        cv2.circle(template, (15, 15), 12, (255, 255, 255), -1)
        cv2.circle(template, (15, 15), 6, (0, 0, 255), -1)

        # Create a frame with the template placed at a known location
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        # Place template at (200, 300)
        frame[200:230, 300:330] = template

        # Register template tracking
        tracker.track_template("marker", template)

        # Update tracker with the frame
        tracker.update(frame)

        code = """
info = table.get_tracked("marker")
if info and info["visible"]:
    table.notify(f"found:{info['center']}")
else:
    table.notify("not_visible")
"""
        runtime.run("tracker_query", code)
        time.sleep(0.2)

        notes = system["notifications"]
        assert len(notes) >= 1
        assert "found:" in notes[0]


class TestProgramFrameCallback:
    def test_program_frame_callback_processes_frames(self, system):
        """Program registers on_frame callback, verify it fires."""
        runtime = system["runtime"]

        code = """
count = [0]
def on_new_frame(f):
    count[0] += 1
    table.notify(f"frame:{count[0]}")

table.on_frame(on_new_frame)
# Keep alive briefly
import time
time.sleep(0.5)
"""
        runtime.run("frame_watcher", code)
        time.sleep(0.1)  # Let program start and register callback

        # Send frames
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        runtime.process_frame(frame)
        runtime.process_frame(frame)
        time.sleep(0.1)

        notes = system["notifications"]
        assert "frame:1" in notes
        assert "frame:2" in notes


class TestProgramNotificationFlow:
    def test_program_notification_flow(self, system):
        """Program calls table.notify(), verify captured."""
        runtime = system["runtime"]

        code = """
table.notify("hello from program")
table.notify("second message")
"""
        runtime.run("notifier", code)
        time.sleep(0.2)

        notes = system["notifications"]
        assert "hello from program" in notes
        assert "second message" in notes


class TestMultipleProgramsIndependent:
    def test_multiple_programs_independent(self, system):
        """Run two programs, stop one, verify the other still runs."""
        runtime = system["runtime"]

        code_a = """
import time
while not table.stopped:
    time.sleep(0.05)
table.notify("a_stopped")
"""
        code_b = """
import time
while not table.stopped:
    time.sleep(0.05)
table.notify("b_stopped")
"""
        runtime.run("prog_a", code_a)
        runtime.run("prog_b", code_b)
        time.sleep(0.1)

        # Both should be running
        status_a = runtime.get_status("prog_a")
        status_b = runtime.get_status("prog_b")
        assert status_a.state == "running"
        assert status_b.state == "running"

        # Stop only A
        runtime.stop("prog_a")
        time.sleep(0.3)

        status_a = runtime.get_status("prog_a")
        status_b = runtime.get_status("prog_b")
        assert status_a.state == "stopped"
        assert status_b.state == "running"

        # Cleanup
        runtime.stop("prog_b")
        time.sleep(0.2)


class TestProgramWithZoneTrigger:
    def test_program_with_zone_trigger(self, system):
        """Program adds zone, tracked object moves in, on_enter fires."""
        tracker = system["tracker"]
        runtime = system["runtime"]

        code = """
def on_enter(obj_name, zone_name):
    table.notify(f"enter:{obj_name}:{zone_name}")

table.add_zone("target_zone", (400, 400, 600, 600), on_enter=on_enter)

import time
time.sleep(1.0)
"""
        runtime.run("zone_test", code)
        time.sleep(0.1)  # Let program start

        # Create a distinctive template (uniform won't match reliably)
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        cv2.circle(template, (15, 15), 12, (255, 255, 255), -1)
        cv2.circle(template, (15, 15), 6, (0, 0, 255), -1)
        tracker.track_template("piece", template)

        # Frame with template OUTSIDE zone (top-left corner)
        frame1 = np.zeros((768, 768, 3), dtype=np.uint8)
        frame1[10:40, 10:40] = template
        tracker.update(frame1)

        # Frame with template INSIDE zone (center ~500,500 normalized = ~384 px)
        frame2 = np.zeros((768, 768, 3), dtype=np.uint8)
        py = int(500 / 1000 * 768) - 15
        px = int(500 / 1000 * 768) - 15
        frame2[py:py + 30, px:px + 30] = template
        tracker.update(frame2)

        time.sleep(0.1)
        runtime.stop("zone_test")
        time.sleep(0.2)

        notes = system["notifications"]
        assert any("enter:piece:target_zone" in n for n in notes)


class TestOverlayStateAsciiAfterProgram:
    def test_overlay_state_to_ascii_after_program_places(self, system):
        """Program places overlays, check ASCII grid shows them."""
        runtime = system["runtime"]
        osm = system["osm"]

        code = """
table.place_overlay("alpha", "annotation", [0, 0, 500, 500], {"text": "A"})
table.place_overlay("beta", "graph", [500, 500, 1000, 1000], {"series": [1]})
"""
        runtime.run("ascii_test", code)
        time.sleep(0.2)

        ascii_grid = osm.to_ascii()
        # 'a' should appear in top-left quadrant, 'b' in bottom-right
        lines = ascii_grid.split("\n")
        # Top-left cell should be 'a' (first char of "alpha")
        assert lines[0][0] == "a"
        # Bottom-right cell should be 'b' (first char of "beta")
        assert lines[-1][-1] == "b"
        # Center should be empty or not
        mid_row = len(lines) // 2
        mid_col = len(lines[0]) // 2
        # No overlaps expected
        assert "#" not in ascii_grid


class TestSessionStorePersistsAcrossPrograms:
    def test_session_store_persists_across_programs(self, system):
        """Program 1 saves image, program 2 loads it."""
        runtime = system["runtime"]

        code1 = """
img = np.full((50, 50, 3), 128, dtype=np.uint8)
table.save_image("shared_data", img)
table.notify("saved")
"""
        runtime.run("saver_prog", code1)
        time.sleep(0.2)
        assert "saved" in system["notifications"]

        code2 = """
img = table.load_image("shared_data")
if img is not None and img.shape == (50, 50, 3):
    table.notify("loaded_ok")
else:
    table.notify("load_failed")
"""
        runtime.run("loader_prog", code2)
        time.sleep(0.2)
        assert "loaded_ok" in system["notifications"]


class TestFullMusicalTableScenario:
    def test_full_musical_table_scenario(self, system):
        """Simplified musical table: place instrument overlays, add zones,
        move tracked piece into zones, verify enter callbacks."""
        runtime = system["runtime"]
        tracker = system["tracker"]

        code = """
# Place two instrument zones as overlays
table.place_overlay("piano", "annotation", [100, 100, 300, 400], {"text": "Piano"})
table.place_overlay("drums", "annotation", [100, 600, 300, 900], {"text": "Drums"})

# Add trigger zones aligned with overlays
def on_enter_piano(obj, zone):
    table.notify(f"play:piano:{obj}")

def on_enter_drums(obj, zone):
    table.notify(f"play:drums:{obj}")

table.add_zone("piano_zone", (100, 100, 300, 400), on_enter=on_enter_piano)
table.add_zone("drums_zone", (100, 600, 300, 900), on_enter=on_enter_drums)

import time
time.sleep(1.0)
"""
        runtime.run("music", code)
        time.sleep(0.1)

        # Track a "finger" template - needs distinctive pattern for matching
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        cv2.circle(template, (15, 15), 12, (255, 255, 255), -1)
        cv2.circle(template, (15, 15), 6, (0, 0, 255), -1)
        tracker.track_template("finger", template)

        # Move finger to piano zone (center ~200,250 normalized -> pixels)
        frame_piano = np.zeros((768, 768, 3), dtype=np.uint8)
        py = int(200 / 1000 * 768) - 15
        px = int(250 / 1000 * 768) - 15
        frame_piano[py:py + 30, px:px + 30] = template
        tracker.update(frame_piano)
        time.sleep(0.05)

        # Move finger to drums zone (center ~200,750 normalized)
        frame_drums = np.zeros((768, 768, 3), dtype=np.uint8)
        py = int(200 / 1000 * 768) - 15
        px = int(750 / 1000 * 768) - 15
        frame_drums[py:py + 30, px:px + 30] = template
        tracker.update(frame_drums)
        time.sleep(0.05)

        runtime.stop("music")
        time.sleep(0.2)

        notes = system["notifications"]
        assert any("play:piano:finger" in n for n in notes)
        assert any("play:drums:finger" in n for n in notes)


class TestProgramErrorDoesntBreakOthers:
    def test_program_error_doesnt_break_others(self, system):
        """Program with error doesn't break a subsequent program."""
        runtime = system["runtime"]

        code_bad = """
raise ValueError("intentional crash")
"""
        runtime.run("bad_prog", code_bad)
        time.sleep(0.2)

        status_bad = runtime.get_status("bad_prog")
        assert status_bad.state == "error"
        assert "ValueError" in status_bad.error

        # Now run a good program
        code_good = """
table.notify("i_am_fine")
"""
        runtime.run("good_prog", code_good)
        time.sleep(0.2)

        assert "i_am_fine" in system["notifications"]
        status_good = runtime.get_status("good_prog")
        assert status_good.state == "stopped"  # finished normally


class TestClearOverlaysResetsState:
    def test_clear_overlays_resets_state(self, system):
        """Place overlays, clear via program, verify state empty."""
        runtime = system["runtime"]
        osm = system["osm"]

        api = system["make_api"]()
        api.place_overlay("x1", "annotation", [0, 0, 100, 100], {"text": "1"})
        api.place_overlay("x2", "annotation", [200, 200, 300, 300], {"text": "2"})
        assert len(osm.list_names()) == 2

        code = """
table.clear_overlays()
table.notify(f"remaining:{len(table.get_overlay_state()['overlays'])}")
"""
        runtime.run("clearer", code)
        time.sleep(0.2)

        assert len(osm.list_names()) == 0
        assert "remaining:0" in system["notifications"]
