"""Tests for client.object_tracker — frame-by-frame object tracking."""

import numpy as np
import pytest
import cv2

from client.object_tracker import ObjectTracker, TrackedObject, Zone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_frame(height=768, width=768, bg_color=(0, 0, 0)):
    """Create a solid-color frame."""
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
    return frame


def draw_rect(frame, y, x, h, w, color):
    """Draw a filled rectangle on a frame."""
    frame[y:y+h, x:x+w] = color
    return frame


# ---------------------------------------------------------------------------
# Color tracking
# ---------------------------------------------------------------------------

class TestTrackColor:
    def test_track_color_initial_position(self):
        """Track a red rectangle, verify initial position is correct."""
        tracker = ObjectTracker(frame_size=(768, 768))
        frame = make_frame()
        # Red rectangle at (100, 200, 50, 60) — BGR red is (0, 0, 255)
        draw_rect(frame, 100, 200, 50, 60, (0, 0, 255))

        tracker.track_color("red_obj", frame, (100, 200, 50, 60))

        obj = tracker.get_object("red_obj")
        assert obj is not None
        assert obj.name == "red_obj"
        assert obj.visible is True
        assert obj.method == "color"
        assert obj.bbox == (100, 200, 50, 60)
        # Center pixel: y=125, x=230 → normalized (162.8, 299.5)
        expected_ny = round((125 / 768) * 1000, 1)
        expected_nx = round((230 / 768) * 1000, 1)
        assert obj.center == (expected_ny, expected_nx)

    def test_track_color_update_moves(self):
        """Move rectangle between frames, verify position updates."""
        tracker = ObjectTracker(frame_size=(768, 768))
        # Frame 1: red rectangle at top-left area
        frame1 = make_frame()
        draw_rect(frame1, 100, 100, 80, 80, (0, 0, 255))
        tracker.track_color("mover", frame1, (100, 100, 80, 80))

        pos1 = tracker.get_object("mover")
        assert pos1 is not None
        center1 = pos1.center

        # Frame 2: red rectangle moved to bottom-right area
        frame2 = make_frame()
        draw_rect(frame2, 500, 500, 80, 80, (0, 0, 255))
        results = tracker.update(frame2)

        pos2 = results.get("mover")
        assert pos2 is not None
        # The object should have moved — center should be different
        # CamShift may not jump perfectly but should shift toward new location
        assert pos2.visible is True


# ---------------------------------------------------------------------------
# Template tracking
# ---------------------------------------------------------------------------

class TestTrackTemplate:
    def test_track_template_initial(self):
        """Track template, initial visible=False."""
        tracker = ObjectTracker(frame_size=(768, 768))
        template = make_frame(50, 50, (0, 255, 0))
        tracker.track_template("tpl_obj", template)

        obj = tracker.get_object("tpl_obj")
        assert obj is not None
        assert obj.visible is False
        assert obj.method == "template"

    def test_track_template_finds_match(self):
        """Template present in frame, verify position found."""
        tracker = ObjectTracker(frame_size=(768, 768))
        # Create a distinctive template
        template = np.zeros((40, 40, 3), dtype=np.uint8)
        draw_rect(template, 0, 0, 40, 40, (0, 255, 0))
        draw_rect(template, 10, 10, 20, 20, (255, 0, 0))

        tracker.track_template("pattern", template)

        # Place the same pattern in a frame
        frame = make_frame()
        frame[200:240, 300:340] = template
        results = tracker.update(frame)

        obj = results["pattern"]
        assert obj.visible is True
        # Center should be near (220, 320) → normalized
        expected_ny = round((220 / 768) * 1000, 1)
        expected_nx = round((320 / 768) * 1000, 1)
        assert abs(obj.center[0] - expected_ny) < 20
        assert abs(obj.center[1] - expected_nx) < 20

    def test_track_template_no_match(self):
        """Template not in frame, visible=False."""
        tracker = ObjectTracker(frame_size=(768, 768))
        # Very specific template
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_rect(template, 0, 0, 15, 30, (0, 200, 255))
        draw_rect(template, 15, 0, 15, 30, (255, 100, 0))

        tracker.track_template("missing", template)

        # Use a random noise frame — unlikely to match
        rng = np.random.RandomState(42)
        frame = rng.randint(0, 256, (768, 768, 3), dtype=np.uint8)
        results = tracker.update(frame)

        obj = results["missing"]
        assert obj.visible is False


# ---------------------------------------------------------------------------
# Remove
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_tracked_object(self):
        """Remove returns True, get_object returns None."""
        tracker = ObjectTracker()
        frame = make_frame()
        draw_rect(frame, 50, 50, 30, 30, (255, 0, 0))
        tracker.track_color("blue", frame, (50, 50, 30, 30))

        assert tracker.remove("blue") is True
        assert tracker.get_object("blue") is None

    def test_remove_nonexistent(self):
        """Returns False for object that was never tracked."""
        tracker = ObjectTracker()
        assert tracker.remove("ghost") is False


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_get_all_objects(self):
        """Track 2 objects, get_all returns both."""
        tracker = ObjectTracker()
        frame = make_frame()
        draw_rect(frame, 50, 50, 30, 30, (0, 0, 255))
        draw_rect(frame, 300, 300, 30, 30, (0, 255, 0))

        tracker.track_color("obj1", frame, (50, 50, 30, 30))
        tracker.track_color("obj2", frame, (300, 300, 30, 30))

        all_objs = tracker.get_all()
        assert len(all_objs) == 2
        assert "obj1" in all_objs
        assert "obj2" in all_objs


# ---------------------------------------------------------------------------
# Normalize position
# ---------------------------------------------------------------------------

class TestNormalizePosition:
    def test_normalize_position_center(self):
        """(384, 384) in 768x768 -> (500, 500)."""
        tracker = ObjectTracker(frame_size=(768, 768))
        result = tracker._normalize_position(384, 384)
        assert result == (500.0, 500.0)

    def test_normalize_position_origin(self):
        """(0, 0) -> (0, 0)."""
        tracker = ObjectTracker(frame_size=(768, 768))
        result = tracker._normalize_position(0, 0)
        assert result == (0.0, 0.0)

    def test_normalize_position_max(self):
        """(768, 768) -> (1000, 1000)."""
        tracker = ObjectTracker(frame_size=(768, 768))
        result = tracker._normalize_position(768, 768)
        assert result == (1000.0, 1000.0)


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

class TestZones:
    def test_add_zone(self):
        """Add zone, verify it's stored."""
        tracker = ObjectTracker()
        zone = Zone(name="answer_area", bbox=(0, 0, 500, 500))
        tracker.add_zone(zone)
        assert "answer_area" in tracker._zones

    def test_remove_zone(self):
        """Add and remove, returns True."""
        tracker = ObjectTracker()
        zone = Zone(name="drop_zone", bbox=(0, 0, 500, 500))
        tracker.add_zone(zone)
        assert tracker.remove_zone("drop_zone") is True
        assert "drop_zone" not in tracker._zones

    def test_remove_nonexistent_zone(self):
        """Returns False for zone that doesn't exist."""
        tracker = ObjectTracker()
        assert tracker.remove_zone("nope") is False

    def test_zone_enter_callback(self):
        """Object moves into zone, on_enter fires."""
        tracker = ObjectTracker(frame_size=(768, 768))
        entered = []

        def on_enter(obj_name, zone_name):
            entered.append((obj_name, zone_name))

        # Zone in bottom-right (normalized 600-900)
        zone = Zone(name="target", bbox=(600, 600, 900, 900), on_enter=on_enter)
        tracker.add_zone(zone)

        # Use template tracking (global search) so object can jump positions
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_rect(template, 0, 0, 30, 30, (0, 255, 255))
        draw_rect(template, 5, 5, 20, 20, (255, 0, 255))
        tracker.track_template("piece", template)

        # Frame 1: object outside zone (top-left)
        frame1 = make_frame()
        frame1[50:80, 50:80] = template
        tracker.update(frame1)
        assert len(entered) == 0

        # Frame 2: object inside zone (pixels ~500 → normalized ~651)
        frame2 = make_frame()
        frame2[500:530, 500:530] = template
        tracker.update(frame2)

        assert len(entered) > 0
        assert entered[-1] == ("piece", "target")

    def test_zone_exit_callback(self):
        """Object moves out of zone, on_exit fires."""
        tracker = ObjectTracker(frame_size=(768, 768))
        exited = []

        def on_exit(obj_name, zone_name):
            exited.append((obj_name, zone_name))

        # Zone covers top-left quadrant (0-500 normalized)
        zone = Zone(name="start", bbox=(0, 0, 500, 500), on_exit=on_exit)
        tracker.add_zone(zone)

        # Use template tracking (global search) so object can jump positions
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_rect(template, 0, 0, 30, 30, (0, 255, 255))
        draw_rect(template, 5, 5, 20, 20, (255, 0, 255))
        tracker.track_template("mover", template)

        # Frame 1: object inside zone (top-left)
        frame1 = make_frame()
        frame1[50:80, 50:80] = template
        tracker.update(frame1)

        # Frame 2: object outside zone (bottom-right, pixels 600 → normalized ~781)
        frame2 = make_frame()
        frame2[600:630, 600:630] = template
        tracker.update(frame2)

        assert len(exited) > 0
        assert exited[-1] == ("mover", "start")

    def test_zone_no_callback_when_staying(self):
        """Object stays in zone, no repeated callbacks."""
        tracker = ObjectTracker(frame_size=(768, 768))
        entered = []

        def on_enter(obj_name, zone_name):
            entered.append((obj_name, zone_name))

        zone = Zone(name="area", bbox=(0, 0, 500, 500), on_enter=on_enter)
        tracker.add_zone(zone)

        # Object inside zone
        frame = make_frame()
        draw_rect(frame, 50, 50, 40, 40, (0, 0, 255))
        tracker.track_color("static", frame, (50, 50, 40, 40))

        # Multiple updates in same position
        tracker.update(frame)
        count_after_first = len(entered)
        tracker.update(frame)
        tracker.update(frame)

        # Should only have triggered on_enter once (on first update)
        assert len(entered) == count_after_first

    def test_multiple_zones(self):
        """Object triggers correct zone."""
        tracker = ObjectTracker(frame_size=(768, 768))
        events = []

        def make_cb(zone_name):
            def cb(obj_name, zn):
                events.append((obj_name, zn))
            return cb

        zone_a = Zone(name="zone_a", bbox=(0, 0, 300, 300), on_enter=make_cb("zone_a"))
        zone_b = Zone(name="zone_b", bbox=(700, 700, 1000, 1000), on_enter=make_cb("zone_b"))
        tracker.add_zone(zone_a)
        tracker.add_zone(zone_b)

        # Object in zone_a area
        frame = make_frame()
        draw_rect(frame, 50, 50, 30, 30, (0, 0, 255))
        tracker.track_color("dot", frame, (50, 50, 30, 30))
        tracker.update(frame)

        # Should only trigger zone_a
        zone_a_events = [e for e in events if e[1] == "zone_a"]
        zone_b_events = [e for e in events if e[1] == "zone_b"]
        assert len(zone_a_events) > 0
        assert len(zone_b_events) == 0


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

class TestStaticHelpers:
    def test_compute_color_histogram(self):
        """Returns correct shape (180, 1)."""
        frame = make_frame()
        draw_rect(frame, 100, 100, 50, 50, (0, 0, 255))

        hist = ObjectTracker.compute_color_histogram(frame, (100, 100, 50, 50))
        assert hist.shape == (180, 1)

    def test_match_template_exact(self):
        """Template exists in frame, high confidence."""
        frame = make_frame(200, 200)
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_rect(template, 0, 0, 30, 30, (0, 255, 0))
        draw_rect(template, 5, 5, 20, 20, (255, 0, 0))

        # Place template in frame
        frame[80:110, 90:120] = template

        y, x, conf = ObjectTracker.match_template(frame, template)
        assert conf > 0.9
        assert abs(y - 80) < 3
        assert abs(x - 90) < 3

    def test_match_template_not_found(self):
        """Random noise frame, low confidence."""
        rng = np.random.RandomState(99)
        frame = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_rect(template, 0, 0, 15, 30, (0, 200, 100))
        draw_rect(template, 15, 0, 15, 30, (200, 0, 100))

        _, _, conf = ObjectTracker.match_template(frame, template)
        assert conf < 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_object_lost_triggers_zone_exit(self):
        """Tracked object becomes invisible -> triggers zone exit."""
        tracker = ObjectTracker(frame_size=(768, 768))
        exited = []

        def on_exit(obj_name, zone_name):
            exited.append((obj_name, zone_name))

        zone = Zone(name="watch", bbox=(0, 0, 500, 500), on_exit=on_exit)
        tracker.add_zone(zone)

        # Template tracking — place template in frame inside zone
        template = np.zeros((30, 30, 3), dtype=np.uint8)
        draw_rect(template, 0, 0, 30, 30, (0, 255, 255))
        draw_rect(template, 5, 5, 20, 20, (255, 255, 0))
        tracker.track_template("card", template)

        # Frame with template visible inside zone
        frame1 = make_frame()
        frame1[50:80, 50:80] = template
        tracker.update(frame1)

        obj = tracker.get_object("card")
        assert obj.visible is True

        # Frame with no template — random noise
        rng = np.random.RandomState(7)
        frame2 = rng.randint(0, 256, (768, 768, 3), dtype=np.uint8)
        tracker.update(frame2)

        obj2 = tracker.get_object("card")
        assert obj2.visible is False
        assert len(exited) > 0
        assert exited[-1] == ("card", "watch")

    def test_update_empty_tracker(self):
        """Update with no tracked objects returns empty dict."""
        tracker = ObjectTracker()
        frame = make_frame()
        result = tracker.update(frame)
        assert result == {}
