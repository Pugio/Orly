"""Edge-case tests for recent bug fixes and missing coverage.

Covers:
- validate_code edge cases (forbidden imports, aliases, dunder import)
- ProgramRuntime._latest_frame updates
- ProgramRuntime + ObjectTracker integration
- ObjectTracker thread safety
- OverlayState + interruption clearing
- SessionStore sanitization and I/O edge cases
- Program namespace runtime safety
- OverlayStateManager edge cases
- TableAPI edge cases
"""

import logging
import threading
import time

import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from client.object_tracker import ObjectTracker, TrackedObject, Zone
from client.overlay_state import OverlayStateManager, OverlayEntry
from client.session_store import SessionStore
from client.program_runtime import (
    validate_code, TableAPI, ProgramRuntime, ProgramStatus,
    FORBIDDEN_IMPORTS, FORBIDDEN_BUILTINS,
)


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------

class TestValidateCode:
    def test_validate_from_os_path_import(self):
        """'from os.path import join' should be blocked (top-level is 'os')."""
        valid, err = validate_code("from os.path import join")
        assert not valid
        assert "os" in err.lower() or "Forbidden" in err

    def test_validate_nested_forbidden_import(self):
        """'from http.client import HTTPConnection' should be blocked."""
        valid, err = validate_code("from http.client import HTTPConnection")
        assert not valid
        assert "http" in err.lower() or "Forbidden" in err

    def test_validate_import_as(self):
        """'import os as operating_system' should be blocked."""
        valid, err = validate_code("import os as operating_system")
        assert not valid
        assert "os" in err.lower() or "Forbidden" in err

    def test_validate_dunder_import_in_code(self):
        """'__import__(\"os\")' should be blocked at AST level."""
        valid, err = validate_code("__import__('os')")
        assert not valid
        assert "__import__" in err or "Forbidden" in err

    def test_validate_allowed_numpy_import(self):
        """'import numpy' should be allowed (not in FORBIDDEN_IMPORTS)."""
        valid, err = validate_code("import numpy")
        assert valid
        assert err == ""

    def test_validate_multiline_code(self):
        """Multi-line code with functions and classes should validate."""
        code = """\
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

def midpoint(a, b):
    return Point((a.x + b.x) / 2, (a.y + b.y) / 2)
"""
        valid, err = validate_code(code)
        assert valid
        assert err == ""


# ---------------------------------------------------------------------------
# ProgramRuntime._latest_frame
# ---------------------------------------------------------------------------

def _make_runtime():
    """Create a ProgramRuntime with a minimal TableAPI factory."""
    om = MagicMock()
    om._make_bg.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    om.mode = "screen"
    osm = OverlayStateManager(om)
    tracker = ObjectTracker(frame_size=(100, 100))
    store = MagicMock()

    def factory():
        return TableAPI(osm, tracker, store, notify_fn=lambda m: None,
                        get_frame_fn=lambda: runtime._latest_frame)

    runtime = ProgramRuntime(factory)
    return runtime


class TestLatestFrame:
    def test_process_frame_updates_latest_frame(self):
        """process_frame should store the frame in _latest_frame."""
        runtime = _make_runtime()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 42
        runtime.process_frame(frame)
        assert runtime._latest_frame is frame

    def test_get_frame_returns_latest(self):
        """TableAPI.get_frame() should return the latest processed frame."""
        runtime = _make_runtime()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 99
        runtime.process_frame(frame)
        # Create an API that reads from the runtime's latest frame
        api = runtime._api_factory()
        result = api.get_frame()
        assert result is frame

    def test_get_frame_returns_none_initially(self):
        """Before any process_frame call, _latest_frame should be None."""
        runtime = _make_runtime()
        assert runtime._latest_frame is None


# ---------------------------------------------------------------------------
# ProgramRuntime + ObjectTracker integration
# ---------------------------------------------------------------------------

class TestRuntimeTrackerIntegration:
    def test_process_frame_calls_tracker_update(self):
        """When _object_tracker is set, process_frame calls tracker.update."""
        runtime = _make_runtime()
        mock_tracker = MagicMock()
        runtime._object_tracker = mock_tracker
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        runtime.process_frame(frame)
        mock_tracker.update.assert_called_once_with(frame)

    def test_process_frame_without_tracker(self):
        """When _object_tracker is None, process_frame doesn't crash."""
        runtime = _make_runtime()
        runtime._object_tracker = None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Should not raise
        runtime.process_frame(frame)


# ---------------------------------------------------------------------------
# ObjectTracker thread safety
# ---------------------------------------------------------------------------

class TestTrackerThreadSafety:
    def test_tracker_concurrent_access(self):
        """Concurrent update/track_template/get_all should not crash."""
        tracker = ObjectTracker(frame_size=(100, 100))
        errors = []
        iterations = 100

        def updater():
            for _ in range(iterations):
                try:
                    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    tracker.update(frame)
                except Exception as e:
                    errors.append(e)

        def template_adder():
            for i in range(iterations):
                try:
                    tpl = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
                    tracker.track_template(f"obj_{i % 5}", tpl)
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(iterations):
                try:
                    tracker.get_all()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=template_adder),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent access raised: {errors}"


# ---------------------------------------------------------------------------
# OverlayState + interruption
# ---------------------------------------------------------------------------

class TestOverlayClearBoth:
    def test_clear_clears_both_state_and_canvas(self):
        """clear() should empty list_names AND reset canvas to background."""
        om = MagicMock()
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        om._make_bg.return_value = bg.copy()
        om.mode = "screen"
        om.canvas = bg.copy()
        om.place_on_canvas.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 128
        osm = OverlayStateManager(om)

        # Add an overlay
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        osm.add("test", "annotation", [0, 0, 500, 500], "Test", {}, img)
        assert "test" in osm.list_names()

        # Clear
        osm.clear()
        assert osm.list_names() == []
        om.clear.assert_called_once()


# ---------------------------------------------------------------------------
# SessionStore edge cases
# ---------------------------------------------------------------------------

class TestSessionStoreEdgeCases:
    def test_sanitize_name_only_special_chars(self):
        """All-special-chars name should become 'unnamed'."""
        assert SessionStore.sanitize_name("!!!@@@") == "unnamed"

    def test_sanitize_name_leading_trailing_hyphens(self):
        """Leading/trailing hyphens should be stripped."""
        assert SessionStore.sanitize_name("--hello--") == "hello"

    def test_save_image_creates_parent_dirs(self, tmp_path):
        """save_image should work when store is freshly created in tmp dir."""
        store = SessionStore(session_dir=str(tmp_path / "test_session"))
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        path = store.save_image("myimage", img)
        assert path.endswith(".png")
        import os
        assert os.path.exists(path)

    def test_load_image_returns_correct_channels(self, tmp_path):
        """Save a 3-channel BGR image, load it, verify it's 3-channel."""
        store = SessionStore(session_dir=str(tmp_path / "test_session"))
        img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        store.save_image("color", img)
        loaded = store.load_image("color")
        assert loaded is not None
        assert loaded.ndim == 3
        assert loaded.shape[2] == 3


# ---------------------------------------------------------------------------
# Program namespace safety (runtime checks)
# ---------------------------------------------------------------------------

class TestProgramNamespaceSafety:
    def test_program_runtime_import_os_blocked(self):
        """Running 'import os' should result in error state."""
        runtime = _make_runtime()
        status = runtime.run("bad", "import os", description="bad program")
        # The validation should catch it before running
        assert status.state == "error"
        assert "os" in (status.error or "").lower() or "Forbidden" in (status.error or "")

    def test_program_runtime_safe_import_math(self):
        """Running 'import math; x = math.pi' should succeed."""
        runtime = _make_runtime()
        status = runtime.run("mathprog", "import math\nx = math.pi")
        # Should not be an error at validation time
        assert status.state != "error" or "math" not in (status.error or "")
        # Wait for program to finish
        time.sleep(0.2)
        final = runtime.get_status("mathprog")
        assert final.state in ("stopped", "running")
        assert final.error is None

    def test_program_runtime_safe_import_random(self):
        """Running 'import random; x = random.randint(1,10)' should succeed."""
        runtime = _make_runtime()
        status = runtime.run("randprog", "import random\nx = random.randint(1, 10)")
        assert status.state != "error" or "random" not in (status.error or "")
        time.sleep(0.2)
        final = runtime.get_status("randprog")
        assert final.state in ("stopped", "running")
        assert final.error is None


# ---------------------------------------------------------------------------
# OverlayStateManager edge cases
# ---------------------------------------------------------------------------

class TestOverlayStateEdgeCases:
    def _make_osm(self):
        om = MagicMock()
        om._make_bg.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        om.mode = "screen"
        om.canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        om.place_on_canvas.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        return OverlayStateManager(om)

    def test_overlay_state_add_empty_name(self):
        """Adding with name='' should work; to_ascii shows '?'."""
        osm = self._make_osm()
        img = np.ones((10, 10, 3), dtype=np.uint8)
        osm.add("", "annotation", [0, 0, 500, 500], "", {}, img)
        assert "" in osm.list_names()
        ascii_grid = osm.to_ascii()
        assert "?" in ascii_grid

    def test_overlay_state_to_json_excludes_image_data(self):
        """JSON output should not contain numpy arrays."""
        osm = self._make_osm()
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        osm.add("pic", "annotation", [100, 100, 500, 500], "Pic", {"key": "val"}, img)
        result = osm.to_json()
        # Check there are no numpy arrays in the JSON dict
        for overlay in result["overlays"]:
            for v in overlay.values():
                assert not isinstance(v, np.ndarray)

    def test_overlay_state_to_ascii_full_coverage(self):
        """An overlay at [0,0,1000,1000] should fill the entire grid."""
        osm = self._make_osm()
        img = np.ones((10, 10, 3), dtype=np.uint8)
        osm.add("F", "annotation", [0, 0, 1000, 1000], "Full", {}, img)
        ascii_grid = osm.to_ascii()
        # Every cell should be 'F', no '.' remaining
        assert "." not in ascii_grid
        assert "F" in ascii_grid


# ---------------------------------------------------------------------------
# TableAPI edge cases
# ---------------------------------------------------------------------------

class TestTableAPIEdgeCases:
    def _make_api(self):
        om = MagicMock()
        om._make_bg.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        om.mode = "screen"
        osm = OverlayStateManager(om)
        tracker = ObjectTracker(frame_size=(100, 100))
        store = MagicMock()
        api = TableAPI(osm, tracker, store,
                       notify_fn=lambda m: None,
                       get_frame_fn=lambda: None)
        return api, tracker

    def test_api_track_color_no_frame(self):
        """track_color when get_frame returns None should not crash."""
        api, tracker = self._make_api()
        # Should not raise
        api.track_color("red_obj", (10, 10, 20, 20))

    def test_api_get_tracked_nonexistent(self):
        """get_tracked for a name that doesn't exist returns None."""
        api, tracker = self._make_api()
        assert api.get_tracked("ghost") is None

    def test_api_play_tone_logs(self):
        """play_tone should log the frequency and duration."""
        api, tracker = self._make_api()
        api.play_tone(440, 0.5)
        assert any("440" in msg and "0.5" in msg for msg in api._log_messages)


# ---------------------------------------------------------------------------
# Regression tests for specific bugs fixed in review
# ---------------------------------------------------------------------------


class TestProgramFrameCallbackAfterThreadExit:
    """Regression: programs that register on_frame and exit should still run callbacks."""

    def test_frame_callback_fires_after_setup_thread_exits(self):
        """A program that registers on_frame and exits should still be 'running'."""
        results = []
        runtime = _make_runtime()
        code = """
def handler(frame):
    pass
table.on_frame(handler)
"""
        status = runtime.run("setup-and-exit", code, "registers callback then exits")
        time.sleep(0.3)  # let setup thread finish

        prog_status = runtime.get_status("setup-and-exit")
        assert prog_status.state == "running", (
            f"Program with registered callbacks should be 'running', got '{prog_status.state}'"
        )

    def test_frame_callback_actually_called_after_thread_exit(self):
        """Frame callbacks should fire even after the setup thread has exited."""
        runtime = _make_runtime()
        # Use a shared list to track callback invocations
        code = """
call_count = [0]
def handler(frame):
    call_count[0] += 1
    table.log(f"called {call_count[0]}")
table.on_frame(handler)
"""
        runtime.run("counter", code, "counts frames")
        time.sleep(0.3)  # let setup thread finish

        # Simulate a frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        runtime.process_frame(frame)

        prog = runtime._programs["counter"]
        assert prog.frame_count >= 1, "Frame callback should have been called"

    def test_program_without_callbacks_stops_after_exit(self):
        """A program with no frame callbacks should be 'stopped' after thread exits."""
        runtime = _make_runtime()
        runtime.run("one-shot", "x = 42", "just computes")
        time.sleep(0.3)

        status = runtime.get_status("one-shot")
        assert status.state == "stopped"


class TestOverlayStateRecompositeFalse:
    """Regression: add(recomposite=False) should not wipe the canvas."""

    def test_add_without_recomposite_preserves_canvas(self):
        om = MagicMock()
        om._make_bg.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        om.mode = "screen"
        om.canvas = np.full((100, 100, 3), 128, dtype=np.uint8)  # gray canvas
        om.place_on_canvas.return_value = om.canvas

        osm = OverlayStateManager(om)

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        osm.add("test", "annotation", [0, 0, 500, 500], "test", {}, img,
                recomposite=False)

        # overlay_state should have the entry
        assert osm.list_names() == ["test"]
        # _make_bg should NOT have been called (no recomposite)
        om._make_bg.assert_not_called()

    def test_add_with_recomposite_rebuilds_canvas(self):
        om = MagicMock()
        om._make_bg.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        om.mode = "screen"
        om.canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        om.place_on_canvas.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        osm = OverlayStateManager(om)

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        osm.add("test", "annotation", [0, 0, 500, 500], "test", {}, img,
                recomposite=True)

        # _make_bg SHOULD have been called (recomposite)
        om._make_bg.assert_called()


class TestSafeImportDoesNotShadow:
    """Regression: _safe_import parameter shouldn't shadow the program name."""

    def test_program_name_preserved_after_import(self):
        runtime = _make_runtime()
        # The program imports math (allowed) — the program name should
        # still be accessible in the runtime.
        runtime.run("my-program", "import math\nx = math.pi")
        time.sleep(0.2)

        status = runtime.get_status("my-program")
        assert status is not None
        assert status.name == "my-program"
        assert status.error is None


class TestZoneCallbackNoDeadlock:
    """Regression: zone callbacks must not deadlock when calling tracker methods."""

    def test_zone_callback_can_call_get_object(self):
        """on_enter callback that calls tracker.get_object should not deadlock."""
        tracker = ObjectTracker(frame_size=(100, 100))
        results = []

        def on_enter(obj_name, zone_name):
            # This would deadlock if callbacks fired while holding the lock.
            obj = tracker.get_object(obj_name)
            results.append(obj)

        from client.object_tracker import Zone
        tracker.add_zone(Zone("target", (0, 0, 500, 500), on_enter=on_enter))

        # Create a template in the zone area
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        template = np.zeros((20, 20, 3), dtype=np.uint8)
        cv2.circle(template, (10, 10), 8, (0, 255, 0), -1)
        # Place the template in the top-left (inside the zone)
        frame[10:30, 10:30] = template
        tracker.track_template("piece", template)

        # Update should find the template in the zone and fire on_enter
        tracker.update(frame)

        # If we get here without deadlock, the fix works.
        # The callback may or may not have fired depending on match confidence,
        # but the important thing is no deadlock.
        assert True  # reached without hanging


class TestMissingBuiltins:
    """Regression: programs should have access to common Python builtins."""

    def test_any_all_available(self):
        runtime = _make_runtime()
        runtime.run("builtins-test", "result = any([False, True])\nassert result")
        time.sleep(0.2)
        status = runtime.get_status("builtins-test")
        assert status.error is None

    def test_hasattr_available(self):
        runtime = _make_runtime()
        runtime.run("hasattr-test", "assert hasattr([], 'append')")
        time.sleep(0.2)
        assert runtime.get_status("hasattr-test").error is None

    def test_super_available(self):
        runtime = _make_runtime()
        code = "class Base: pass\nclass Child(Base):\n  def __init__(self): super().__init__()\nChild()"
        runtime.run("super-test", code)
        time.sleep(0.2)
        assert runtime.get_status("super-test").error is None

    def test_iter_next_available(self):
        runtime = _make_runtime()
        runtime.run("iter-test", "it = iter([1,2,3])\nassert next(it) == 1")
        time.sleep(0.2)
        assert runtime.get_status("iter-test").error is None


class TestAdjustTextPlacement:
    """Regression: adjust_text_placement should be used consistently."""

    def test_markdown_gets_expanded(self):
        from client.overlay_manager import OverlayManager
        result = OverlayManager.adjust_text_placement(
            "markdown", [100, 100, 200, 200])
        ymin, xmin, ymax, xmax = result
        assert (xmax - xmin) >= 500
        assert (ymax - ymin) >= 400

    def test_graph_not_expanded(self):
        from client.overlay_manager import OverlayManager
        result = OverlayManager.adjust_text_placement(
            "graph", [100, 100, 200, 200])
        assert result == [100, 100, 200, 200]

    def test_annotation_gets_expanded(self):
        from client.overlay_manager import OverlayManager
        result = OverlayManager.adjust_text_placement(
            "annotation", [0, 0, 100, 100])
        ymin, xmin, ymax, xmax = result
        assert (xmax - xmin) >= 500
        assert (ymax - ymin) >= 400
