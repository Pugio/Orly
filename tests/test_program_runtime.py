"""Tests for client.program_runtime — mini-program runtime with restricted namespace."""

import time

import numpy as np
import pytest
from unittest.mock import MagicMock

from client.program_runtime import validate_code, TableAPI, ProgramRuntime, ProgramStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_osm():
    """Mock OverlayStateManager."""
    osm = MagicMock()
    osm.to_json.return_value = {"overlays": [], "count": 0, "dimensions": [1000, 1000]}
    osm._om = MagicMock()
    osm._om.render_overlay.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    return osm


@pytest.fixture
def mock_tracker():
    """Mock ObjectTracker."""
    return MagicMock()


@pytest.fixture
def mock_session():
    """Mock SessionStore."""
    return MagicMock()


@pytest.fixture
def notifications():
    """Capture notifications."""
    msgs = []
    return msgs


@pytest.fixture
def table_api(mock_osm, mock_tracker, mock_session, notifications):
    frame = np.zeros((768, 768, 3), dtype=np.uint8)
    return TableAPI(
        mock_osm, mock_tracker, mock_session,
        notify_fn=lambda msg: notifications.append(msg),
        get_frame_fn=lambda: frame,
    )


@pytest.fixture
def runtime(mock_osm, mock_tracker, mock_session, notifications):
    def factory():
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        return TableAPI(
            mock_osm, mock_tracker, mock_session,
            notify_fn=lambda msg: notifications.append(msg),
            get_frame_fn=lambda: frame,
        )
    return ProgramRuntime(table_api_factory=factory)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidateCode:
    def test_validate_valid_code(self):
        valid, err = validate_code("x = 1 + 2")
        assert valid is True
        assert err == ""

    def test_validate_syntax_error(self):
        valid, err = validate_code("def (")
        assert valid is False
        assert "Syntax error" in err

    def test_validate_forbidden_import_os(self):
        valid, err = validate_code("import os")
        assert valid is False
        assert "Forbidden import: os" in err

    def test_validate_forbidden_import_subprocess(self):
        valid, err = validate_code("import subprocess")
        assert valid is False
        assert "Forbidden import: subprocess" in err

    def test_validate_forbidden_from_import(self):
        valid, err = validate_code("from os import path")
        assert valid is False
        assert "Forbidden import: os" in err

    def test_validate_forbidden_exec(self):
        valid, err = validate_code("exec('print(1)')")
        assert valid is False
        assert "Forbidden builtin: exec" in err

    def test_validate_forbidden_eval(self):
        valid, err = validate_code("eval('1+1')")
        assert valid is False
        assert "Forbidden builtin: eval" in err

    def test_validate_forbidden_open(self):
        valid, err = validate_code("open('file.txt')")
        assert valid is False
        assert "Forbidden builtin: open" in err

    def test_validate_allowed_imports(self):
        valid, err = validate_code("import math\nimport time")
        assert valid is True
        assert err == ""

    def test_validate_empty_code(self):
        valid, err = validate_code("")
        assert valid is True
        assert err == ""


# ---------------------------------------------------------------------------
# TableAPI tests
# ---------------------------------------------------------------------------

class TestTableAPI:
    def test_api_place_overlay(self, table_api, mock_osm):
        table_api.place_overlay("label1", "text", {"x": 100, "y": 200}, {"text": "hi"})
        mock_osm._om.render_overlay.assert_called_once()
        mock_osm.add.assert_called_once()

    def test_api_remove_overlay(self, table_api, mock_osm):
        table_api.remove_overlay("label1")
        mock_osm.remove.assert_called_once_with("label1")

    def test_api_get_overlay_state(self, table_api, mock_osm):
        result = table_api.get_overlay_state()
        assert result == {"overlays": [], "count": 0, "dimensions": [1000, 1000]}
        mock_osm.to_json.assert_called_once()

    def test_api_get_frame(self, table_api):
        frame = table_api.get_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (768, 768, 3)

    def test_api_notify(self, table_api, notifications):
        table_api.notify("hello agent")
        assert "hello agent" in notifications

    def test_api_log(self, table_api):
        table_api.log("debug msg")
        assert "debug msg" in table_api._log_messages

    def test_api_on_frame_registers(self, table_api):
        cb = lambda frame: None
        table_api.on_frame(cb)
        assert cb in table_api._frame_callbacks

    def test_api_stop(self, table_api):
        assert table_api.stopped is False
        table_api.stop()
        assert table_api.stopped is True

    def test_api_track_color(self, table_api, mock_tracker):
        table_api.track_color("pen", (100, 100, 50, 50))
        mock_tracker.track_color.assert_called_once()

    def test_api_add_zone(self, table_api, mock_tracker):
        table_api.add_zone("dropzone", (0, 0, 500, 500))
        mock_tracker.add_zone.assert_called_once()

    def test_api_save_image(self, table_api, mock_session):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        table_api.save_image("test_img", img)
        mock_session.save_image.assert_called_once_with("test_img", img)

    def test_api_load_image(self, table_api, mock_session):
        table_api.load_image("test_img")
        mock_session.load_image.assert_called_once_with("test_img")


# ---------------------------------------------------------------------------
# ProgramRuntime tests
# ---------------------------------------------------------------------------

class TestProgramRuntime:
    def test_run_simple_program(self, runtime):
        status = runtime.run("test", "x = 1")
        time.sleep(0.1)
        status = runtime.get_status("test")
        # Fast program finishes quickly — should be stopped (or running if we're very fast)
        assert status.state in ("running", "stopped")

    def test_run_invalid_code(self, runtime):
        status = runtime.run("bad", "def (")
        assert status.state == "error"
        assert "Syntax error" in status.error

    def test_run_forbidden_code(self, runtime):
        status = runtime.run("evil", "import os")
        assert status.state == "error"
        assert "Forbidden import: os" in status.error

    def test_run_program_with_notify(self, runtime, notifications):
        runtime.run("notifier", "table.notify('hello')")
        time.sleep(0.2)
        assert "hello" in notifications

    def test_run_program_with_on_frame(self, runtime):
        results = []
        code = (
            "def cb(frame):\n"
            "    table.notify('got_frame')\n"
            "table.on_frame(cb)\n"
            "# Keep alive briefly so callback can be invoked\n"
            "import time\n"
            "time.sleep(0.5)\n"
        )
        runtime.run("framer", code)
        time.sleep(0.1)
        # Process a frame — the callback should fire
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        runtime.process_frame(frame)

        # Check the program's status — frame_count should be incremented
        status = runtime.get_status("framer")
        assert status.frame_count >= 1
        runtime.stop("framer")

    def test_stop_program(self, runtime):
        code = "import time\nwhile not table.stopped:\n    time.sleep(0.05)\n"
        runtime.run("sleeper", code)
        time.sleep(0.1)
        assert runtime.get_status("sleeper").state == "running"
        result = runtime.stop("sleeper")
        assert result is True
        assert runtime.get_status("sleeper").state == "stopped"

    def test_stop_nonexistent(self, runtime):
        assert runtime.stop("nope") is False

    def test_stop_all(self, runtime):
        code = "import time\nwhile not table.stopped:\n    time.sleep(0.05)\n"
        runtime.run("a", code)
        runtime.run("b", code)
        time.sleep(0.1)
        runtime.stop_all()
        assert runtime.get_status("a").state == "stopped"
        assert runtime.get_status("b").state == "stopped"

    def test_list_programs(self, runtime):
        runtime.run("p1", "x = 1")
        runtime.run("p2", "y = 2")
        time.sleep(0.1)
        progs = runtime.list_programs()
        names = {p.name for p in progs}
        assert names == {"p1", "p2"}

    def test_get_status(self, runtime):
        runtime.run("prog", "x = 1", description="test program")
        time.sleep(0.1)
        status = runtime.get_status("prog")
        assert isinstance(status, ProgramStatus)
        assert status.name == "prog"
        assert status.description == "test program"

    def test_get_status_nonexistent(self, runtime):
        assert runtime.get_status("nope") is None

    def test_run_replaces_existing(self, runtime):
        code = "import time\nwhile not table.stopped:\n    time.sleep(0.05)\n"
        runtime.run("dup", code)
        time.sleep(0.1)
        assert runtime.get_status("dup").state == "running"
        # Run again with same name — should stop the first one
        runtime.run("dup", "x = 1")
        time.sleep(0.1)
        # There should only be one program named "dup"
        progs = runtime.list_programs()
        dup_progs = [p for p in progs if p.name == "dup"]
        assert len(dup_progs) == 1

    def test_runtime_error_in_program(self, runtime):
        runtime.run("divzero", "1/0")
        time.sleep(0.2)
        status = runtime.get_status("divzero")
        assert status.state == "error"
        assert "ZeroDivisionError" in status.error

    def test_frame_callback_error(self, runtime):
        code = (
            "def bad_cb(frame):\n"
            "    raise ValueError('boom')\n"
            "table.on_frame(bad_cb)\n"
            "import time\n"
            "time.sleep(0.5)\n"
        )
        runtime.run("badframe", code)
        time.sleep(0.1)
        frame = np.zeros((768, 768, 3), dtype=np.uint8)
        runtime.process_frame(frame)
        status = runtime.get_status("badframe")
        assert status.state == "error"
        assert "ValueError" in status.error
        runtime.stop("badframe")

    def test_program_namespace_has_numpy(self, runtime):
        runtime.run("nptest", "x = np.array([1, 2, 3])")
        time.sleep(0.2)
        status = runtime.get_status("nptest")
        assert status.state != "error", f"Unexpected error: {status.error}"

    def test_program_namespace_has_cv2(self, runtime):
        code = "gray = cv2.cvtColor(table.get_frame(), cv2.COLOR_BGR2GRAY)"
        runtime.run("cv2test", code)
        time.sleep(0.2)
        status = runtime.get_status("cv2test")
        assert status.state != "error", f"Unexpected error: {status.error}"

    def test_program_namespace_has_math(self, runtime):
        runtime.run("mathtest", "x = math.sqrt(2)")
        time.sleep(0.2)
        status = runtime.get_status("mathtest")
        assert status.state != "error", f"Unexpected error: {status.error}"

    def test_program_cannot_access_os(self, runtime):
        status = runtime.run("ostest", "import os")
        assert status.state == "error"
        assert "Forbidden import" in status.error

    def test_program_print_redirected(self, runtime):
        runtime.run("printer", "print('hello')")
        time.sleep(0.2)
        # Access the internal API to check log messages
        prog = runtime._programs.get("printer")
        assert prog is not None
        assert "hello" in prog.api._log_messages
