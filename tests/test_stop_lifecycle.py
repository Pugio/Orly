"""Tests for Bug 4: stop_program reports stopped while thread is alive."""

import threading
import time

import numpy as np
import pytest

from client.program_runtime import ProgramRuntime, TableAPI, _RunningProgram
from client.overlay_state import OverlayStateManager
from client.overlay_manager import OverlayManager


def _make_runtime():
    """Create a ProgramRuntime with a minimal TableAPI factory."""
    om = OverlayManager(H_proj=None, proj_width=100, proj_height=100)
    osm = OverlayStateManager(om)
    notifications = []

    def factory():
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        return TableAPI(
            osm, None, None,
            notify_fn=lambda msg: notifications.append(msg),
            get_frame_fn=lambda: frame,
        )

    return ProgramRuntime(table_api_factory=factory)


class TestStopLifecycle:
    def test_state_reports_stopping_while_thread_alive(self):
        """state property should return 'stopping' when stop_event is set but thread alive."""
        om = OverlayManager(H_proj=None, proj_width=100, proj_height=100)
        osm = OverlayStateManager(om)
        api = TableAPI(osm, None, None,
                       notify_fn=lambda msg: None,
                       get_frame_fn=lambda: np.zeros((10, 10, 3), dtype=np.uint8))

        # Create a thread that blocks until we release it
        release = threading.Event()

        def slow_work():
            release.wait(timeout=5)

        thread = threading.Thread(target=slow_work, daemon=True)
        thread.start()

        prog = _RunningProgram("test", "test program", api, thread)

        # Thread is alive, stop not requested → running
        assert prog.state == "running"

        # Set stop event but don't release thread → should be "stopping"
        api.stop()
        assert thread.is_alive()
        assert prog.state == "stopping"

        # Release thread → should be "stopped"
        release.set()
        thread.join(timeout=2)
        assert prog.state == "stopped"

    def test_state_after_thread_exits(self):
        """After thread exits naturally, state should be 'stopped'."""
        runtime = _make_runtime()
        code = "pass  # exits immediately"
        runtime.run("quick", code, "quick program")
        time.sleep(0.2)
        status = runtime.get_status("quick")
        assert status is not None
        assert status.state == "stopped"
