"""Tests for interruption handling robustness.

Covers the full interruption chain: Gemini → backend → client → overlay clear,
plus edge cases like async image completion after interrupt, rapid interrupts,
and interactions with programs/video/music.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.main import format_interrupted
from client.overlay_state import OverlayStateManager


# ---------------------------------------------------------------------------
# Helper: create a real OverlayManager with in-memory canvas (no projector)
# ---------------------------------------------------------------------------

def make_overlay_manager(**kwargs):
    """Create a real OverlayManager for testing (no projector, no Gemini)."""
    from client.overlay_manager import OverlayManager
    defaults = dict(
        H_proj=None, proj_width=200, proj_height=200, mode="screen",
    )
    defaults.update(kwargs)
    return OverlayManager(**defaults)


def make_test_overlay(w=50, h=50):
    """Create a bright test overlay image."""
    return np.ones((h, w, 3), dtype=np.uint8) * 200


# ===========================================================================
# 1. Interruption clears overlays and resets canvas
# ===========================================================================


class TestInterruptionClearsOverlays:
    def test_clear_resets_canvas_to_black(self):
        """After interruption, canvas should be all zeros (black)."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)
        om.overlay_state = osm

        # Add an overlay so canvas has content
        img = make_test_overlay()
        osm.add("math_graph", "annotation", [0, 0, 500, 500], "Graph", {}, img)
        assert om._has_content is True
        assert om.canvas.sum() > 0

        # Simulate interruption
        osm.clear()
        assert om._has_content is False
        assert om.canvas.sum() == 0
        assert osm.list_names() == []

    def test_clear_with_multiple_overlays(self):
        """All overlays should be removed, not just the latest."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)
        om.overlay_state = osm

        for name in ["overlay_a", "overlay_b", "overlay_c"]:
            img = make_test_overlay()
            osm.add(name, "annotation", [0, 0, 250, 250], name, {}, img)

        assert len(osm.list_names()) == 3
        osm.clear()
        assert osm.list_names() == []
        assert om.canvas.sum() == 0


# ===========================================================================
# 2. Interruption with no overlays active (no-op)
# ===========================================================================


class TestInterruptionNoOp:
    def test_clear_empty_state_is_safe(self):
        """Clearing when nothing is active should not crash."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)

        osm.clear()  # should not raise
        assert osm.list_names() == []
        assert om._has_content is False

    def test_double_clear_is_safe(self):
        """Calling clear() twice should not crash."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)

        img = make_test_overlay()
        osm.add("test", "annotation", [0, 0, 500, 500], "Test", {}, img)
        osm.clear()
        osm.clear()  # second clear
        assert osm.list_names() == []


# ===========================================================================
# 3. Multiple rapid interruptions
# ===========================================================================


class TestRapidInterruptions:
    def test_rapid_clear_no_corruption(self):
        """Rapid add-clear cycles should not corrupt state."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)
        om.overlay_state = osm

        for i in range(20):
            img = make_test_overlay()
            osm.add(f"overlay_{i}", "annotation", [0, 0, 500, 500], f"O{i}", {}, img)
            osm.clear()

        assert osm.list_names() == []
        assert om._has_content is False
        assert om.canvas.sum() == 0

    def test_concurrent_add_and_clear(self):
        """Concurrent adds and clears from different threads should not crash."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)
        om.overlay_state = osm

        errors = []

        def adder():
            try:
                for i in range(50):
                    img = make_test_overlay()
                    osm.add(f"t_{i}", "annotation", [0, 0, 500, 500], f"T{i}", {}, img)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                for _ in range(50):
                    osm.clear()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=clearer)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert errors == [], f"Thread errors: {errors}"
        # State should be consistent (either empty or has some overlays)
        names = osm.list_names()
        assert isinstance(names, list)


# ===========================================================================
# 4. Async image generation completing after interruption
# ===========================================================================


class TestAsyncImageAfterInterrupt:
    def test_generation_id_blocks_stale_image(self):
        """clear() increments _generation_id so _generate_image_async
        skips showing the image when it completes after an interruption."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)
        om.overlay_state = osm

        gen_before = om._generation_id

        # Add a loading placeholder
        img = make_test_overlay()
        osm.add("slow_image", "annotation", [0, 0, 500, 500], "Loading", {}, img)

        # Interruption clears everything and bumps generation
        osm.clear()
        assert om._generation_id == gen_before + 1

    def test_show_overlay_still_works_without_clear(self):
        """Direct _show_overlay without interruption still works normally."""
        om = make_overlay_manager()
        late_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        om._show_overlay(late_image, [0, 0, 500, 500], "image")
        assert om._has_content is True
        assert om.canvas.sum() > 0


# ===========================================================================
# 5. Interruption during refresh cycle
# ===========================================================================


class TestInterruptionDuringRefresh:
    def test_clear_during_refresh_cancels_refresh(self):
        """If interrupted while a refresh is in progress, the refresh cycle
        should be cancelled so complete_refresh() doesn't restore stale canvas."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)
        om.overlay_state = osm

        # Add overlay then start a refresh
        img = make_test_overlay()
        osm.add("test", "annotation", [0, 0, 500, 500], "Test", {}, img)
        om.request_refresh()
        assert om._refresh_requested is True
        assert om._saved_canvas is not None

        # Interruption during refresh
        osm.clear()

        # Refresh should be cancelled
        assert om._refresh_requested is False
        assert om._saved_canvas is None
        assert osm.list_names() == []
        assert om._has_content is False

    def test_complete_refresh_after_clear_is_noop(self):
        """complete_refresh() after clear() should not restore stale canvas."""
        om = make_overlay_manager()
        osm = OverlayStateManager(om)
        om.overlay_state = osm

        img = make_test_overlay()
        osm.add("test", "annotation", [0, 0, 500, 500], "Test", {}, img)
        om.request_refresh()
        osm.clear()

        # This would have restored stale canvas before the fix
        om.complete_refresh()
        assert om.canvas.sum() == 0
        assert om._has_content is False


# ===========================================================================
# 6. video_loop sends live frame after interruption
# ===========================================================================


class TestVideoLoopAfterInterrupt:
    @pytest.mark.asyncio
    async def test_sends_live_frame_after_clear(self):
        """After clear(), video_loop should send the live camera frame,
        not the cached last_clean_frame."""
        from client.main import video_loop

        mock_camera = MagicMock()
        live_jpeg = b"live_frame_data"
        test_frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mock_camera.get_rectified_frame.return_value = (live_jpeg, test_frame, np.eye(3))

        mock_client = AsyncMock()
        sent_frames = []

        async def capture_send(data):
            sent_frames.append(data)

        mock_client.send_video = capture_send

        # Create overlay manager with _has_content = False (post-interrupt state)
        om = make_overlay_manager()
        om._has_content = False

        iteration = 0
        original_sleep = asyncio.sleep

        async def limited_sleep(duration):
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                raise asyncio.CancelledError
            await original_sleep(0)

        with patch("client.main.asyncio.sleep", limited_sleep):
            with pytest.raises(asyncio.CancelledError):
                await video_loop(mock_camera, mock_client, fps=30.0,
                                 overlay_manager=om)

        # Should have sent the live frame (not a cached stale one)
        assert live_jpeg in sent_frames


# ===========================================================================
# 7. Programs survive interruption
# ===========================================================================


class TestProgramsSurviveInterrupt:
    def test_programs_not_stopped_by_clear(self):
        """overlay_state.clear() should not stop running programs."""
        from client.program_runtime import ProgramRuntime, TableAPI
        from client.object_tracker import ObjectTracker

        om = make_overlay_manager()
        osm = OverlayStateManager(om)
        om.overlay_state = osm

        def make_api():
            return TableAPI(
                overlay_state_manager=osm,
                object_tracker=ObjectTracker(),
                session_store=MagicMock(),
                notify_fn=lambda msg: None,
                get_frame_fn=lambda: None,
            )

        runtime = ProgramRuntime(table_api_factory=make_api)
        code = "import time\nwhile not table.stopped: time.sleep(0.01)"
        runtime.run("test_prog", code, "test")

        # Give the thread a moment to start
        time.sleep(0.05)
        status = runtime.get_status("test_prog")
        assert status.state == "running"

        # Simulate interruption (clear overlays, but don't touch runtime)
        osm.clear()

        # Program should still be running
        status_after = runtime.get_status("test_prog")
        assert status_after.state == "running"

        # Cleanup
        runtime.stop_all()


# ===========================================================================
# 8. Full WS client interrupt dispatch chain
# ===========================================================================


class TestWSClientInterruptChain:
    @pytest.mark.asyncio
    async def test_interrupted_message_triggers_callback(self):
        """The full chain: JSON {"type": "interrupted"} → callback fires."""
        from client.ws_client import TableLightClient

        client = TableLightClient("ws://fake")
        interrupted_count = 0

        async def on_interrupted():
            nonlocal interrupted_count
            interrupted_count += 1

        client.on_interrupted(on_interrupted)

        # Mock WS to yield 3 interrupted messages then stop
        messages = [
            json.dumps({"type": "interrupted"}),
            json.dumps({"type": "interrupted"}),
            json.dumps({"type": "interrupted"}),
        ]

        mock_ws = AsyncMock()
        idx = 0

        async def anext_impl(self):
            nonlocal idx
            if idx < len(messages):
                msg = messages[idx]
                idx += 1
                return msg
            raise StopAsyncIteration

        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = anext_impl
        client.ws = mock_ws

        await client.receive_loop()
        assert interrupted_count == 3

    @pytest.mark.asyncio
    async def test_interrupted_without_callback_is_safe(self):
        """Receiving interrupted with no registered callback should not crash."""
        from client.ws_client import TableLightClient

        client = TableLightClient("ws://fake")
        # Don't register on_interrupted

        messages = [json.dumps({"type": "interrupted"})]
        mock_ws = AsyncMock()
        idx = 0

        async def anext_impl(self):
            nonlocal idx
            if idx < len(messages):
                msg = messages[idx]
                idx += 1
                return msg
            raise StopAsyncIteration

        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = anext_impl
        client.ws = mock_ws

        await client.receive_loop()  # should not raise


# ===========================================================================
# 9. Backend format_interrupted sanity
# ===========================================================================


class TestBackendInterruptFormat:
    def test_format_is_minimal(self):
        """Interrupt message should be exactly {"type": "interrupted"}."""
        msg = format_interrupted()
        assert msg == {"type": "interrupted"}
        assert len(msg) == 1  # no extra fields
