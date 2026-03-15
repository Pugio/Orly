"""Tests for client.hand_detector — pointing gesture detection (pure functions)."""

import pytest

from client.hand_detector import (
    PointingResult,
    PointingTracker,
    fingertip_to_table_coords,
    is_pointing_gesture,
)


# --- PointingResult ---


class TestPointingResult:
    def test_pointing_result_fields(self):
        r = PointingResult(fingertip=(300, 450), confidence=0.9, is_pointing=True)
        assert r.fingertip == (300, 450)
        assert r.confidence == 0.9
        assert r.is_pointing is True


# --- is_pointing_gesture ---


def _make_landmarks(
    *,
    index_tip_y: float = 0.0,
    index_mcp_y: float = 0.5,
    middle_tip_y: float = 0.8,
    middle_mcp_y: float = 0.5,
    ring_tip_y: float = 0.8,
    ring_mcp_y: float = 0.5,
    pinky_tip_y: float = 0.8,
    pinky_mcp_y: float = 0.5,
) -> list[tuple[float, float, float]]:
    """Build a minimal 21-landmark list with controlled finger positions.

    MediaPipe landmarks: index tip=8, index MCP=5, middle tip=12, middle MCP=9,
    ring tip=16, ring MCP=13, pinky tip=20, pinky MCP=17.
    We only care about those 8 landmarks; fill the rest with (0.5, 0.5, 0).
    """
    lm = [(0.5, 0.5, 0.0)] * 21
    lm[8] = (0.5, index_tip_y, 0.0)    # index tip
    lm[5] = (0.5, index_mcp_y, 0.0)    # index MCP
    lm[12] = (0.5, middle_tip_y, 0.0)  # middle tip
    lm[9] = (0.5, middle_mcp_y, 0.0)   # middle MCP
    lm[16] = (0.5, ring_tip_y, 0.0)    # ring tip
    lm[13] = (0.5, ring_mcp_y, 0.0)    # ring MCP
    lm[20] = (0.5, pinky_tip_y, 0.0)   # pinky tip
    lm[17] = (0.5, pinky_mcp_y, 0.0)   # pinky MCP
    return lm


class TestIsPointingGesture:
    def test_is_pointing_index_extended_others_curled(self):
        """Index extended (tip far from MCP), others curled (tips near MCPs)."""
        lm = _make_landmarks(
            index_tip_y=0.0, index_mcp_y=0.5,   # extended: dist=0.5
            middle_tip_y=0.52, middle_mcp_y=0.5, # curled: dist=0.02
            ring_tip_y=0.51, ring_mcp_y=0.5,     # curled: dist=0.01
            pinky_tip_y=0.49, pinky_mcp_y=0.5,   # curled: dist=0.01
        )
        assert is_pointing_gesture(lm) is True

    def test_is_not_pointing_all_extended(self):
        """All fingers extended = open hand, not pointing."""
        lm = _make_landmarks(
            index_tip_y=0.0, index_mcp_y=0.5,
            middle_tip_y=0.0, middle_mcp_y=0.5,
            ring_tip_y=0.0, ring_mcp_y=0.5,
            pinky_tip_y=0.0, pinky_mcp_y=0.5,
        )
        assert is_pointing_gesture(lm) is False

    def test_is_not_pointing_fist(self):
        """All fingers curled = fist, not pointing."""
        lm = _make_landmarks(
            index_tip_y=0.52, index_mcp_y=0.5,
            middle_tip_y=0.52, middle_mcp_y=0.5,
            ring_tip_y=0.51, ring_mcp_y=0.5,
            pinky_tip_y=0.49, pinky_mcp_y=0.5,
        )
        assert is_pointing_gesture(lm) is False

    def test_is_pointing_threshold_boundary(self):
        """Index barely at threshold = not pointing (need strictly above)."""
        lm = _make_landmarks(
            index_tip_y=0.4, index_mcp_y=0.5,   # dist=0.1 = exactly threshold
            middle_tip_y=0.52, middle_mcp_y=0.5,
            ring_tip_y=0.51, ring_mcp_y=0.5,
            pinky_tip_y=0.49, pinky_mcp_y=0.5,
        )
        assert is_pointing_gesture(lm, extension_threshold=0.1) is False


# --- fingertip_to_table_coords ---


class TestFingertipToTableCoords:
    def test_center(self):
        y, x = fingertip_to_table_coords(384, 384, 768, 768)
        assert (y, x) == pytest.approx((500, 500), abs=1)

    def test_top_left(self):
        y, x = fingertip_to_table_coords(0, 0, 768, 768)
        assert (y, x) == (0, 0)

    def test_bottom_right(self):
        y, x = fingertip_to_table_coords(768, 768, 768, 768)
        assert (y, x) == pytest.approx((1000, 1000), abs=1)

    def test_arbitrary(self):
        y, x = fingertip_to_table_coords(576, 192, 768, 768)
        assert (y, x) == pytest.approx((750, 250), abs=1)


# --- PointingTracker ---


class TestPointingTracker:
    def _result(self, y=500, x=500, conf=0.9):
        return PointingResult(fingertip=(y, x), confidence=conf, is_pointing=True)

    def test_no_detection_returns_none(self):
        tracker = PointingTracker()
        tracker.update(None, now=0.0)
        assert tracker.current_point() is None
        assert tracker.is_stable() is False

    def test_single_detection_not_yet_stable(self):
        tracker = PointingTracker(stability_time=1.0)
        tracker.update(self._result(), now=0.0)
        assert tracker.current_point() == (500, 500)
        assert tracker.is_stable() is False

    def test_stable_after_duration(self):
        tracker = PointingTracker(stability_time=1.0)
        tracker.update(self._result(), now=0.0)
        tracker.update(self._result(), now=0.5)
        assert tracker.is_stable() is False
        tracker.update(self._result(), now=1.1)
        assert tracker.is_stable() is True

    def test_position_jump_resets_stability(self):
        tracker = PointingTracker(stability_time=1.0, jitter_threshold=50.0)
        tracker.update(self._result(500, 500), now=0.0)
        tracker.update(self._result(500, 500), now=0.8)
        # Big jump — resets stability timer
        tracker.update(self._result(800, 800), now=0.9)
        assert tracker.is_stable() is False
        # Even after enough time from original start, still not stable (reset)
        tracker.update(self._result(800, 800), now=1.5)
        assert tracker.is_stable() is False
        # Stable after full duration from reset
        tracker.update(self._result(800, 800), now=2.0)
        assert tracker.is_stable() is True

    def test_debounce_prevents_rapid_notifications(self):
        tracker = PointingTracker(stability_time=0.5, debounce=2.0)
        tracker.update(self._result(), now=0.0)
        tracker.update(self._result(), now=0.6)
        assert tracker.is_stable() is True
        assert tracker.should_notify(now=0.6) is True
        # Calling should_notify consumes it
        assert tracker.should_notify(now=0.7) is False

    def test_debounce_allows_after_window(self):
        tracker = PointingTracker(stability_time=0.5, debounce=2.0)
        tracker.update(self._result(), now=0.0)
        tracker.update(self._result(), now=0.6)
        assert tracker.should_notify(now=0.6) is True
        # After debounce window, should allow again
        tracker.update(self._result(), now=2.7)
        assert tracker.should_notify(now=2.7) is True

    def test_loss_of_pointing_clears_state(self):
        tracker = PointingTracker(stability_time=0.5)
        tracker.update(self._result(), now=0.0)
        tracker.update(self._result(), now=0.6)
        assert tracker.is_stable() is True
        # Hand removed
        tracker.update(None, now=1.0)
        assert tracker.current_point() is None
        assert tracker.is_stable() is False


# --- System prompt ---


class TestSystemPrompt:
    def test_system_prompt_contains_pointing_section(self):
        from backend.agent import SYSTEM_PROMPT

        assert "POINTING" in SYSTEM_PROMPT
        assert "pointing notification" in SYSTEM_PROMPT.lower() or "pointing" in SYSTEM_PROMPT.lower()
