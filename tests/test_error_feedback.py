"""Tests for Feature 9: Error recovery UX (client/error_feedback.py)."""

import numpy as np
import pytest

from client.error_feedback import ErrorFeedback


class TestMarkerStatus:
    def test_marker_within_grace_no_message(self):
        fb = ErrorFeedback()
        fb.update_marker_status(False, now=0.0)
        assert fb.get_user_message(now=2.0) is None  # within 3s grace

    def test_marker_past_grace_shows_message(self):
        fb = ErrorFeedback()
        fb.update_marker_status(False, now=0.0)
        msg = fb.get_user_message(now=4.0)
        assert msg is not None
        assert "marker" in msg.lower()

    def test_marker_detected_clears_error(self):
        fb = ErrorFeedback()
        fb.update_marker_status(False, now=0.0)
        fb.update_marker_status(True, now=2.0)
        assert fb.get_user_message(now=5.0) is None


class TestConnectionStatus:
    def test_connection_lost_immediate_message(self):
        fb = ErrorFeedback()
        fb.update_connection_status(False, now=0.0)
        msg = fb.get_user_message(now=0.1)
        assert msg is not None
        assert "connection" in msg.lower()

    def test_connection_restored_clears_error(self):
        fb = ErrorFeedback()
        fb.update_connection_status(False, now=0.0)
        fb.update_connection_status(True, now=1.0)
        assert fb.get_user_message(now=2.0) is None


class TestCameraStatus:
    def test_camera_within_grace_no_message(self):
        fb = ErrorFeedback()
        fb.update_camera_status(False, now=0.0)
        assert fb.get_user_message(now=4.0) is None  # within 5s grace

    def test_camera_past_grace_shows_message(self):
        fb = ErrorFeedback()
        fb.update_camera_status(False, now=0.0)
        msg = fb.get_user_message(now=6.0)
        assert msg is not None
        assert "camera" in msg.lower()


class TestGeminiStatus:
    def test_gemini_within_grace_no_message(self):
        fb = ErrorFeedback()
        fb.update_gemini_status(True, now=0.0)
        assert fb.get_user_message(now=9.0) is None  # within 10s grace

    def test_gemini_past_grace_shows_message(self):
        fb = ErrorFeedback()
        fb.update_gemini_status(True, now=0.0)
        msg = fb.get_user_message(now=11.0)
        assert msg is not None
        assert "gemini" in msg.lower() or "ai" in msg.lower()


class TestPriority:
    def test_connection_has_highest_priority(self):
        fb = ErrorFeedback()
        fb.update_connection_status(False, now=0.0)
        fb.update_marker_status(False, now=0.0)
        msg = fb.get_user_message(now=10.0)
        assert "connection" in msg.lower()

    def test_no_errors_returns_none(self):
        fb = ErrorFeedback()
        assert fb.get_user_message(now=0.0) is None


class TestOverlay:
    def test_overlay_generated_when_error(self):
        fb = ErrorFeedback()
        fb.update_connection_status(False, now=0.0)
        result = fb.get_overlay(now=0.1)
        assert result is not None
        img, placement = result
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3
        assert len(placement) == 4

    def test_no_overlay_when_no_error(self):
        fb = ErrorFeedback()
        assert fb.get_overlay(now=0.0) is None

    def test_overlay_reserved_name(self):
        """Error overlay should use the __system_error reserved name."""
        fb = ErrorFeedback()
        assert fb.OVERLAY_NAME == "__system_error"
