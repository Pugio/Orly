"""Tests for AnimatedOverlay and animated loading placeholder."""

import time
import threading

import numpy as np
from unittest.mock import MagicMock, patch

from client.animated_overlay import AnimatedOverlay
from client.renderer.image import render_loading_frame


# ---------------------------------------------------------------------------
# render_loading_frame unit tests
# ---------------------------------------------------------------------------


class TestRenderLoadingFrame:
    """Test the animated loading frame generator."""

    def test_output_shape(self):
        img = render_loading_frame(0.0, 640, 480, "test")
        assert img.shape == (480, 640, 3)

    def test_dtype(self):
        img = render_loading_frame(0.0, 640, 480, "test")
        assert img.dtype == np.uint8

    def test_has_content_at_zero(self):
        """Even at t=0 should show border and text."""
        img = render_loading_frame(0.0, 640, 480, "test")
        assert img.max() > 0

    def test_fill_increases_over_time(self):
        """More pixels should be lit as elapsed increases (water rises)."""
        early = render_loading_frame(5.0, 640, 480, "test")
        late = render_loading_frame(50.0, 640, 480, "test")
        early_lit = (early.sum(axis=2) > 0).sum()
        late_lit = (late.sum(axis=2) > 0).sum()
        assert late_lit > early_lit

    def test_fill_at_zero_elapsed(self):
        """At t=0 the water fill region should be empty (only border/text)."""
        at_zero = render_loading_frame(0.0, 640, 480, "test")
        at_ten = render_loading_frame(10.0, 640, 480, "test")
        # The interior fill area should have more lit pixels at t=10 than t=0
        # (border region is the same, only fill differs)
        zero_lit = (at_zero.sum(axis=2) > 0).sum()
        ten_lit = (at_ten.sum(axis=2) > 0).sum()
        assert ten_lit > zero_lit

    def test_fill_at_sixty_seconds(self):
        """At t=60 the fill should reach the top."""
        img = render_loading_frame(60.0, 640, 480, "test")
        # A large portion of the bottom half should be lit
        bottom_half = img[240:, :, :]
        lit_pixels = (bottom_half.sum(axis=2) > 0).sum()
        total_pixels = bottom_half.shape[0] * bottom_half.shape[1]
        assert lit_pixels > total_pixels * 0.5

    def test_small_dimensions(self):
        """Should not crash on small overlay sizes."""
        img = render_loading_frame(10.0, 100, 80, "tiny")
        assert img.shape == (80, 100, 3)

    def test_long_prompt_truncated(self):
        """Long prompts should not crash."""
        long_prompt = "x" * 200
        img = render_loading_frame(5.0, 400, 300, long_prompt)
        assert img.shape == (300, 400, 3)


# ---------------------------------------------------------------------------
# AnimatedOverlay unit tests
# ---------------------------------------------------------------------------


def _make_om():
    """Create a minimal mock OverlayManager."""
    om = MagicMock()
    om.proj_width = 1280
    om.proj_height = 720
    om.transform = MagicMock()
    om.transform.orient_overlay = lambda x: x
    om.transform.placement_pixel_size = lambda p: (
        int((p[3] - p[1]) / 1000 * 1280),
        int((p[2] - p[0]) / 1000 * 720),
    )
    return om


class TestAnimatedOverlayStartStop:
    """Test animation lifecycle."""

    def test_start_creates_animation(self):
        om = _make_om()
        anim = AnimatedOverlay(om)
        called = threading.Event()

        def frame_fn(elapsed, w, h):
            called.set()
            return np.zeros((h, w, 3), dtype=np.uint8)

        anim.start("test", frame_fn, [0, 0, 1000, 1000])
        assert called.wait(timeout=2.0), "frame_fn should be called"
        anim.stop("test")

    def test_stop_returns_true_if_running(self):
        om = _make_om()
        anim = AnimatedOverlay(om)

        def frame_fn(elapsed, w, h):
            return np.zeros((h, w, 3), dtype=np.uint8)

        anim.start("test", frame_fn, [0, 0, 1000, 1000])
        time.sleep(0.05)
        assert anim.stop("test") is True

    def test_stop_returns_false_if_not_running(self):
        om = _make_om()
        anim = AnimatedOverlay(om)
        assert anim.stop("nonexistent") is False

    def test_stop_all(self):
        om = _make_om()
        anim = AnimatedOverlay(om)

        def frame_fn(elapsed, w, h):
            return np.zeros((h, w, 3), dtype=np.uint8)

        anim.start("a", frame_fn, [0, 0, 500, 500])
        anim.start("b", frame_fn, [500, 500, 1000, 1000])
        time.sleep(0.05)
        anim.stop_all()
        # Both should be gone
        assert anim.stop("a") is False
        assert anim.stop("b") is False

    def test_start_replaces_existing(self):
        """Starting an animation with the same name stops the old one."""
        om = _make_om()
        anim = AnimatedOverlay(om)
        call_count = {"a": 0, "b": 0}

        def frame_fn_a(elapsed, w, h):
            call_count["a"] += 1
            return np.zeros((h, w, 3), dtype=np.uint8)

        def frame_fn_b(elapsed, w, h):
            call_count["b"] += 1
            return np.zeros((h, w, 3), dtype=np.uint8)

        anim.start("test", frame_fn_a, [0, 0, 1000, 1000])
        time.sleep(0.1)
        anim.start("test", frame_fn_b, [0, 0, 1000, 1000])
        time.sleep(0.1)
        anim.stop("test")

        assert call_count["b"] > 0, "Second frame_fn should have been called"

    def test_frame_fn_receives_increasing_elapsed(self):
        """elapsed parameter should increase over calls."""
        om = _make_om()
        anim = AnimatedOverlay(om)
        elapsed_values = []

        def frame_fn(elapsed, w, h):
            elapsed_values.append(elapsed)
            return np.zeros((h, w, 3), dtype=np.uint8)

        anim.start("test", frame_fn, [0, 0, 1000, 1000], fps=30)
        time.sleep(0.15)
        anim.stop("test")

        assert len(elapsed_values) >= 2
        assert elapsed_values[-1] > elapsed_values[0]

    def test_updates_canvas(self):
        """Animation should call _show_overlay on the overlay manager."""
        om = _make_om()
        anim = AnimatedOverlay(om)

        def frame_fn(elapsed, w, h):
            return np.ones((h, w, 3), dtype=np.uint8) * 128

        anim.start("test", frame_fn, [0, 0, 1000, 1000])
        time.sleep(0.1)
        anim.stop("test")

        assert om._show_overlay.called

    def test_is_running(self):
        om = _make_om()
        anim = AnimatedOverlay(om)

        def frame_fn(elapsed, w, h):
            return np.zeros((h, w, 3), dtype=np.uint8)

        assert not anim.is_running("test")
        anim.start("test", frame_fn, [0, 0, 1000, 1000])
        time.sleep(0.05)
        assert anim.is_running("test")
        anim.stop("test")
        time.sleep(0.05)
        assert not anim.is_running("test")


# ---------------------------------------------------------------------------
# Integration: overlay_manager uses animated loading
# ---------------------------------------------------------------------------


class TestOverlayManagerAnimatedLoading:
    """Test that overlay_manager starts/stops loading animation for images."""

    def _make_real_om(self):
        from client.overlay_manager import OverlayManager
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        return om

    def test_image_create_starts_animation(self):
        om = self._make_real_om()
        result = {
            "action": "create",
            "content_type": "image",
            "placement": [0, 0, 1000, 1000],
            "title": "test_img",
            "data": {"prompt": "a cat"},
        }
        with patch("client.overlay_manager.render_image") as mock_gen:
            # Make generation block so we can check animation is running
            mock_gen.side_effect = lambda *a, **kw: (
                time.sleep(0.3) or np.zeros((480, 640, 3), dtype=np.uint8)
            )
            om.handle_tool_result("overlay", result)
            time.sleep(0.1)
            assert om.animated is not None
            assert om.animated.is_running("test_img")
            # Wait for generation to finish
            time.sleep(0.5)

    def test_image_complete_stops_animation_and_shows_image(self):
        om = self._make_real_om()
        result = {
            "action": "create",
            "content_type": "image",
            "placement": [0, 0, 1000, 1000],
            "title": "test_img",
            "data": {"prompt": "a cat"},
        }
        # Return a bright green image so we can distinguish it from the
        # loading animation (which uses cyan/teal).
        green_img = np.zeros((480, 640, 3), dtype=np.uint8)
        green_img[:, :, 1] = 255  # BGR green
        with patch("client.overlay_manager.render_image") as mock_gen:
            mock_gen.return_value = green_img
            om.handle_tool_result("overlay", result)
            time.sleep(0.3)
            # Animation should have stopped after generation completed
            assert not om.animated.is_running("test_img")
            # The real image (green) must be on canvas, not the loading
            # animation. Check the green channel dominates.
            assert om.canvas[:, :, 1].max() == 255, (
                "Generated image should be visible on canvas"
            )

    def test_generation_failure_stops_animation(self):
        om = self._make_real_om()
        result = {
            "action": "create",
            "content_type": "image",
            "placement": [0, 0, 1000, 1000],
            "title": "test_img",
            "data": {"prompt": "a cat"},
        }
        with patch("client.overlay_manager.render_image") as mock_gen:
            mock_gen.side_effect = RuntimeError("API down")
            om.handle_tool_result("overlay", result)
            time.sleep(0.3)
            # Animation must stop even when generation raises
            assert not om.animated.is_running("test_img")

    def test_clear_stops_animation(self):
        om = self._make_real_om()
        result = {
            "action": "create",
            "content_type": "image",
            "placement": [0, 0, 1000, 1000],
            "title": "test_img",
            "data": {"prompt": "a cat"},
        }
        with patch("client.overlay_manager.render_image") as mock_gen:
            mock_gen.side_effect = lambda *a, **kw: (
                time.sleep(1.0) or np.zeros((480, 640, 3), dtype=np.uint8)
            )
            om.handle_tool_result("overlay", result)
            time.sleep(0.1)
            assert om.animated.is_running("test_img")
            om.clear()
            time.sleep(0.05)
            assert not om.animated.is_running("test_img")
