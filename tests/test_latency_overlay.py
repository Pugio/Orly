"""Tests for the latency debug overlay renderer."""

from __future__ import annotations

import numpy as np
import pytest

from client.latency_tracker import LatencyTracker
from client.latency_overlay import render_latency_overlay


class TestRenderLatencyOverlay:
    def test_returns_bgr_image(self):
        tracker = LatencyTracker()
        tracker.record("camera", 12.0)
        img = render_latency_overlay(tracker, width=300, height=100)
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 300, 3)
        assert img.dtype == np.uint8

    def test_default_size(self):
        tracker = LatencyTracker()
        tracker.record("camera", 10.0)
        img = render_latency_overlay(tracker)
        assert img.shape[0] > 0
        assert img.shape[1] > 0
        assert img.shape[2] == 3

    def test_empty_tracker(self):
        tracker = LatencyTracker()
        img = render_latency_overlay(tracker, width=200, height=60)
        assert img.shape == (60, 200, 3)

    def test_dark_background(self):
        """Overlay should have a mostly dark background (projector-friendly)."""
        tracker = LatencyTracker()
        tracker.record("camera", 10.0)
        img = render_latency_overlay(tracker, width=300, height=80)
        # Mean brightness should be low (dark bg with bright text)
        assert img.mean() < 100

    def test_has_visible_text(self):
        """Should contain some bright pixels (the text)."""
        tracker = LatencyTracker()
        tracker.record("camera", 50.0)
        tracker.record("encode", 5.0)
        img = render_latency_overlay(tracker, width=400, height=100)
        bright_pixels = np.sum(img > 200)
        assert bright_pixels > 0

    def test_multiple_stages_fit(self):
        tracker = LatencyTracker()
        tracker.record("camera", 10.0)
        tracker.record("encode", 3.0)
        tracker.record("ws_send", 1.5)
        tracker.record("ws_recv", 0.8)
        tracker.record("render", 5.0)
        tracker.record("display", 2.0)
        img = render_latency_overlay(tracker, width=500, height=150)
        assert img.shape == (150, 500, 3)


class TestOverlayComposite:
    def test_composite_onto_canvas(self):
        """Overlay should composite onto a canvas at a given position."""
        from client.latency_overlay import composite_debug_overlay

        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracker = LatencyTracker()
        tracker.record("camera", 10.0)
        overlay = render_latency_overlay(tracker, width=300, height=60)
        result = composite_debug_overlay(canvas, overlay, x=10, y=10)
        # The canvas should have non-zero pixels where overlay was placed
        roi = result[10:70, 10:310]
        assert roi.sum() > 0

    def test_composite_clips_to_bounds(self):
        """Overlay that extends beyond canvas should be clipped, not crash."""
        from client.latency_overlay import composite_debug_overlay

        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker = LatencyTracker()
        tracker.record("camera", 10.0)
        overlay = render_latency_overlay(tracker, width=200, height=50)
        # Place at (50, 60) — extends beyond canvas edges
        result = composite_debug_overlay(canvas, overlay, x=50, y=60)
        assert result.shape == (100, 100, 3)
