"""Tests for pipeline latency tracking."""

from __future__ import annotations

import time

import pytest

from client.latency_tracker import LatencyTracker


# ===== LatencyTracker core tests =====


class TestLatencyTracker:
    def test_begin_end_records_span(self):
        tracker = LatencyTracker()
        tracker.begin("camera_capture")
        tracker.end("camera_capture")
        summary = tracker.summary()
        assert "camera_capture" in summary
        assert summary["camera_capture"] is not None
        assert summary["camera_capture"] >= 0.0

    def test_begin_end_measures_duration(self):
        tracker = LatencyTracker()
        t0 = time.monotonic()
        tracker.begin("encode", timestamp=t0)
        tracker.end("encode", timestamp=t0 + 0.025)
        summary = tracker.summary()
        assert summary["encode"] == pytest.approx(25.0, abs=1.0)

    def test_unmatched_end_is_ignored(self):
        tracker = LatencyTracker()
        tracker.end("nonexistent")  # should not raise

    def test_multiple_stages(self):
        tracker = LatencyTracker()
        tracker.begin("a", timestamp=0.0)
        tracker.end("a", timestamp=0.01)
        tracker.begin("b", timestamp=0.01)
        tracker.end("b", timestamp=0.03)
        s = tracker.summary()
        assert s["a"] == pytest.approx(10.0)
        assert s["b"] == pytest.approx(20.0)


# ===== Rolling average tests =====


class TestRollingAverage:
    def test_avg_over_window(self):
        tracker = LatencyTracker(window_size=3)
        for i in range(3):
            tracker.begin("x", timestamp=float(i))
            tracker.end("x", timestamp=float(i) + 0.01 * (i + 1))
        avg = tracker.averages()
        # durations: 10, 20, 30 -> avg = 20
        assert avg["x"] == pytest.approx(20.0)

    def test_window_drops_old(self):
        tracker = LatencyTracker(window_size=2)
        # Record 3 spans, oldest should be dropped
        tracker.begin("x", timestamp=0.0)
        tracker.end("x", timestamp=0.1)  # 100ms
        tracker.begin("x", timestamp=1.0)
        tracker.end("x", timestamp=1.01)  # 10ms
        tracker.begin("x", timestamp=2.0)
        tracker.end("x", timestamp=2.02)  # 20ms
        avg = tracker.averages()
        # Only last 2: 10, 20 -> avg = 15
        assert avg["x"] == pytest.approx(15.0)

    def test_averages_empty(self):
        tracker = LatencyTracker()
        avg = tracker.averages()
        assert avg == {}


# ===== Record shorthand tests =====


class TestRecordShorthand:
    def test_record_directly(self):
        tracker = LatencyTracker()
        tracker.record("ws_send", 42.0)
        s = tracker.summary()
        assert s["ws_send"] == pytest.approx(42.0)

    def test_record_contributes_to_average(self):
        tracker = LatencyTracker(window_size=2)
        tracker.record("ws_send", 10.0)
        tracker.record("ws_send", 30.0)
        assert tracker.averages()["ws_send"] == pytest.approx(20.0)


# ===== Format for display =====


class TestFormatDisplay:
    def test_format_stats_string(self):
        tracker = LatencyTracker()
        tracker.record("camera", 12.5)
        tracker.record("encode", 3.2)
        tracker.record("ws_send", 1.1)
        text = tracker.format_stats()
        assert "camera" in text
        assert "encode" in text
        assert "ws_send" in text

    def test_format_stats_empty(self):
        tracker = LatencyTracker()
        text = tracker.format_stats()
        assert isinstance(text, str)

    def test_format_stats_shows_ms(self):
        tracker = LatencyTracker()
        tracker.record("camera", 12.0)
        text = tracker.format_stats()
        assert "ms" in text


# ===== Total pipeline latency =====


class TestTotalPipeline:
    def test_total_from_stages(self):
        tracker = LatencyTracker()
        tracker.record("camera", 10.0)
        tracker.record("encode", 5.0)
        tracker.record("ws_send", 2.0)
        s = tracker.summary()
        assert s["_total"] == pytest.approx(17.0)

    def test_total_empty(self):
        tracker = LatencyTracker()
        s = tracker.summary()
        assert s.get("_total", 0.0) == 0.0


# ===== Context manager =====


class TestContextManager:
    def test_track_context_manager(self):
        tracker = LatencyTracker()
        with tracker.track("encode"):
            pass  # instant
        s = tracker.summary()
        assert "encode" in s
        assert s["encode"] >= 0.0

    def test_track_records_duration(self):
        tracker = LatencyTracker()
        with tracker.track("slow"):
            time.sleep(0.01)
        s = tracker.summary()
        assert s["slow"] >= 5.0  # at least 5ms (sleep jitter)
