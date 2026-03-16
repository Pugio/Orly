"""Tests for structured latency logging."""

from __future__ import annotations

import json
import logging

import pytest

from client.latency_tracker import LatencyTracker


class TestStructuredLogging:
    def test_log_json_output(self, caplog):
        """LatencyTracker.log_stats should emit structured JSON log."""
        tracker = LatencyTracker()
        tracker.record("camera", 12.0)
        tracker.record("encode", 3.0)

        with caplog.at_level(logging.INFO, logger="client.latency_tracker"):
            tracker.log_stats()

        assert len(caplog.records) >= 1
        record = caplog.records[-1]
        # Parse the JSON from the message
        data = json.loads(record.message)
        assert "camera" in data
        assert "encode" in data
        assert data["camera"] == pytest.approx(12.0)
        assert data["encode"] == pytest.approx(3.0)

    def test_log_includes_total(self, caplog):
        tracker = LatencyTracker()
        tracker.record("a", 10.0)
        tracker.record("b", 5.0)

        with caplog.at_level(logging.INFO, logger="client.latency_tracker"):
            tracker.log_stats()

        data = json.loads(caplog.records[-1].message)
        assert "_total" in data
        assert data["_total"] == pytest.approx(15.0)

    def test_log_nothing_when_empty(self, caplog):
        tracker = LatencyTracker()
        with caplog.at_level(logging.INFO, logger="client.latency_tracker"):
            tracker.log_stats()
        # Should not log when there's no data
        assert len(caplog.records) == 0

    def test_log_uses_averages(self, caplog):
        """log_stats should report rolling averages, not just latest."""
        tracker = LatencyTracker(window_size=2)
        tracker.record("x", 10.0)
        tracker.record("x", 30.0)

        with caplog.at_level(logging.INFO, logger="client.latency_tracker"):
            tracker.log_stats()

        data = json.loads(caplog.records[-1].message)
        assert data["x"] == pytest.approx(20.0)


class TestPeriodicLogging:
    def test_log_every_n(self, caplog):
        """log_stats_periodic should only log every N calls."""
        tracker = LatencyTracker()
        tracker.record("x", 10.0)

        with caplog.at_level(logging.INFO, logger="client.latency_tracker"):
            for _ in range(10):
                tracker.log_stats_periodic(every_n=5)

        # Should log twice (at call 5 and 10)
        assert len(caplog.records) == 2

    def test_log_every_n_first_call_no_log(self, caplog):
        """First call should not log (counter starts at 0)."""
        tracker = LatencyTracker()
        tracker.record("x", 10.0)

        with caplog.at_level(logging.INFO, logger="client.latency_tracker"):
            tracker.log_stats_periodic(every_n=100)

        assert len(caplog.records) == 0
