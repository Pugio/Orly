"""Pipeline latency tracker — records per-stage timing with rolling averages.

Instrument each stage of the pipeline (camera capture, encode, WS send,
WS receive, render, display) and query real-time stats for debugging.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class LatencyTracker:
    """Tracks per-stage pipeline latencies with rolling window averages.

    Usage:
        tracker = LatencyTracker()

        # Option 1: begin/end
        tracker.begin("camera")
        ...
        tracker.end("camera")

        # Option 2: context manager
        with tracker.track("encode"):
            ...

        # Option 3: record duration directly (ms)
        tracker.record("ws_send", 3.5)

        # Query
        tracker.summary()    # latest values per stage
        tracker.averages()   # rolling averages per stage
        tracker.format_stats()  # human-readable string
    """

    def __init__(self, window_size: int = 60):
        self._window_size = window_size
        self._pending: dict[str, float] = {}  # stage -> start timestamp
        self._history: dict[str, deque[float]] = {}  # stage -> deque of ms values
        self._latest: dict[str, float] = {}  # stage -> latest ms value
        self._log_counter: int = 0

    def begin(self, stage: str, timestamp: float | None = None) -> None:
        """Mark the start of a pipeline stage."""
        self._pending[stage] = timestamp if timestamp is not None else time.monotonic()

    def end(self, stage: str, timestamp: float | None = None) -> None:
        """Mark the end of a pipeline stage."""
        start = self._pending.pop(stage, None)
        if start is None:
            return
        t = timestamp if timestamp is not None else time.monotonic()
        duration_ms = (t - start) * 1000
        self._store(stage, duration_ms)

    def record(self, stage: str, duration_ms: float) -> None:
        """Record a duration directly (in milliseconds)."""
        self._store(stage, duration_ms)

    @contextmanager
    def track(self, stage: str):
        """Context manager for timing a pipeline stage."""
        self.begin(stage)
        try:
            yield
        finally:
            self.end(stage)

    def _store(self, stage: str, duration_ms: float) -> None:
        if stage not in self._history:
            self._history[stage] = deque(maxlen=self._window_size)
        self._history[stage].append(duration_ms)
        self._latest[stage] = duration_ms

    def summary(self) -> dict[str, float]:
        """Latest duration per stage (ms), plus _total."""
        result = dict(self._latest)
        if result:
            result["_total"] = sum(result.values())
        return result

    def averages(self) -> dict[str, float]:
        """Rolling average duration per stage (ms)."""
        result = {}
        for stage, history in self._history.items():
            if history:
                result[stage] = sum(history) / len(history)
        return result

    def format_stats(self) -> str:
        """Human-readable latency stats string for debug display."""
        avgs = self.averages()
        if not avgs:
            return "No latency data"
        parts = []
        for stage, ms in avgs.items():
            parts.append(f"{stage}: {ms:.1f}ms")
        total = sum(avgs.values())
        parts.append(f"total: {total:.1f}ms")
        return " | ".join(parts)

    def log_stats(self) -> None:
        """Emit a structured JSON log line with rolling average latencies."""
        avgs = self.averages()
        if not avgs:
            return
        data = {k: round(v, 1) for k, v in avgs.items()}
        data["_total"] = round(sum(avgs.values()), 1)
        logger.info(json.dumps(data))

    def log_stats_periodic(self, every_n: int = 60) -> None:
        """Call log_stats every N invocations (for use inside hot loops)."""
        self._log_counter += 1
        if self._log_counter % every_n == 0:
            self.log_stats()
