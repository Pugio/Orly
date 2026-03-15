"""Proactive tutoring cooldown logic.

Controls how often the agent can make unsolicited observations
about the student's work, preventing over-interruption.
"""

from __future__ import annotations

import os
import time


class ProactiveCooldown:
    def __init__(self, cooldown_secs: float | None = None):
        if cooldown_secs is None:
            cooldown_secs = float(os.environ.get("PROACTIVE_COOLDOWN_SECS", "15"))
        self.cooldown_secs = cooldown_secs
        self._last_ts: float = 0.0
        self.enabled: bool = True

    def should_suppress(self, now: float | None = None) -> bool:
        """Return True if proactive comments should be suppressed right now."""
        if not self.enabled:
            return True
        if now is None:
            now = time.time()
        return (now - self._last_ts) < self.cooldown_secs

    def record(self, now: float | None = None):
        """Record that speech/comment just happened."""
        if now is None:
            now = time.time()
        self._last_ts = now
