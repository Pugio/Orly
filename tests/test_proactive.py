"""Tests for proactive tutoring: cooldown logic, system prompt, and message parsing."""

from __future__ import annotations

import json
import os
import time


def test_proactive_cooldown_init():
    """New ProactiveCooldown with default cooldown — should_suppress() returns False."""
    from backend.proactive import ProactiveCooldown

    cd = ProactiveCooldown(cooldown_secs=15)
    assert cd.should_suppress() is False


def test_proactive_cooldown_within_window():
    """Speech recorded at T, check at T+5 — should suppress (within 15s window)."""
    from backend.proactive import ProactiveCooldown

    cd = ProactiveCooldown(cooldown_secs=15)
    t = time.time()
    cd.record(now=t)
    assert cd.should_suppress(now=t + 5) is True


def test_proactive_cooldown_after_window():
    """Speech recorded at T, check at T+20 — should NOT suppress (past 15s window)."""
    from backend.proactive import ProactiveCooldown

    cd = ProactiveCooldown(cooldown_secs=15)
    t = time.time()
    cd.record(now=t)
    assert cd.should_suppress(now=t + 20) is False


def test_proactive_cooldown_record_resets():
    """Past window, record() called, should_suppress immediately returns True."""
    from backend.proactive import ProactiveCooldown

    cd = ProactiveCooldown(cooldown_secs=15)
    t = time.time()
    cd.record(now=t)
    # Past the window
    assert cd.should_suppress(now=t + 20) is False
    # Record again
    t2 = t + 20
    cd.record(now=t2)
    assert cd.should_suppress(now=t2) is True


def test_proactive_cooldown_disabled():
    """When enabled=False, should_suppress always returns True."""
    from backend.proactive import ProactiveCooldown

    cd = ProactiveCooldown(cooldown_secs=15)
    cd.enabled = False
    # Even with no prior record, disabled means suppress
    assert cd.should_suppress() is True


def test_proactive_cooldown_env_var(monkeypatch):
    """PROACTIVE_COOLDOWN_SECS env var is respected when no arg given."""
    from backend.proactive import ProactiveCooldown

    monkeypatch.setenv("PROACTIVE_COOLDOWN_SECS", "30")
    cd = ProactiveCooldown()
    assert cd.cooldown_secs == 30.0

    t = time.time()
    cd.record(now=t)
    # Within 30s window
    assert cd.should_suppress(now=t + 25) is True
    # Past 30s window
    assert cd.should_suppress(now=t + 35) is False


def test_proactive_system_prompt_contains_observation_section():
    """SYSTEM_PROMPT must contain a PROACTIVE OBSERVATION section."""
    from backend.agent import SYSTEM_PROMPT

    assert "PROACTIVE OBSERVATION" in SYSTEM_PROMPT


def test_proactive_toggle_message_parsing():
    """{"type": "set_proactive", "enabled": false} parsed correctly."""
    from backend.main import parse_text_message

    msg_type, payload = parse_text_message(
        json.dumps({"type": "set_proactive", "enabled": False})
    )
    assert msg_type == "set_proactive"
    assert payload == "false"


def test_proactive_toggle_message_enabled():
    """{"type": "set_proactive", "enabled": true} parsed correctly."""
    from backend.main import parse_text_message

    msg_type, payload = parse_text_message(
        json.dumps({"type": "set_proactive", "enabled": True})
    )
    assert msg_type == "set_proactive"
    assert payload == "true"
