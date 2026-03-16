"""Tests for session persistence across restarts (Feature 11)."""

import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from client.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(session_dir=str(tmp_path / "session"))


# --- save_overlay_state / load_overlay_state ---


def test_save_overlay_state_creates_file(store):
    """save_overlay_state writes session/state.json."""
    import os

    state = {"overlays": [{"name": "graph1", "placement": [0, 0, 500, 500]}]}
    store.save_overlay_state(state)
    assert os.path.exists(os.path.join(store.session_dir, "state.json"))


def test_save_and_load_overlay_state_roundtrip(store):
    """Saved overlay state loads back identically."""
    state = {
        "overlays": [
            {"name": "graph1", "content_type": "graph", "placement": [0, 0, 500, 500]},
            {"name": "hint", "content_type": "text", "placement": [500, 0, 1000, 500]},
        ],
        "count": 2,
    }
    store.save_overlay_state(state)
    loaded = store.load_overlay_state()
    assert loaded == state


def test_load_overlay_state_missing_file(store):
    """Returns empty dict when state.json doesn't exist."""
    assert store.load_overlay_state() == {}


def test_save_overlay_state_overwrites(store):
    """Second save replaces the first."""
    store.save_overlay_state({"overlays": [{"name": "a"}]})
    store.save_overlay_state({"overlays": [{"name": "b"}]})
    loaded = store.load_overlay_state()
    assert loaded == {"overlays": [{"name": "b"}]}


# --- save_scene_order / load_scene_order ---


def test_save_and_load_scene_order(store):
    """Scene order saves and loads back in order."""
    order = ["scene-intro", "scene-graph", "scene-summary"]
    store.save_scene_order(order)
    assert store.load_scene_order() == order


def test_load_scene_order_missing_file(store):
    """Returns empty list when state.json doesn't exist."""
    assert store.load_scene_order() == []


def test_scene_order_merges_with_overlay_state(store):
    """save_scene_order preserves existing overlay state in state.json."""
    state = {"overlays": [{"name": "g1"}], "count": 1}
    store.save_overlay_state(state)
    store.save_scene_order(["s1", "s2"])
    loaded = store.load_overlay_state()
    assert loaded["overlays"] == [{"name": "g1"}]
    assert store.load_scene_order() == ["s1", "s2"]


def test_overlay_state_preserves_scene_order(store):
    """save_overlay_state preserves existing scene_order in state.json."""
    store.save_scene_order(["s1", "s2"])
    store.save_overlay_state({"overlays": [{"name": "g1"}], "count": 1})
    assert store.load_scene_order() == ["s1", "s2"]


# --- restore_session_state ---


def test_restore_session_state_returns_overlay_count(store):
    """restore_session_state returns the number of overlays restored."""
    from client.session_restore import restore_session_state

    state = {
        "overlays": [
            {
                "name": "graph1",
                "content_type": "graph",
                "placement": [0, 0, 500, 500],
                "title": "My Graph",
                "data": {"expression": "x"},
            },
            {
                "name": "hint",
                "content_type": "annotation",
                "placement": [500, 0, 1000, 500],
                "title": "Hint",
                "data": {"text": "hello"},
            },
        ],
        "count": 2,
    }
    store.save_overlay_state(state)

    overlay_manager = MagicMock()

    count = restore_session_state(store, overlay_manager)
    assert count == 2
    # Verify render_overlay + _show_overlay were called for each non-image overlay
    assert overlay_manager.render_overlay.call_count == 2
    assert overlay_manager._show_overlay.call_count == 2


def test_restore_session_state_empty(store):
    """restore_session_state returns 0 when no saved state."""
    from client.session_restore import restore_session_state

    overlay_manager = MagicMock()
    count = restore_session_state(store, overlay_manager)
    assert count == 0


def test_save_session_state(store):
    """save_session_state persists overlay state and scene order."""
    from client.session_restore import save_session_state

    overlay_state = MagicMock()
    overlay_state.to_json.return_value = {
        "overlays": [{"name": "g1", "content_type": "graph"}],
        "count": 1,
    }
    scene_order = ["s1", "s2"]

    save_session_state(store, overlay_state, scene_order)

    loaded = store.load_overlay_state()
    assert loaded["count"] == 1
    assert store.load_scene_order() == ["s1", "s2"]


# --- DebouncedSaver ---


def test_debounced_saver_flush(store):
    """flush() forces immediate save."""
    from client.session_restore import DebouncedSaver

    call_count = 0

    def save_fn():
        nonlocal call_count
        call_count += 1

    saver = DebouncedSaver(save_fn, delay=10.0)
    saver.trigger()
    saver.flush()
    assert call_count == 1


def test_debounced_saver_no_trigger_no_save():
    """flush() with no pending trigger does nothing."""
    from client.session_restore import DebouncedSaver

    call_count = 0

    def save_fn():
        nonlocal call_count
        call_count += 1

    saver = DebouncedSaver(save_fn, delay=10.0)
    saver.flush()
    assert call_count == 0


def test_debounced_saver_delays_save():
    """Trigger doesn't save immediately; save happens after delay."""
    from client.session_restore import DebouncedSaver

    call_count = 0

    def save_fn():
        nonlocal call_count
        call_count += 1

    saver = DebouncedSaver(save_fn, delay=0.1)
    saver.trigger()
    # Should not have saved yet
    assert call_count == 0
    # Wait for debounce
    time.sleep(0.25)
    assert call_count == 1


def test_debounced_saver_coalesces_triggers():
    """Multiple rapid triggers result in single save."""
    from client.session_restore import DebouncedSaver

    call_count = 0

    def save_fn():
        nonlocal call_count
        call_count += 1

    saver = DebouncedSaver(save_fn, delay=0.15)
    saver.trigger()
    time.sleep(0.05)
    saver.trigger()
    time.sleep(0.05)
    saver.trigger()
    # Wait for debounce after last trigger
    time.sleep(0.3)
    assert call_count == 1
