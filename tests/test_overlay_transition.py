"""Tests for TransitionState in client.renderer.transition."""

import time

import numpy as np

from client.renderer.transition import TransitionState, crossfade


class TestTransitionState:
    def test_transition_state_init(self):
        old = np.zeros((100, 100, 3), dtype=np.uint8)
        new = np.full((100, 100, 3), 255, dtype=np.uint8)
        ts = TransitionState(old_frame=old, new_frame=new, duration=0.3)
        assert ts.progress() < 0.01  # essentially zero right after creation

    def test_transition_state_progress_midway(self):
        old = np.zeros((100, 100, 3), dtype=np.uint8)
        new = np.full((100, 100, 3), 255, dtype=np.uint8)
        ts = TransitionState(
            old_frame=old,
            new_frame=new,
            duration=0.3,
            start_time=time.monotonic() - 0.15,
        )
        prog = ts.progress()
        assert 0.4 <= prog <= 0.6  # roughly 0.5

    def test_transition_state_progress_complete(self):
        old = np.zeros((100, 100, 3), dtype=np.uint8)
        new = np.full((100, 100, 3), 255, dtype=np.uint8)
        ts = TransitionState(
            old_frame=old,
            new_frame=new,
            duration=0.3,
            start_time=time.monotonic() - 0.4,
        )
        assert ts.progress() == 1.0

    def test_transition_state_is_done(self):
        old = np.zeros((100, 100, 3), dtype=np.uint8)
        new = np.full((100, 100, 3), 255, dtype=np.uint8)
        ts = TransitionState(
            old_frame=old,
            new_frame=new,
            duration=0.3,
            start_time=time.monotonic() - 0.31,
        )
        assert ts.is_done()

    def test_transition_state_current_frame(self):
        old = np.full((100, 100, 3), 255, dtype=np.uint8)  # white
        new = np.zeros((100, 100, 3), dtype=np.uint8)  # black
        ts = TransitionState(
            old_frame=old,
            new_frame=new,
            duration=0.3,
            start_time=time.monotonic() - 0.15,
        )
        frame = ts.current_frame()
        # At ~50% progress, white→black should yield ~127
        mean_val = frame.mean()
        assert 90 < mean_val < 170  # roughly half
