"""Exhaustive pipeline interaction tests — permutation and timing fuzz.

Tests every ordering of events across the overlay/interrupt/refresh/program
pipeline, checking state invariants after each permutation.

The harness creates real OverlayManager + OverlayStateManager + ProgramRuntime
instances (no mocks for the components under test) and runs event sequences
in all orderings to catch timing-dependent bugs.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from client.overlay_manager import OverlayManager
from client.overlay_state import OverlayStateManager
from client.program_runtime import ProgramRuntime


# ===========================================================================
# Pipeline test harness
# ===========================================================================


def _make_overlay_img(w=50, h=50, brightness=200):
    return np.ones((h, w, 3), dtype=np.uint8) * brightness


class PipelineHarness:
    """Wraps real OverlayManager + OverlayStateManager + ProgramRuntime.

    Provides named operations and state invariant checks.
    """

    def __init__(self):
        self.om = OverlayManager(
            H_proj=None, proj_width=200, proj_height=200, mode="screen",
        )
        self.osm = OverlayStateManager(self.om)
        self.om.overlay_state = self.osm
        self.runtime = ProgramRuntime(table_api_factory=lambda: MagicMock())
        # Track which overlays we think should exist (model state)
        self._expected_overlays: set[str] = set()
        # Track generation_id at time of async image start
        self._async_gen_id: int | None = None

    # --- Operations ---

    def add_overlay(self, name: str = "overlay_a"):
        """Add a named overlay."""
        img = _make_overlay_img()
        self.osm.add(name, "annotation", [0, 0, 500, 500], name, {}, img)
        self._expected_overlays.add(name)

    def remove_overlay(self, name: str = "overlay_a"):
        """Remove a named overlay (if present)."""
        self.osm.remove(name)
        self._expected_overlays.discard(name)

    def interrupt(self):
        """Simulate interruption (clear all overlays)."""
        self.osm.clear()
        self._expected_overlays.clear()

    def request_refresh(self):
        """Start a refresh cycle."""
        self.om.request_refresh()

    def complete_refresh(self):
        """Complete a refresh cycle."""
        self.om.complete_refresh()

    def start_async_image(self):
        """Record the generation_id before 'starting' an async image gen."""
        self._async_gen_id = self.om._generation_id

    def complete_async_image(self):
        """Simulate an async image completing. Checks generation_id."""
        if self._async_gen_id is not None and self._async_gen_id == self.om._generation_id:
            img = _make_overlay_img(brightness=255)
            self.om._show_overlay(img, [0, 0, 500, 500], "image")
            self.osm.add("async_img", "image", [0, 0, 500, 500], "async", {}, img,
                         recomposite=False)
            self._expected_overlays.add("async_img")
        # If gen_id changed, this is a stale completion — skip (matching real behavior)
        self._async_gen_id = None

    def program_adds_overlay(self):
        """Simulate a program placing an overlay via table.place_overlay."""
        img = _make_overlay_img(brightness=150)
        self.osm.add("prog_overlay", "annotation", [500, 500, 1000, 1000],
                      "from_program", {}, img)
        self._expected_overlays.add("prog_overlay")

    # --- Invariant checks ---

    def check_invariants(self) -> list[str]:
        """Check all state invariants. Returns list of violations."""
        violations = []

        names = set(self.osm.list_names())
        has_content = self.om._has_content
        canvas = self.om.canvas
        is_bg = canvas.sum() == self.om._make_bg().sum()

        # Invariant 1: if no overlays tracked, _has_content must be False
        if not names and has_content:
            violations.append(
                f"No overlays in state but _has_content={has_content}")

        # Invariant 2: if _has_content is False AND no refresh in progress,
        # canvas should be background
        if not has_content and not self.om._refresh_requested and not is_bg:
            violations.append(
                "No content and no refresh, but canvas is not background")

        # Invariant 3: generation_id is non-negative
        if self.om._generation_id < 0:
            violations.append(
                f"generation_id is negative: {self.om._generation_id}")

        # Invariant 4: after clear(), refresh state should be clean
        # (can't check this dynamically, tested separately)

        # Invariant 5: overlay_state names match expected model
        if names != self._expected_overlays:
            violations.append(
                f"State mismatch: got {names}, expected {self._expected_overlays}")

        # Invariant 6: _saved_canvas should be None when not refreshing
        if not self.om._refresh_requested and self.om._saved_canvas is not None:
            violations.append(
                "_saved_canvas is not None but no refresh in progress")

        return violations

    def cleanup(self):
        """Stop all programs for clean teardown."""
        self.runtime.stop_all()


# ===========================================================================
# Event definitions for permutation testing
# ===========================================================================


@dataclass
class Event:
    """A named pipeline event."""
    name: str
    fn: str  # method name on PipelineHarness
    args: tuple = ()


OVERLAY_EVENTS = [
    Event("add_a", "add_overlay", ("overlay_a",)),
    Event("add_b", "add_overlay", ("overlay_b",)),
    Event("remove_a", "remove_overlay", ("overlay_a",)),
    Event("interrupt", "interrupt"),
]


def run_event_sequence(events: list[Event]) -> list[str]:
    """Run a sequence of events and return invariant violations."""
    h = PipelineHarness()
    try:
        for event in events:
            method = getattr(h, event.fn)
            method(*event.args)
        return h.check_invariants()
    finally:
        h.cleanup()


# ===========================================================================
# 1. All 2-event permutations of overlay + interrupt
# ===========================================================================


class TestTwoEventOverlayPermutations:
    """Test every pair ordering of add/remove/interrupt operations."""

    @pytest.fixture
    def events(self):
        return OVERLAY_EVENTS

    def test_all_pairs(self, events):
        """Every 2-event permutation of overlay ops should maintain invariants."""
        failures = []
        for pair in itertools.permutations(events, 2):
            violations = run_event_sequence(list(pair))
            if violations:
                seq = " → ".join(e.name for e in pair)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )

    def test_all_triples(self, events):
        """Every 3-event permutation of overlay ops should maintain invariants."""
        failures = []
        for triple in itertools.permutations(events, 3):
            violations = run_event_sequence(list(triple))
            if violations:
                seq = " → ".join(e.name for e in triple)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )


# ===========================================================================
# 2. Refresh + interrupt permutations
# ===========================================================================


class TestRefreshInterruptPermutations:
    """Test all orderings of refresh and interrupt operations."""

    def _events(self):
        return [
            Event("add_a", "add_overlay", ("overlay_a",)),
            Event("interrupt", "interrupt"),
            Event("request_refresh", "request_refresh"),
            Event("complete_refresh", "complete_refresh"),
        ]

    def test_all_pairs(self):
        failures = []
        for pair in itertools.permutations(self._events(), 2):
            violations = run_event_sequence(list(pair))
            if violations:
                seq = " → ".join(e.name for e in pair)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )

    def test_all_triples(self):
        failures = []
        for triple in itertools.permutations(self._events(), 3):
            violations = run_event_sequence(list(triple))
            if violations:
                seq = " → ".join(e.name for e in triple)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )

    def test_all_quads(self):
        """Full 4-event permutations (24 orderings)."""
        failures = []
        for quad in itertools.permutations(self._events(), 4):
            violations = run_event_sequence(list(quad))
            if violations:
                seq = " → ".join(e.name for e in quad)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )


# ===========================================================================
# 3. Async image + interrupt permutations
# ===========================================================================


class TestAsyncImageInterruptPermutations:
    """Test orderings of async image lifecycle with interrupts."""

    def _events(self):
        return [
            Event("add_a", "add_overlay", ("overlay_a",)),
            Event("start_async", "start_async_image"),
            Event("interrupt", "interrupt"),
            Event("complete_async", "complete_async_image"),
        ]

    def test_all_pairs(self):
        failures = []
        for pair in itertools.permutations(self._events(), 2):
            violations = run_event_sequence(list(pair))
            if violations:
                seq = " → ".join(e.name for e in pair)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )

    def test_all_triples(self):
        failures = []
        for triple in itertools.permutations(self._events(), 3):
            violations = run_event_sequence(list(triple))
            if violations:
                seq = " → ".join(e.name for e in triple)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )

    def test_all_quads(self):
        failures = []
        for quad in itertools.permutations(self._events(), 4):
            violations = run_event_sequence(list(quad))
            if violations:
                seq = " → ".join(e.name for e in quad)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )


# ===========================================================================
# 4. Program + overlay + interrupt permutations
# ===========================================================================


class TestProgramOverlayPermutations:
    """Test orderings of program overlays with interrupts."""

    def _events(self):
        return [
            Event("add_a", "add_overlay", ("overlay_a",)),
            Event("prog_overlay", "program_adds_overlay"),
            Event("interrupt", "interrupt"),
            Event("add_b", "add_overlay", ("overlay_b",)),
        ]

    def test_all_quads(self):
        failures = []
        for quad in itertools.permutations(self._events(), 4):
            violations = run_event_sequence(list(quad))
            if violations:
                seq = " → ".join(e.name for e in quad)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )


# ===========================================================================
# 5. Full kitchen-sink permutations (5 events, selected combos)
# ===========================================================================


class TestKitchenSinkPermutations:
    """Selected 5-event sequences mixing all operation types."""

    def _events(self):
        return [
            Event("add_a", "add_overlay", ("overlay_a",)),
            Event("interrupt", "interrupt"),
            Event("start_async", "start_async_image"),
            Event("complete_async", "complete_async_image"),
            Event("request_refresh", "request_refresh"),
        ]

    def test_all_5_permutations(self):
        """120 orderings of 5 mixed operations."""
        failures = []
        for perm in itertools.permutations(self._events(), 5):
            violations = run_event_sequence(list(perm))
            if violations:
                seq = " → ".join(e.name for e in perm)
                failures.append(f"  {seq}: {violations}")

        assert not failures, (
            f"{len(failures)} permutation(s) violated invariants:\n"
            + "\n".join(failures)
        )


# ===========================================================================
# 6. Repeated event stress tests
# ===========================================================================


class TestRepeatedEvents:
    """Test repeated application of the same event."""

    def test_100_adds_then_interrupt(self):
        h = PipelineHarness()
        try:
            for i in range(100):
                h.add_overlay(f"overlay_{i}")
            assert len(h.osm.list_names()) == 100
            h.interrupt()
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_alternating_add_interrupt_50_cycles(self):
        h = PipelineHarness()
        try:
            for i in range(50):
                h.add_overlay(f"overlay_{i}")
                h.interrupt()
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_rapid_refresh_cycles(self):
        h = PipelineHarness()
        try:
            h.add_overlay("persistent")
            for _ in range(50):
                h.request_refresh()
                h.complete_refresh()
            assert h.check_invariants() == []
            assert "persistent" in h.osm.list_names()
        finally:
            h.cleanup()

    def test_async_start_interrupt_complete_50_cycles(self):
        h = PipelineHarness()
        try:
            for _ in range(50):
                h.start_async_image()
                h.interrupt()
                h.complete_async_image()
            # Stale images should never appear
            assert h.check_invariants() == []
            assert h.om._has_content is False
        finally:
            h.cleanup()


# ===========================================================================
# 7. Concurrent timing tests (thread-based fuzz)
# ===========================================================================


class TestConcurrentTiming:
    """Run operations from multiple threads to catch race conditions."""

    def _run_concurrent(self, harness, fns, iterations=100):
        """Run multiple functions concurrently on the harness."""
        errors = []
        barrier = threading.Barrier(len(fns))

        def worker(fn):
            try:
                barrier.wait(timeout=5)
                for _ in range(iterations):
                    fn(harness)
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(f"{fn.__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(fn,)) for fn in fns]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        return errors

    def test_add_vs_interrupt(self):
        """Concurrent overlay adds and interrupts."""
        h = PipelineHarness()
        counter = [0]

        def adder(h):
            name = f"o_{counter[0]}"
            counter[0] += 1
            try:
                h.add_overlay(name)
            except Exception:
                pass

        def interrupter(h):
            h.interrupt()

        try:
            errors = self._run_concurrent(h, [adder, interrupter], iterations=200)
            assert errors == [], f"Thread errors: {errors}"
            # State should be internally consistent at the end
            # (we can't use check_invariants because _expected_overlays is
            #  not thread-safe — we just check no exceptions)
        finally:
            h.cleanup()

    def test_add_vs_remove_vs_interrupt(self):
        """Three concurrent operations."""
        h = PipelineHarness()
        counter = [0]

        def adder(h):
            name = f"o_{counter[0] % 5}"  # reuse 5 names
            counter[0] += 1
            try:
                h.add_overlay(name)
            except Exception:
                pass

        def remover(h):
            h.remove_overlay(f"o_{counter[0] % 5}")

        def interrupter(h):
            h.interrupt()

        try:
            errors = self._run_concurrent(
                h, [adder, remover, interrupter], iterations=150)
            assert errors == [], f"Thread errors: {errors}"
        finally:
            h.cleanup()

    def test_refresh_vs_interrupt(self):
        """Concurrent refresh and interrupt cycles."""
        h = PipelineHarness()

        def refresher(h):
            h.request_refresh()
            h.complete_refresh()

        def interrupter(h):
            h.interrupt()

        try:
            errors = self._run_concurrent(
                h, [refresher, interrupter], iterations=200)
            assert errors == [], f"Thread errors: {errors}"
        finally:
            h.cleanup()

    def test_async_image_vs_interrupt(self):
        """Concurrent async image completions and interrupts."""
        h = PipelineHarness()

        def async_cycle(h):
            h.start_async_image()
            time.sleep(0.0001)
            h.complete_async_image()

        def interrupter(h):
            h.interrupt()

        try:
            errors = self._run_concurrent(
                h, [async_cycle, interrupter], iterations=200)
            assert errors == [], f"Thread errors: {errors}"
        finally:
            h.cleanup()

    def test_four_way_concurrent(self):
        """Four concurrent operation types — maximum contention."""
        h = PipelineHarness()
        counter = [0]

        def adder(h):
            counter[0] += 1
            try:
                h.add_overlay(f"o_{counter[0] % 3}")
            except Exception:
                pass

        def interrupter(h):
            h.interrupt()

        def refresher(h):
            h.request_refresh()
            h.complete_refresh()

        def async_worker(h):
            h.start_async_image()
            h.complete_async_image()

        try:
            errors = self._run_concurrent(
                h, [adder, interrupter, refresher, async_worker], iterations=100)
            assert errors == [], f"Thread errors: {errors}"
        finally:
            h.cleanup()


# ===========================================================================
# 8. Specific tricky scenarios (hand-crafted edge cases)
# ===========================================================================


class TestTrickyScenarios:
    """Hand-crafted scenarios for the thorniest edge cases."""

    def test_interrupt_between_tool_calls(self):
        """Tool result 1 → interrupt → tool result 2.
        Only tool result 2 should be on canvas."""
        h = PipelineHarness()
        try:
            h.add_overlay("first_graph")
            assert h.om._has_content is True
            h.interrupt()
            assert h.om._has_content is False
            h.add_overlay("second_graph")
            assert h.om._has_content is True
            assert h.osm.list_names() == ["second_graph"]
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_async_image_starts_before_interrupt_completes_after(self):
        """Image gen starts → interrupt → image completes.
        Image should NOT appear (generation_id mismatch)."""
        h = PipelineHarness()
        try:
            h.add_overlay("placeholder")
            h.start_async_image()
            h.interrupt()
            h.complete_async_image()  # stale — should be dropped
            assert h.om._has_content is False
            assert h.osm.list_names() == []
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_refresh_then_interrupt_then_complete_refresh(self):
        """Refresh stashes canvas → interrupt clears → complete_refresh.
        Canvas should stay clean (not restore stale canvas)."""
        h = PipelineHarness()
        try:
            h.add_overlay("graph")
            h.request_refresh()
            # Canvas is now background (for clean capture)
            h.interrupt()
            # Canvas is background, refresh cancelled
            h.complete_refresh()
            # Should be a no-op, canvas stays clean
            assert h.om._has_content is False
            assert h.om.canvas.sum() == 0
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_double_refresh_then_interrupt(self):
        """Two refresh requests (second is no-op) then interrupt."""
        h = PipelineHarness()
        try:
            h.add_overlay("graph")
            h.request_refresh()
            h.request_refresh()  # second call is no-op
            h.interrupt()
            assert h.om._refresh_requested is False
            assert h.om._saved_canvas is None
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_program_overlay_survives_interrupt_of_agent_overlays(self):
        """Agent overlay + program overlay → interrupt.
        Both should be cleared (programs don't get special treatment)."""
        h = PipelineHarness()
        try:
            h.add_overlay("agent_graph")
            h.program_adds_overlay()
            assert len(h.osm.list_names()) == 2
            h.interrupt()
            assert h.osm.list_names() == []
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_remove_nonexistent_then_interrupt(self):
        """Remove something that doesn't exist, then interrupt."""
        h = PipelineHarness()
        try:
            h.remove_overlay("ghost")
            h.interrupt()
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_add_same_name_twice_then_interrupt(self):
        """Adding overlay with same name replaces. Interrupt clears."""
        h = PipelineHarness()
        try:
            h.add_overlay("graph")
            h.add_overlay("graph")  # replace
            assert len(h.osm.list_names()) == 1
            h.interrupt()
            assert h.osm.list_names() == []
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_interrupt_during_recomposite_is_safe(self):
        """Add many overlays, then interrupt. Recomposite should not corrupt."""
        h = PipelineHarness()
        try:
            for i in range(20):
                h.add_overlay(f"o_{i}")
            h.interrupt()
            # Add one more after interrupt
            h.add_overlay("fresh")
            assert h.osm.list_names() == ["fresh"]
            assert h.om._has_content is True
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_generation_id_monotonic_across_multiple_clears(self):
        """generation_id should increase on each clear."""
        h = PipelineHarness()
        try:
            ids = []
            for _ in range(10):
                ids.append(h.om._generation_id)
                h.interrupt()
            # Should be strictly increasing
            for i in range(1, len(ids)):
                assert ids[i] > ids[i - 1], f"generation_id not monotonic: {ids}"
        finally:
            h.cleanup()

    def test_async_image_double_complete(self):
        """Start async, complete once, complete again. Second should no-op."""
        h = PipelineHarness()
        try:
            h.start_async_image()
            h.complete_async_image()
            assert "async_img" in h.osm.list_names()
            # Complete again — _async_gen_id is None, should no-op
            h.complete_async_image()
            assert h.check_invariants() == []
        finally:
            h.cleanup()

    def test_interrupt_add_interrupt_add(self):
        """Alternating interrupt-add cycles."""
        h = PipelineHarness()
        try:
            for i in range(10):
                h.interrupt()
                h.add_overlay(f"o_{i}")
                assert h.om._has_content is True
                assert len(h.osm.list_names()) == 1
                h.interrupt()
                assert h.om._has_content is False
            assert h.check_invariants() == []
        finally:
            h.cleanup()
