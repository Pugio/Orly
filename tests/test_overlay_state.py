"""Tests for OverlayStateManager — named overlay tracking."""

import numpy as np
import pytest
import cv2

from client.overlay_state import OverlayStateManager, OverlayEntry


class MockOverlayManager:
    def __init__(self):
        self.proj_width = 1280
        self.proj_height = 720
        self.mode = "screen"
        self._has_content = False
        self.canvas = self._make_bg()

    def _make_bg(self):
        return np.zeros((self.proj_height, self.proj_width, 3), dtype=np.uint8)

    def clear(self):
        self.canvas = self._make_bg()
        self._has_content = False

    def place_on_canvas(self, overlay, placement):
        ymin, xmin, ymax, xmax = placement
        canvas = self.canvas.copy()
        px_min = max(0, int(xmin / 1000.0 * self.proj_width))
        py_min = max(0, int(ymin / 1000.0 * self.proj_height))
        px_max = min(self.proj_width, int(xmax / 1000.0 * self.proj_width))
        py_max = min(self.proj_height, int(ymax / 1000.0 * self.proj_height))
        if px_max > px_min and py_max > py_min:
            resized = cv2.resize(overlay, (px_max - px_min, py_max - py_min))
            canvas[py_min:py_max, px_min:px_max] = resized
        return canvas


def _make_image(color=(255, 255, 255)):
    """Create a small test overlay image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = color
    return img


@pytest.fixture
def state():
    om = MockOverlayManager()
    return OverlayStateManager(om)


class TestAddOverlay:
    def test_add_overlay(self, state):
        img = _make_image()
        state.add("graph1", "graph", [0, 0, 500, 500], "My Graph", {"expr": "x^2"}, img)
        entry = state.get("graph1")
        assert entry is not None
        assert entry.name == "graph1"
        assert entry.content_type == "graph"
        assert entry.placement == [0, 0, 500, 500]
        assert entry.title == "My Graph"
        assert entry.data == {"expr": "x^2"}
        assert entry.image.shape == img.shape
        assert entry.created_at > 0

    def test_add_replaces_existing(self, state):
        img1 = _make_image((255, 0, 0))
        img2 = _make_image((0, 255, 0))
        state.add("overlay", "graph", [0, 0, 500, 500], "v1", {}, img1)
        state.add("overlay", "annotation", [100, 100, 600, 600], "v2", {}, img2)
        assert len(state.list_names()) == 1
        entry = state.get("overlay")
        assert entry.content_type == "annotation"
        assert entry.title == "v2"


class TestRemoveOverlay:
    def test_remove_existing(self, state):
        state.add("x", "graph", [0, 0, 500, 500], "t", {}, _make_image())
        assert state.remove("x") is True
        assert state.get("x") is None

    def test_remove_nonexistent(self, state):
        assert state.remove("nope") is False


class TestClearAndList:
    def test_clear(self, state):
        for i in range(3):
            state.add(f"o{i}", "graph", [0, 0, 500, 500], "t", {}, _make_image())
        state.clear()
        assert state.list_names() == []

    def test_list_names(self, state):
        state.add("alpha", "graph", [0, 0, 100, 100], "a", {}, _make_image())
        state.add("beta", "annotation", [100, 100, 200, 200], "b", {}, _make_image())
        state.add("gamma", "highlight", [200, 200, 300, 300], "c", {}, _make_image())
        names = state.list_names()
        assert set(names) == {"alpha", "beta", "gamma"}
        assert len(names) == 3


class TestToJson:
    def test_to_json_empty(self, state):
        result = state.to_json()
        assert result["overlays"] == []
        assert result["count"] == 0
        assert result["dimensions"] == [1000, 1000]

    def test_to_json_with_overlays(self, state):
        state.add("g1", "graph", [0, 0, 500, 500], "Graph 1", {"expr": "x"}, _make_image())
        state.add("a1", "annotation", [500, 500, 1000, 1000], "Note", {"text": "hi"}, _make_image())
        result = state.to_json()
        assert result["count"] == 2
        assert len(result["overlays"]) == 2
        entry = result["overlays"][0]
        assert "name" in entry
        assert "content_type" in entry
        assert "placement" in entry
        assert "title" in entry
        assert "created_at" in entry
        # image and data should NOT be in JSON (not JSON-safe)
        assert "image" not in entry
        assert "data" not in entry


class TestToAscii:
    def test_to_ascii_empty(self, state):
        ascii_grid = state.to_ascii(width=10, height=5)
        lines = ascii_grid.split("\n")
        assert len(lines) == 5
        for line in lines:
            assert line == ".........."

    def test_to_ascii_single_overlay(self, state):
        # Place overlay in top-left quadrant (0-500, 0-500)
        state.add("graph", "graph", [0, 0, 500, 500], "t", {}, _make_image())
        ascii_grid = state.to_ascii(width=10, height=10)
        lines = ascii_grid.split("\n")
        # Top-left 5x5 should be 'g', rest '.'
        for r in range(5):
            for c in range(10):
                if c < 5:
                    assert lines[r][c] == "g", f"Expected 'g' at ({r},{c})"
                else:
                    assert lines[r][c] == ".", f"Expected '.' at ({r},{c})"
        for r in range(5, 10):
            assert lines[r] == "..........", f"Row {r} should be all dots"

    def test_to_ascii_overlap(self, state):
        state.add("alpha", "graph", [0, 0, 500, 500], "a", {}, _make_image())
        state.add("beta", "annotation", [250, 250, 750, 750], "b", {}, _make_image())
        ascii_grid = state.to_ascii(width=20, height=20)
        lines = ascii_grid.split("\n")
        # The overlap region (250-500, 250-500) mapped to grid should have '#'
        # row 5-9, col 5-9 in a 20x20 grid
        assert lines[6][6] == "#"

    def test_to_ascii_multiple_non_overlapping(self, state):
        state.add("alpha", "graph", [0, 0, 300, 300], "a", {}, _make_image())
        state.add("beta", "annotation", [700, 700, 1000, 1000], "b", {}, _make_image())
        ascii_grid = state.to_ascii(width=10, height=10)
        lines = ascii_grid.split("\n")
        # Top-left should have 'a'
        assert lines[0][0] == "a"
        # Bottom-right should have 'b'
        assert lines[8][8] == "b"
        # Center should be empty
        assert lines[5][5] == "."


class TestRecomposite:
    def test_recomposite_after_remove(self, state):
        white = _make_image((255, 255, 255))
        red = _make_image((0, 0, 255))
        state.add("white_ov", "graph", [0, 0, 500, 500], "w", {}, white)
        state.add("red_ov", "annotation", [500, 500, 1000, 1000], "r", {}, red)
        # Remove white overlay
        state.remove("white_ov")
        canvas = state._om.canvas
        # Top-left region should be black (cleared)
        tl = canvas[0:100, 0:100]
        assert np.all(tl == 0), "Top-left should be black after removing overlay"
        # Bottom-right region should still have red content
        br_y = int(500 / 1000 * 720)
        br_x = int(500 / 1000 * 1280)
        br = canvas[br_y + 10:br_y + 50, br_x + 10:br_x + 50]
        assert np.any(br > 0), "Bottom-right should still have red overlay"

    def test_add_triggers_recomposite(self, state):
        white = _make_image((255, 255, 255))
        state.add("test_ov", "graph", [0, 0, 500, 500], "t", {}, white)
        canvas = state._om.canvas
        # Check that overlay content is on the canvas
        region = canvas[10:50, 10:50]
        assert np.any(region > 0), "Canvas should have content after add"
