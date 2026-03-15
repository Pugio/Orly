"""Tests for Bug 2: Calibration (x,y) vs rendering (y,x) axis order.

The calibration defines H_proj with (x,y) input convention.
The overlay_manager must convert [y,x] placements to [x,y] before
applying perspectiveTransform.
"""

import cv2
import numpy as np
import pytest

from client.overlay_manager import OverlayManager


def _calibration_style_H(proj_width=1280, proj_height=720):
    """Create H_proj the way calibration does it: (x,y) → (px,py).

    Maps table (x,y) in 0-1000 to projector pixels.
    x_table maps to x_proj, y_table maps to y_proj.
    """
    # Calibration-style: source points are (x, y)
    src = np.array([
        [0, 0], [1000, 0], [1000, 1000], [0, 1000]
    ], dtype=np.float32)
    dst = np.array([
        [0, 0], [proj_width, 0], [proj_width, proj_height], [0, proj_height]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


class TestAxisConvention:
    def test_top_left_placement_lands_top_left(self):
        """Placement [ymin=0, xmin=0, ymax=200, xmax=200] should land top-left."""
        H = _calibration_style_H()
        mgr = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 200, 200])

        # Content should be in top-left quadrant
        top_left = canvas[:360, :640]
        bottom_right = canvas[360:, 640:]
        assert top_left.sum() > 0, "Content should be in top-left"
        assert bottom_right.sum() == 0, "Bottom-right should be empty"

    def test_bottom_right_placement_lands_bottom_right(self):
        """Placement [ymin=800, xmin=800, ymax=1000, xmax=1000] should land bottom-right."""
        H = _calibration_style_H()
        mgr = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [800, 800, 1000, 1000])

        # Content should be in bottom-right quadrant
        top_left = canvas[:360, :640]
        bottom_right = canvas[360:, 640:]
        assert bottom_right.sum() > 0, "Content should be in bottom-right"
        assert top_left.sum() == 0, "Top-left should be empty"

    def test_placement_ymin_xmin_lands_at_correct_position(self):
        """A small overlay at [100, 800, 200, 900] should land at
        top-y, right-x — i.e. top-right of the canvas."""
        H = _calibration_style_H()
        mgr = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 800, 200, 1000])

        # y=0-200 → top, x=800-1000 → right
        top_right = canvas[:200, 900:]
        bottom_left = canvas[500:, :400]
        assert top_right.sum() > 0, "Content should be in top-right"
        assert bottom_left.sum() == 0, "Bottom-left should be empty"
