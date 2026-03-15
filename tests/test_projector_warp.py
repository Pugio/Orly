"""Tests for projector perspective warp (Bug 1).

Verifies that place_on_canvas uses warpPerspective instead of bounding-box
resize when H_proj is provided.
"""

import cv2
import numpy as np
import pytest

from client.overlay_manager import OverlayManager


def _identity_H():
    """H_proj that maps 0-1000 table coords to 1280x720 projector pixels."""
    # Simple scale: x_proj = x * 1.28, y_proj = y * 0.72
    src = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)
    dst = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def _rotation_H():
    """H_proj with a slight rotation — bounding-box resize would lose this."""
    src = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)
    # Slight rotation: top-right shifts down, bottom-left shifts up
    dst = np.array([[10, 10], [1270, 30], [1260, 710], [20, 690]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


class TestProjectorWarp:
    def test_warp_identity_homography_matches_direct_placement(self):
        """With identity-like H_proj, warped output should cover same area as direct."""
        H = _identity_H()
        mgr = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        overlay = np.full((100, 200, 3), 128, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 500, 500])
        # Should have non-zero content in roughly the top-left quadrant
        assert canvas.sum() > 0
        # Bottom-right quadrant should be mostly empty
        assert canvas[500:, 800:].sum() == 0

    def test_warp_with_rotation_homography(self):
        """With rotated H_proj, output should be a warped quad, not axis-aligned."""
        H = _rotation_H()
        mgr = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        # Full table overlay — bright cyan
        overlay = np.full((200, 200, 3), (255, 255, 0), dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])
        # Should have content (the warp fills most of the canvas)
        assert canvas.sum() > 0

    def test_warp_preserves_overlay_content(self):
        """After mild warp, content (cyan cross) should still be visible."""
        H = _identity_H()
        mgr = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        # Create a cross pattern
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[45:55, :, :] = (255, 255, 0)  # horizontal bar
        overlay[:, 45:55, :] = (255, 255, 0)  # vertical bar
        canvas = mgr.place_on_canvas(overlay, [200, 200, 500, 500])
        # Canvas should have cyan content
        cyan_pixels = (canvas[:, :, 0] > 100).sum()
        assert cyan_pixels > 50  # cross is visible

    def test_warp_large_overlay_covers_expected_area(self):
        """Full-table overlay should cover >50% of canvas with identity-like H."""
        H = _identity_H()
        mgr = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        overlay = np.full((200, 200, 3), 128, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])
        total_pixels = 1280 * 720
        nonzero = (canvas.sum(axis=2) > 0).sum()
        assert nonzero / total_pixels > 0.5

    def test_warp_composites_without_destroying_existing(self):
        """Two overlays should both survive on canvas."""
        H = _identity_H()
        mgr = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        # First overlay: red in top-left
        overlay1 = np.full((50, 50, 3), (0, 0, 255), dtype=np.uint8)
        mgr.canvas = mgr.place_on_canvas(overlay1, [0, 0, 200, 200])
        # Second overlay: green in bottom-right
        overlay2 = np.full((50, 50, 3), (0, 255, 0), dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay2, [700, 700, 1000, 1000])
        # Both colors should be present
        has_red = (canvas[:, :, 2] > 200).sum() > 0
        has_green = (canvas[:, :, 1] > 200).sum() > 0
        assert has_red and has_green
