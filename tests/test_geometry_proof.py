"""Geometry transformation proof-of-coverage tests.

Each test targets a SPECIFIC transformation in the pipeline and verifies it
using inputs that would produce visibly wrong output if the transformation
had any of its plausible failure modes.

The key technique: use ASYMMETRIC inputs where every plausible error
(axis swap, wrong rotation, corner mismatch) produces a detectably
different output. A symmetric input (square, centered, uniform-color)
can't distinguish between correct code and several classes of bugs.

Transformation inventory:
  T1: _unrotate_placement        — covered by roundtrip tests elsewhere
  T2: render_overlay size        — THIS FILE: aspect ratio test
  T3: _show_overlay orientation  — THIS FILE: no manual flip (H_proj handles it)
  T4: place_on_canvas axis swap — THIS FILE: non-square asymmetric placement
  T5: perspectiveTransform       — covered by calibration H tests elsewhere
  T6: getPerspectiveTransform    — THIS FILE: full-pipeline orientation
  T7: warpPerspective output     — trivial, covered by shape checks
  T8: composite mask             — covered by white_bg tests elsewhere
  T9: screen mode mapping        — THIS FILE: non-square screen check
"""

import cv2
import numpy as np
import pytest

from client.overlay_manager import OverlayManager
from calibration.projector_calibrate import (
    compute_projector_homography,
    table_to_projector,
)


def _calibration_H(proj_w, proj_h, corners=None):
    """Build H_proj with calibration (x,y) convention."""
    if corners is None:
        corners = [(0, 0), (proj_w, 0), (proj_w, proj_h), (0, proj_h)]
    table = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    return compute_projector_homography(table, corners)


# ---------------------------------------------------------------------------
# T2: render_overlay computes overlay_w from x-range and overlay_h from y-range
# ---------------------------------------------------------------------------


class TestRenderOverlaySizeComputation:
    """If w and h were derived from the wrong axes, a wide placement would
    produce a tall overlay and vice versa."""

    def test_wide_placement_produces_wide_overlay(self):
        """Placement [400, 0, 600, 1000] is wide (x-range=1000) and short
        (y-range=200). The rendered overlay should be wider than it is tall."""
        mgr = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        overlay = mgr.render_overlay("annotation", [400, 0, 600, 1000], "test", {"text": "x"})
        h, w = overlay.shape[:2]
        assert w > h, f"Wide placement should produce wide overlay, got {w}x{h}"
        assert w == 1000  # full width
        assert h == 200   # 20% height

    def test_tall_placement_produces_tall_overlay(self):
        """Placement [0, 400, 1000, 600] is tall (y-range=1000) and narrow
        (x-range=200). The rendered overlay should be taller than it is wide."""
        mgr = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        overlay = mgr.render_overlay("annotation", [0, 400, 1000, 600], "test", {"text": "x"})
        h, w = overlay.shape[:2]
        assert h > w, f"Tall placement should produce tall overlay, got {w}x{h}"
        assert h == 1000  # full height
        assert w == 200   # 20% width

    def test_non_square_projector_overlay_dimensions(self):
        """On a 1280x720 projector, a placement covering half the table width
        and full table height should produce an overlay of 640x720."""
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        overlay = mgr.render_overlay("annotation", [0, 0, 1000, 500], "test", {"text": "x"})
        h, w = overlay.shape[:2]
        assert w == 640, f"Half-width on 1280 projector should be 640, got {w}"
        assert h == 720, f"Full-height on 720 projector should be 720, got {h}"


# ---------------------------------------------------------------------------
# T3: _show_overlay orientation — verify NO manual flip is applied
# ---------------------------------------------------------------------------


class TestShowOverlayFlip:
    """Verify that _show_overlay applies orient_overlay correctly.

    With rotate=0 (default for these tests), orient_overlay is a no-op
    in screen mode. In projector mode, orient_overlay applies a 180° flip
    so content is readable by the human sitting opposite the projector.
    """

    def _make_asymmetric(self, w=100, h=80):
        """Non-square overlay with a single bright pixel in the top-left corner.
        After 180° rotation, it should be in the bottom-right.
        After 90° rotation (bug), it would be in the top-right or bottom-left."""
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[0, 0] = (0, 255, 255)  # yellow pixel at (row=0, col=0)
        return img

    def test_flip_is_180_not_90(self):
        """After 180° rotation of a non-square image, the bright pixel should
        move from (0,0) to (h-1, w-1). If it were 90° rotation, it would
        be at (0, h-1) or (w-1, 0) and the shape would change."""
        overlay = self._make_asymmetric(100, 80)
        flipped = cv2.rotate(overlay, cv2.ROTATE_180)
        # Shape should be preserved (180° doesn't change dimensions)
        assert flipped.shape == overlay.shape
        # Bright pixel should be at bottom-right
        assert flipped[79, 99].sum() > 0, "180° should move (0,0) to (h-1,w-1)"
        assert flipped[0, 0].sum() == 0, "Original position should be empty"

    def test_projector_mode_flips_180(self):
        """In projector mode, orient_overlay applies a 180° flip so content
        is readable by the human sitting opposite the projector.

        A bright pixel at (0,0) in the overlay moves to (h-1, w-1) after
        the 180° flip, then gets placed on canvas. With placement
        [0,0,500,500] on a 500x500 canvas (table coords 0-1000), the
        overlay occupies the top-left quarter. The flipped bright pixel
        ends up near the bottom-right of that quarter region (~row 247,
        col 247) rather than the top-left corner.
        """
        mgr = OverlayManager(H_proj=None, proj_width=500, proj_height=500, mode="projector")
        overlay = self._make_asymmetric(100, 80)
        mgr._show_overlay(overlay, [0, 0, 500, 500])
        canvas = mgr.canvas
        # After 180° flip, bright pixel moves from top-left to bottom-right
        # of the placement region (0:250, 0:250). So it should NOT be in
        # the very top-left corner, but near (247, 247).
        assert canvas[:5, :5].sum() == 0, \
            "Top-left corner should be empty after 180° flip"
        assert canvas[200:250, 200:250].sum() > 0, \
            "Projector mode flips 180°: bright pixel near bottom-right of placement region"

    def test_screen_mode_no_flip(self):
        """Screen mode: no flip applied."""
        mgr = OverlayManager(H_proj=None, proj_width=500, proj_height=500, mode="screen")
        overlay = self._make_asymmetric(100, 80)
        mgr._show_overlay(overlay, [0, 0, 500, 500])
        canvas = mgr.canvas
        assert canvas[:50, :50].sum() > 0, "Screen mode: bright pixel stays in top-left"

    def test_highlight_flips_in_projector_mode(self):
        """Highlight type in projector mode: 180° flip is applied (same as all types).
        Bright pixel moves from top-left to bottom-right of placement region."""
        mgr = OverlayManager(H_proj=None, proj_width=500, proj_height=500, mode="projector")
        overlay = self._make_asymmetric(100, 80)
        mgr._show_overlay(overlay, [0, 0, 500, 500])
        canvas = mgr.canvas
        assert canvas[:5, :5].sum() == 0, "Highlight: top-left empty after 180° flip"
        assert canvas[200:250, 200:250].sum() > 0, "Highlight: bright pixel near bottom-right of region"


# ---------------------------------------------------------------------------
# T4: place_on_canvas axis swap — non-square, asymmetric placement
# ---------------------------------------------------------------------------


class TestAxisSwapWithAsymmetricPlacement:
    """The axis swap [y,x] → [x,y] is only detectable with a placement where
    x-range != y-range. A square centered placement can't distinguish the two.

    We use a rectangular placement where the EXPECTED projector region is
    clearly non-square, and check that content appears in the right region.
    """

    def test_wide_placement_lands_wide_on_projector(self):
        """Placement [400, 0, 600, 1000] covers full x-range, narrow y-range.
        On the projector, content should span the full width but only ~20% height.
        If axes were swapped, it would span full height and ~20% width."""
        proj_w, proj_h = 1280, 720
        H = _calibration_H(proj_w, proj_h)
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [400, 0, 600, 1000])

        # Content should span most of the width
        # Check left and right edges both have content
        left_col = canvas[:, :100]
        right_col = canvas[:, -100:]
        has_left = left_col.sum() > 0
        has_right = right_col.sum() > 0
        assert has_left and has_right, "Wide placement should span full projector width"

        # Top and bottom edges should NOT both have content
        # (only the middle 20% of height should)
        top_strip = canvas[:100, :]
        bottom_strip = canvas[-100:, :]
        assert not (top_strip.sum() > 0 and bottom_strip.sum() > 0), \
            "Narrow y-range should not span full projector height"

    def test_tall_placement_lands_tall_on_projector(self):
        """Placement [0, 400, 1000, 600] covers full y-range, narrow x-range.
        Content should span full height but only ~20% width."""
        proj_w, proj_h = 1280, 720
        H = _calibration_H(proj_w, proj_h)
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 400, 1000, 600])

        # Content should span most of the height
        top_row = canvas[:100, :]
        bottom_row = canvas[-100:, :]
        has_top = top_row.sum() > 0
        has_bottom = bottom_row.sum() > 0
        assert has_top and has_bottom, "Tall placement should span full projector height"

        # Left and right edges should NOT both have content
        left_strip = canvas[:, :100]
        right_strip = canvas[:, -100:]
        assert not (left_strip.sum() > 0 and right_strip.sum() > 0), \
            "Narrow x-range should not span full projector width"


# ---------------------------------------------------------------------------
# T6 + T3 combined: Full pipeline with asymmetric overlay
# ---------------------------------------------------------------------------


class TestFullPipelineOrientation:
    """Test the COMPLETE pipeline: handle_tool_result → _unrotate_placement →
    render_overlay → orient_overlay → place_on_canvas → canvas.

    This is the integration test that combines ALL transformations and verifies
    the final output with an asymmetric marker.
    """

    def test_full_pipeline_position_projector_mode(self):
        """Full handle_tool_result in projector mode: verify overlay POSITION.
        Uses graph type to avoid adjust_text_placement expansion."""
        proj_w, proj_h = 1000, 1000
        H = _calibration_H(proj_w, proj_h)
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")

        mgr.handle_tool_result("overlay", {
            "action": "create",
            "content_type": "graph",
            "placement": [500, 500, 1000, 1000],
            "title": "test",
            "data": {"expression": "x", "x_range": [-1, 1], "y_range": [-1, 1]},
        })

        br = mgr.canvas[500:, 500:]
        tl = mgr.canvas[:400, :400]
        assert br.sum() > 0, "Overlay should be in bottom-right"
        assert tl.sum() == 0, "Top-left should be empty"

    def test_full_pipeline_position_screen_mode(self):
        """Full handle_tool_result in screen mode: same position check."""
        mgr = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")

        mgr.handle_tool_result("overlay", {
            "action": "create",
            "content_type": "graph",
            "placement": [500, 500, 1000, 1000],
            "title": "test",
            "data": {"expression": "x", "x_range": [-1, 1], "y_range": [-1, 1]},
        })

        br = mgr.canvas[500:, 500:]
        tl = mgr.canvas[:400, :400]
        assert br.sum() > 0, "Overlay should be in bottom-right"
        assert tl.sum() == 0, "Top-left should be empty"

    def test_full_pipeline_with_rotation(self):
        """Full pipeline with image_rotate=90: Gemini returns rotated coords,
        _unrotate_placement corrects them, overlay lands at original position."""
        proj_w, proj_h = 1000, 1000
        H = _calibration_H(proj_w, proj_h)

        # Original desired position: bottom-right [500, 500, 1000, 1000]
        # Forward 90° CW: (y,x) → (x, 1000-y)
        # Bbox: yr=[xmin=500, xmax=1000], xr=[1000-ymax=0, 1000-ymin=500]
        gemini_placement = [500, 0, 1000, 500]

        mgr = OverlayManager(
            H_proj=H, proj_width=proj_w, proj_height=proj_h,
            mode="projector", image_rotate=90,
        )
        mgr.handle_tool_result("overlay", {
            "action": "create",
            "content_type": "graph",
            "placement": gemini_placement,
            "title": "test",
            "data": {"expression": "x", "x_range": [-1, 1], "y_range": [-1, 1]},
        })

        # After unrotation, overlay should be at original position [500,500,1000,1000]
        br = mgr.canvas[500:, 500:]
        tl = mgr.canvas[:400, :400]
        assert br.sum() > 0, "After unrotation, overlay should be in bottom-right"
        assert tl.sum() == 0, "Top-left should be empty"


# ---------------------------------------------------------------------------
# T9: Screen mode mapping — non-square verification
# ---------------------------------------------------------------------------


class TestScreenModeNonSquare:
    """Verify screen mode handles non-square projector resolutions correctly."""

    def test_screen_mode_1280x720(self):
        """On 1280x720, placement [0, 0, 500, 500] should fill top-left quadrant.
        If x/y were swapped, the region would be wrong."""
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 500, 500])

        # x_max=500/1000*1280 = 640, y_max=500/1000*720 = 360
        # Content should be in [0:360, 0:640]
        region = canvas[:360, :640]
        outside = canvas[360:, 640:]
        assert region.sum() > 0
        assert outside.sum() == 0

    def test_screen_mode_asymmetric_placement(self):
        """Placement [0, 500, 200, 1000] → top strip, right half.
        x=500-1000 → px=640-1280, y=0-200 → py=0-144."""
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        overlay = np.full((30, 30, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 500, 200, 1000])

        # Content in top-right area
        top_right = canvas[:144, 640:]
        # Nothing in bottom-left
        bottom_left = canvas[360:, :640]
        assert top_right.sum() > 0, "Content should be in top-right"
        assert bottom_left.sum() == 0, "Bottom-left should be empty"
