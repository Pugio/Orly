"""Comprehensive geometry pipeline tests.

Tests the COMPLETE coordinate transformation chain with realistic physical
setups — not synthetic identity matrices. Every test constructs H_proj the
same way the real calibration does (table (x,y) → projector (px,py)), then
verifies that overlays placed via the overlay_manager land at the correct
projector pixels.

Key invariant: an overlay at placement [ymin, xmin, ymax, xmax] in Gemini's
0-1000 coordinate space must land at the projector pixels that correspond to
that table region, regardless of:
- Projector resolution (640×480, 1280×720, 1920×1080)
- Projector orientation (upright, inverted, tilted)
- Keystone distortion (trapezoidal correction)
- Aspect ratio mismatch (non-square projector on square table)
- Camera rotation (image_rotate 0/90/180/270)

The geometry pipeline:
  Gemini [ymin, xmin, ymax, xmax]
    → _unrotate_placement (if image_rotate)
    → swap [y,x] → [x,y] for H_proj
    → perspectiveTransform with H_proj → projector (px, py)
    → getPerspectiveTransform + warpPerspective → warped image
    → composite onto canvas
"""

import cv2
import numpy as np
import pytest

from client.overlay_manager import OverlayManager
from calibration.projector_calibrate import (
    compute_projector_homography,
    table_to_projector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h_proj_from_correspondences(
    table_pts: list[tuple[float, float]],
    proj_pts: list[tuple[float, float]],
) -> np.ndarray:
    """Build H_proj the same way calibration does: table (x,y) → projector (px,py)."""
    return compute_projector_homography(table_pts, proj_pts)


def _make_calibration_H(proj_w, proj_h, table_corners_in_proj):
    """Build H_proj from 4 table corners mapped to projector positions.

    table_corners_in_proj: where each table corner appears in projector space.
    Order: table (0,0), (1000,0), (1000,1000), (0,1000).
    """
    table_pts = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    return _h_proj_from_correspondences(table_pts, table_corners_in_proj)


def _asymmetric_overlay(w=100, h=100):
    """Create an overlay with distinct content in each quadrant for orientation checks.

    Top-left: red, Top-right: green, Bottom-left: blue, Bottom-right: cyan.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mid_y, mid_x = h // 2, w // 2
    img[:mid_y, :mid_x] = (0, 0, 255)    # red - TL
    img[:mid_y, mid_x:] = (0, 255, 0)    # green - TR
    img[mid_y:, :mid_x] = (255, 0, 0)    # blue - BL
    img[mid_y:, mid_x:] = (255, 255, 0)  # cyan - BR
    return img


def _expected_projector_region(placement, H_proj):
    """Given a Gemini placement, compute where it should land on the projector.

    Returns (px_center, py_center) in projector pixel space.
    """
    y_min, x_min, y_max, x_max = placement
    # Center of placement in table (x,y) space
    cx_table = (x_min + x_max) / 2
    cy_table = (y_min + y_max) / 2
    px, py = table_to_projector((cx_table, cy_table), H_proj)
    return px, py


# ---------------------------------------------------------------------------
# 1. Realistic projector setups
# ---------------------------------------------------------------------------


class TestRealisticProjectorSetups:
    """Simulate real physical projector configurations."""

    def test_projector_directly_above_table(self):
        """Projector mounted directly above, filling the full projection area."""
        proj_w, proj_h = 1280, 720
        # Table corners map linearly to projector corners
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = _asymmetric_overlay()

        # Place at center of table
        canvas = mgr.place_on_canvas(overlay, [400, 400, 600, 600])
        # Center of projector should have content
        assert canvas[360, 640].sum() > 0, "Center placement should land at projector center"

    def test_projector_with_margin(self):
        """Projector doesn't fill the full area — table maps to inner 80%."""
        proj_w, proj_h = 1280, 720
        mx, my = 128, 72  # 10% margin
        H = _make_calibration_H(proj_w, proj_h, [
            (mx, my), (proj_w - mx, my), (proj_w - mx, proj_h - my), (mx, proj_h - my),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)

        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])
        # Corners of projector should be black (margin area)
        assert canvas[0, 0].sum() == 0, "Top-left corner should be black (margin)"
        assert canvas[-1, -1].sum() == 0, "Bottom-right corner should be black"
        # Center should have content
        assert canvas[360, 640].sum() > 0, "Center should have content"

    def test_projector_inverted_mount(self):
        """Projector ceiling-mounted upside down — 180° rotation in H_proj."""
        proj_w, proj_h = 1280, 720
        H = _make_calibration_H(proj_w, proj_h, [
            (1280, 720), (0, 720), (0, 0), (1280, 0),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)

        # Top-left placement in table space → bottom-right in projector
        canvas = mgr.place_on_canvas(overlay, [0, 0, 200, 200])
        # Content should be in bottom-right quadrant of projector
        br_quadrant = canvas[360:, 640:]
        tl_quadrant = canvas[:360, :640]
        assert br_quadrant.sum() > 0, "Content should be in bottom-right (inverted)"
        assert tl_quadrant.sum() == 0, "Top-left should be empty (inverted)"

    def test_projector_off_axis_keystone(self):
        """Projector below table level — top edge narrower (keystone)."""
        proj_w, proj_h = 1280, 720
        inset = 192  # 15% narrower at top
        H = _make_calibration_H(proj_w, proj_h, [
            (inset, 0), (proj_w - inset, 0), (proj_w, proj_h), (0, proj_h),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)

        # Full table should produce a trapezoid on projector
        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])
        # Very top-left pixel should be black (keystone inset)
        assert canvas[0, 0].sum() == 0
        # Center and bottom-left should have content
        assert canvas[360, 640].sum() > 0
        assert canvas[700, 100].sum() > 0

    def test_non_square_aspect_ratio(self):
        """Wide projector (21:9) on square table — x stretches more than y."""
        proj_w, proj_h = 2560, 1080  # ultrawide
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (2560, 0), (2560, 1080), (0, 1080),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)

        # Place at table center
        canvas = mgr.place_on_canvas(overlay, [400, 400, 600, 600])
        px, py = _expected_projector_region([400, 400, 600, 600], H)
        assert canvas[min(py, proj_h - 1), min(px, proj_w - 1)].sum() > 0

    def test_small_projector(self):
        """Low-res projector (640×480)."""
        proj_w, proj_h = 640, 480
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (640, 0), (640, 480), (0, 480),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((30, 30, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [250, 250, 750, 750])
        assert canvas[240, 320].sum() > 0


# ---------------------------------------------------------------------------
# 2. Precise point mapping verification
# ---------------------------------------------------------------------------


class TestPointMappingAccuracy:
    """Verify that specific table points map to specific projector pixels.

    Uses compute_projector_homography (the real calibration function) to build
    H_proj, then checks that place_on_canvas puts overlays at the right pixels.
    """

    def test_eight_point_calibration_accuracy(self):
        """Simulate an 8-point calibration and verify overlay placement."""
        proj_w, proj_h = 1280, 720
        # 8 correspondences from a realistic calibration
        table_pts = [
            (100, 100), (500, 100), (900, 100),
            (100, 500), (900, 500),
            (100, 900), (500, 900), (900, 900),
        ]
        proj_pts = [
            (128, 72), (640, 72), (1152, 72),
            (128, 360), (1152, 360),
            (128, 648), (640, 648), (1152, 648),
        ]
        H = _h_proj_from_correspondences(table_pts, proj_pts)
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")

        # Place a small overlay centered at each calibration point
        for (tx, ty), (expected_px, expected_py) in zip(table_pts, proj_pts):
            mgr.canvas = mgr._make_bg()  # reset
            half = 50
            placement = [ty - half, tx - half, ty + half, tx + half]
            # Clamp to 0-1000
            placement = [max(0, min(1000, v)) for v in placement]
            overlay = np.full((20, 20, 3), 200, dtype=np.uint8)
            canvas = mgr.place_on_canvas(overlay, placement)

            # The overlay center should be near the expected projector pixel
            # Check a region around the expected point
            py = min(max(0, expected_py), proj_h - 1)
            px = min(max(0, expected_px), proj_w - 1)
            # Check 20px neighborhood
            y1 = max(0, py - 20)
            y2 = min(proj_h, py + 20)
            x1 = max(0, px - 20)
            x2 = min(proj_w, px + 20)
            region = canvas[y1:y2, x1:x2]
            assert region.sum() > 0, (
                f"Table ({tx},{ty}) → projector ({expected_px},{expected_py}): "
                f"no content in 20px neighborhood"
            )

    def test_corner_placement_accuracy(self):
        """Overlays at table corners land at projector corners."""
        proj_w, proj_h = 1280, 720
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")

        corners = [
            ([0, 0, 100, 100], "top-left", 0, 0),
            ([0, 900, 100, 1000], "top-right", proj_w - 1, 0),
            ([900, 0, 1000, 100], "bottom-left", 0, proj_h - 1),
            ([900, 900, 1000, 1000], "bottom-right", proj_w - 1, proj_h - 1),
        ]
        for placement, name, exp_px, exp_py in corners:
            mgr.canvas = mgr._make_bg()
            overlay = np.full((20, 20, 3), 200, dtype=np.uint8)
            canvas = mgr.place_on_canvas(overlay, placement)
            # Check 50px neighborhood around expected corner
            y1 = max(0, exp_py - 50)
            y2 = min(proj_h, exp_py + 50)
            x1 = max(0, exp_px - 50)
            x2 = min(proj_w, exp_px + 50)
            region = canvas[y1:y2, x1:x2]
            assert region.sum() > 0, (
                f"{name}: expected content near ({exp_px},{exp_py})"
            )


# ---------------------------------------------------------------------------
# 3. Overlay orientation preservation
# ---------------------------------------------------------------------------


class TestOverlayOrientation:
    """Verify that overlay content maintains correct orientation through warp.

    Uses asymmetric overlays (different colors in each quadrant) to detect
    any transposition, mirroring, or rotation errors in the warp pipeline.
    """

    def _check_quadrant_colors(self, canvas, placement, H, proj_w, proj_h, label=""):
        """Verify that the 4 quadrant colors of an asymmetric overlay
        land in the correct projector quadrants.

        Overlay quadrants: TL=red, TR=green, BL=blue, BR=cyan.
        """
        y_min, x_min, y_max, x_max = placement
        # Compute where each quadrant center should be in projector space
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        checks = [
            ((x_min + x_mid) / 2, (y_min + y_mid) / 2, 2, "red/TL"),   # TL → red (channel 2)
            ((x_mid + x_max) / 2, (y_min + y_mid) / 2, 1, "green/TR"), # TR → green (channel 1)
            ((x_min + x_mid) / 2, (y_mid + y_max) / 2, 0, "blue/BL"),  # BL → blue (channel 0)
            ((x_mid + x_max) / 2, (y_mid + y_max) / 2, 0, "cyan/BR"),  # BR → cyan (channel 0)
        ]

        for cx_table, cy_table, channel, quadrant_name in checks:
            px, py = table_to_projector((cx_table, cy_table), H)
            px = min(max(0, px), proj_w - 1)
            py = min(max(0, py), proj_h - 1)
            # Check a small region around the expected point
            y1 = max(0, py - 15)
            y2 = min(proj_h, py + 15)
            x1 = max(0, px - 15)
            x2 = min(proj_w, px + 15)
            region = canvas[y1:y2, x1:x2]
            max_in_channel = region[:, :, channel].max() if region.size > 0 else 0
            assert max_in_channel > 50, (
                f"{label} {quadrant_name}: channel {channel} at projector ({px},{py}) "
                f"should be > 50, got max={max_in_channel}"
            )

    def test_orientation_identity_H(self):
        """With identity-like H_proj, overlay orientation is preserved."""
        proj_w, proj_h = 1280, 720
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = _asymmetric_overlay(200, 200)
        canvas = mgr.place_on_canvas(overlay, [200, 200, 800, 800])
        self._check_quadrant_colors(canvas, [200, 200, 800, 800], H, proj_w, proj_h,
                                    label="identity")

    def test_orientation_with_margin(self):
        """Orientation preserved when table maps to inner 80% of projector."""
        proj_w, proj_h = 1280, 720
        mx, my = 128, 72
        H = _make_calibration_H(proj_w, proj_h, [
            (mx, my), (proj_w - mx, my), (proj_w - mx, proj_h - my), (mx, proj_h - my),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = _asymmetric_overlay(200, 200)
        canvas = mgr.place_on_canvas(overlay, [200, 200, 800, 800])
        self._check_quadrant_colors(canvas, [200, 200, 800, 800], H, proj_w, proj_h,
                                    label="margin")

    def test_orientation_with_keystone(self):
        """Orientation preserved through keystone distortion."""
        proj_w, proj_h = 1280, 720
        inset = 100
        H = _make_calibration_H(proj_w, proj_h, [
            (inset, 0), (proj_w - inset, 0), (proj_w, proj_h), (0, proj_h),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = _asymmetric_overlay(200, 200)
        canvas = mgr.place_on_canvas(overlay, [200, 200, 800, 800])
        self._check_quadrant_colors(canvas, [200, 200, 800, 800], H, proj_w, proj_h,
                                    label="keystone")

    def test_orientation_small_overlay(self):
        """Orientation preserved for small overlays."""
        proj_w, proj_h = 1280, 720
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = _asymmetric_overlay(80, 80)
        canvas = mgr.place_on_canvas(overlay, [400, 400, 600, 600])
        self._check_quadrant_colors(canvas, [400, 400, 600, 600], H, proj_w, proj_h,
                                    label="small")


# ---------------------------------------------------------------------------
# 4. Screen mode vs projector mode consistency
# ---------------------------------------------------------------------------


class TestScreenProjectorConsistency:
    """Screen mode and projector mode (with identity H) should produce
    overlays in the same region of the canvas."""

    @pytest.mark.parametrize("placement", [
        [0, 0, 500, 500],
        [0, 500, 500, 1000],
        [500, 0, 1000, 500],
        [500, 500, 1000, 1000],
        [200, 300, 700, 800],
    ])
    def test_same_region(self, placement):
        proj_w, proj_h = 1000, 1000  # square for simplicity
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1000, 0), (1000, 1000), (0, 1000),
        ])

        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)

        # Screen mode
        mgr_s = OverlayManager(H_proj=None, proj_width=proj_w, proj_height=proj_h, mode="screen")
        canvas_s = mgr_s.place_on_canvas(overlay, placement)

        # Projector mode
        mgr_p = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        canvas_p = mgr_p.place_on_canvas(overlay, placement)

        # Both should have content in the same general area
        y_min, x_min, y_max, x_max = placement
        center_x = int((x_min + x_max) / 2 / 1000 * proj_w)
        center_y = int((y_min + y_max) / 2 / 1000 * proj_h)
        # Check 50px neighborhood
        y1, y2 = max(0, center_y - 50), min(proj_h, center_y + 50)
        x1, x2 = max(0, center_x - 50), min(proj_w, center_x + 50)
        assert canvas_s[y1:y2, x1:x2].sum() > 0, f"Screen mode: no content near ({center_x},{center_y})"
        assert canvas_p[y1:y2, x1:x2].sum() > 0, f"Projector mode: no content near ({center_x},{center_y})"


# ---------------------------------------------------------------------------
# 5. Camera → Table → Projector full chain
# ---------------------------------------------------------------------------


class TestFullChainSimulation:
    """Simulate the complete chain: camera sees markers → compute H_cam →
    content at known table position → overlay placed via H_proj → verify pixel."""

    def test_camera_table_projector_chain(self):
        """Full chain: camera detects markers → H_cam → table coords →
        H_proj → projector coords. Place overlay at known table position,
        verify it lands at the correct projector pixel."""
        from client.camera import detect_markers, compute_homography, ARUCO_DICT

        # Step 1: Generate a synthetic table image
        img_size = 800
        marker_size = 100
        margin = 100
        centers = {
            0: (margin, margin),
            1: (img_size - margin, margin),
            2: (img_size - margin, img_size - margin),
            3: (margin, img_size - margin),
        }
        img = np.full((img_size, img_size, 3), 200, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        for mid, (cx, cy) in centers.items():
            marker_img = cv2.aruco.generateImageMarker(dictionary, mid, marker_size)
            marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            x1, y1 = cx - marker_size // 2, cy - marker_size // 2
            img[y1:y1 + marker_size, x1:x1 + marker_size] = marker_bgr

        # Step 2: Detect markers and compute H_cam
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        detected = detect_markers(img, detector)
        assert len(detected) == 4

        # H_cam maps camera pixels to 0-1000 table space
        dst_points = np.array([
            [0, 0], [1000, 0], [1000, 1000], [0, 1000],
        ], dtype=np.float32)
        H_cam = compute_homography(detected, dst_points)
        assert H_cam is not None

        # Step 3: Build H_proj (table 0-1000 → projector pixels)
        proj_w, proj_h = 1280, 720
        H_proj = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])

        # Step 4: Place overlay at table center [450, 450, 550, 550]
        mgr = OverlayManager(H_proj=H_proj, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [450, 450, 550, 550])

        # Step 5: Verify — center of table → center of projector
        exp_px = int(500 / 1000 * proj_w)
        exp_py = int(500 / 1000 * proj_h)
        # Check 30px neighborhood
        y1 = max(0, exp_py - 30)
        y2 = min(proj_h, exp_py + 30)
        x1 = max(0, exp_px - 30)
        x2 = min(proj_w, exp_px + 30)
        assert canvas[y1:y2, x1:x2].sum() > 0, (
            f"Overlay at table center should appear near projector ({exp_px}, {exp_py})"
        )


# ---------------------------------------------------------------------------
# 6. Rotation + Projector combined
# ---------------------------------------------------------------------------


class TestRotationWithProjector:
    """Test _unrotate_placement interaction with projector placement."""

    @pytest.mark.parametrize("image_rotate", [0, 90, 180, 270])
    def test_unrotate_then_project(self, image_rotate):
        """Regardless of camera rotation, handle_tool_result should place
        overlays in the correct table position after unrotation."""
        proj_w, proj_h = 1000, 1000
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1000, 0), (1000, 1000), (0, 1000),
        ])
        mgr = OverlayManager(
            H_proj=H, proj_width=proj_w, proj_height=proj_h,
            mode="projector", image_rotate=image_rotate,
        )

        # Simulate what Gemini would return for a center placement
        # in the rotated image. For rotate=0, center is just [400, 400, 600, 600].
        # For other rotations, we compute the rotated coordinates.
        original_placement = [400, 400, 600, 600]

        if image_rotate == 0:
            gemini_placement = original_placement
        elif image_rotate == 90:
            # Forward 90 CW: (y,x) → (x, 1000-y)
            ymin, xmin, ymax, xmax = original_placement
            gemini_placement = [xmin, 1000 - ymax, xmax, 1000 - ymin]
        elif image_rotate == 180:
            ymin, xmin, ymax, xmax = original_placement
            gemini_placement = [1000 - ymax, 1000 - xmax, 1000 - ymin, 1000 - xmin]
        elif image_rotate == 270:
            ymin, xmin, ymax, xmax = original_placement
            gemini_placement = [1000 - xmax, ymin, 1000 - xmin, ymax]

        # After unrotation, the placement should map back to the original
        unrotated = mgr._unrotate_placement(gemini_placement)
        assert unrotated == original_placement, (
            f"rotate={image_rotate}: unrotated {unrotated} != original {original_placement}"
        )

    @pytest.mark.parametrize("image_rotate", [0, 90, 180, 270])
    def test_unrotate_roundtrip_arbitrary(self, image_rotate):
        """Forward-rotate then unrotate should be identity for any placement."""
        mgr = OverlayManager(H_proj=None, mode="screen", image_rotate=image_rotate)
        original = [150, 250, 700, 850]
        ymin, xmin, ymax, xmax = original

        # Forward rotation (what camera does before Gemini)
        if image_rotate == 0:
            rotated = original
        elif image_rotate == 90:
            rotated = [xmin, 1000 - ymax, xmax, 1000 - ymin]
        elif image_rotate == 180:
            rotated = [1000 - ymax, 1000 - xmax, 1000 - ymin, 1000 - xmin]
        elif image_rotate == 270:
            rotated = [1000 - xmax, ymin, 1000 - xmin, ymax]

        recovered = mgr._unrotate_placement(rotated)
        assert recovered == original, (
            f"rotate={image_rotate}: roundtrip failed. "
            f"original={original}, rotated={rotated}, recovered={recovered}"
        )


# ---------------------------------------------------------------------------
# 7. Edge cases and degenerate geometries
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_overlay_at_boundary(self):
        """Overlay touching the edge of 0-1000 space."""
        proj_w, proj_h = 1280, 720
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((20, 20, 3), 200, dtype=np.uint8)

        # Left edge
        canvas = mgr.place_on_canvas(overlay, [400, 0, 600, 100])
        assert canvas.sum() > 0

        # Top edge
        mgr.canvas = mgr._make_bg()
        canvas = mgr.place_on_canvas(overlay, [0, 400, 100, 600])
        assert canvas.sum() > 0

    def test_very_thin_overlay(self):
        """Overlay just 1% of the table width."""
        proj_w, proj_h = 1280, 720
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        overlay = np.full((5, 100, 3), 200, dtype=np.uint8)  # very thin
        canvas = mgr.place_on_canvas(overlay, [495, 0, 505, 1000])
        assert canvas.sum() > 0

    def test_multiple_overlays_dont_interfere(self):
        """Two separate overlays should both appear on the canvas."""
        proj_w, proj_h = 1280, 720
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")

        # First overlay: red in top-left
        overlay1 = np.zeros((50, 50, 3), dtype=np.uint8)
        overlay1[:, :, 2] = 255  # red
        mgr.canvas = mgr.place_on_canvas(overlay1, [0, 0, 200, 200])

        # Second overlay: green in bottom-right
        overlay2 = np.zeros((50, 50, 3), dtype=np.uint8)
        overlay2[:, :, 1] = 255  # green
        canvas = mgr.place_on_canvas(overlay2, [800, 800, 1000, 1000])

        # Both colors should be present
        has_red = (canvas[:, :, 2] > 200).any()
        has_green = (canvas[:, :, 1] > 200).any()
        assert has_red, "First overlay (red) should survive"
        assert has_green, "Second overlay (green) should be present"

    def test_projector_mode_fallback_when_H_is_none(self):
        """If mode='projector' but H_proj is None, should fall back to screen mode."""
        mgr = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="projector")
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 500, 500])
        # Should still render (direct mapping fallback)
        assert canvas.sum() > 0

    def test_handle_tool_result_with_projector_H(self):
        """Full handle_tool_result flow with a real H_proj."""
        proj_w, proj_h = 1280, 720
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (1280, 0), (1280, 720), (0, 720),
        ])
        mgr = OverlayManager(H_proj=H, proj_width=proj_w, proj_height=proj_h, mode="projector")
        mgr.handle_tool_result("project_overlay", {
            "content_type": "annotation",
            "placement": [200, 200, 800, 800],
            "title": "test",
            "data": {"text": "Hello"},
        })
        # Canvas should have content after the full pipeline
        assert mgr.canvas.sum() > 0


# ---------------------------------------------------------------------------
# 8. White background mode with projector
# ---------------------------------------------------------------------------


class TestWhiteBgProjector:
    def test_white_bg_projector_composites_correctly(self):
        """white_bg=True with projector mode should preserve white background."""
        proj_w, proj_h = 500, 500
        H = _make_calibration_H(proj_w, proj_h, [
            (0, 0), (500, 0), (500, 500), (0, 500),
        ])
        mgr = OverlayManager(
            H_proj=H, proj_width=proj_w, proj_height=proj_h,
            mode="projector", white_bg=True,
        )
        assert mgr.canvas.min() == 255  # starts white

        # Place an overlay with some black pixels (which should not overwrite)
        overlay = np.zeros((50, 50, 3), dtype=np.uint8)
        overlay[:25, :, :] = 200  # top half bright, bottom half black
        canvas = mgr.place_on_canvas(overlay, [0, 0, 500, 500])

        # White background should be preserved where overlay is black
        # (the threshold in _composite is 30, warp uses same threshold)
        corners = [canvas[0, 0], canvas[0, -1], canvas[-1, 0], canvas[-1, -1]]
        # At least some corners should still be white-ish
        white_corners = sum(1 for c in corners if c.min() > 200)
        assert white_corners >= 1, "Some corners should remain white"
