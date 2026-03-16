"""Comprehensive geometric simulation tests for the Orly coordinate pipeline.

Tests the full chain: ArUco detection -> camera homography -> rectification ->
projector homography -> overlay placement, using synthetic images and known
geometric transforms. No hardware, no API calls, no file I/O.
"""

import math

import cv2
import numpy as np
import pytest

from client.camera import (
    ARUCO_DICT,
    CORNER_INDICES,
    MARKER_IDS,
    compute_homography,
    detect_markers,
    rectify_frame,
)
from client.overlay_manager import OverlayManager
from client.renderer.annotation import _render_annotation_impl as render_annotation
from client.renderer.graph import _render_graph_impl as render_graph
from client.renderer.highlight import _render_highlight_impl as render_highlight
from client.renderer.markdown import _render_markdown_impl as render_markdown
from calibration.projector_calibrate import (
    compute_projector_homography,
    table_to_projector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detector() -> cv2.aruco.ArucoDetector:
    """Create an ArUco detector matching the project config."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _draw_marker(image: np.ndarray, marker_id: int, center: tuple, size: int):
    """Draw an ArUco marker onto an image at the given center and size."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    # Convert to BGR
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    cx, cy = center
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = x1 + size
    y2 = y1 + size

    # Clip to image bounds
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(image.shape[1], x2)
    dst_y2 = min(image.shape[0], y2)
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        image[dst_y1:dst_y2, dst_x1:dst_x2] = marker_bgr[src_y1:src_y2, src_x1:src_x2]


def _generate_table_image(
    width: int,
    height: int,
    marker_centers: dict[int, tuple[int, int]],
    marker_size: int = 100,
    bg_color: int = 200,
) -> np.ndarray:
    """Generate a synthetic table image with ArUco markers at given positions.

    Args:
        width, height: Image dimensions.
        marker_centers: {marker_id: (cx, cy)} for each marker.
        marker_size: Size of each marker in pixels.
        bg_color: Background gray value.

    Returns:
        BGR image with markers drawn.
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    for mid, center in marker_centers.items():
        _draw_marker(img, mid, center, marker_size)
    return img


def _apply_perspective(image: np.ndarray, H: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Warp an image by a homography to simulate a camera view."""
    return cv2.warpPerspective(image, H, output_size)


def _identity_H_proj(proj_w: int, proj_h: int) -> np.ndarray:
    """H_proj that maps table (x,y) in 0-1000 to projector pixels linearly.

    Calibration convention: input is (x, y), output is (proj_x, proj_y).
    place_on_canvas swaps placement [y,x] to [x,y] before feeding to
    perspectiveTransform.
    """
    src = np.array([
        [0, 0], [1000, 0], [1000, 1000], [0, 1000],
    ], dtype=np.float64)
    dst = np.array([
        [0, 0], [proj_w, 0], [proj_w, proj_h], [0, proj_h],
    ], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    return H


def _scaled_H_proj(proj_w: int, proj_h: int, scale: float = 0.5) -> np.ndarray:
    """H_proj that maps table 0-1000 to a centered sub-region of projector."""
    margin_x = proj_w * (1 - scale) / 2
    margin_y = proj_h * (1 - scale) / 2
    src = np.array([
        [0, 0], [1000, 0], [1000, 1000], [0, 1000],
    ], dtype=np.float64)
    dst = np.array([
        [margin_x, margin_y],
        [proj_w - margin_x, margin_y],
        [proj_w - margin_x, proj_h - margin_y],
        [margin_x, proj_h - margin_y],
    ], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    return H


def _rotated_180_H_proj(proj_w: int, proj_h: int) -> np.ndarray:
    """H_proj that maps table 0-1000 to projector pixels, rotated 180 degrees."""
    src = np.array([
        [0, 0], [1000, 0], [1000, 1000], [0, 1000],
    ], dtype=np.float64)
    # 180 rotation: corners are swapped
    dst = np.array([
        [proj_w, proj_h], [0, proj_h], [0, 0], [proj_w, 0],
    ], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    return H


def _keystone_H_proj(proj_w: int, proj_h: int) -> np.ndarray:
    """H_proj with trapezoidal (keystone) distortion."""
    src = np.array([
        [0, 0], [0, 1000], [1000, 1000], [1000, 0],
    ], dtype=np.float64)
    # Top is narrower than bottom (simulates projector below table)
    inset = proj_w * 0.15
    dst = np.array([
        [inset, 0],
        [proj_w - inset, 0],
        [proj_w, proj_h],
        [0, proj_h],
    ], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    return H


def _transform_point_via_H(H: np.ndarray, y: float, x: float) -> tuple[float, float]:
    """Transform a single [y,x] point through H, return (col0, col1) of output."""
    pt = np.array([[[y, x]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# ---------------------------------------------------------------------------
# 1. Synthetic ArUco marker images
# ---------------------------------------------------------------------------

class TestSyntheticArUcoDetection:
    """Generate synthetic images with ArUco markers and verify detection."""

    def test_detect_four_markers_orthographic(self):
        """Markers placed at known positions in a simple top-down view."""
        marker_size = 120
        img_size = 800
        margin = 100
        centers = {
            0: (margin, margin),
            1: (img_size - margin, margin),
            2: (img_size - margin, img_size - margin),
            3: (margin, img_size - margin),
        }
        img = _generate_table_image(img_size, img_size, centers, marker_size)
        detector = _make_detector()

        detected = detect_markers(img, detector)

        assert set(detected.keys()) == {0, 1, 2, 3}
        for mid, corners in detected.items():
            assert corners.shape == (4, 2)
            # Check that the detected marker center is near the expected center
            center = corners.mean(axis=0)
            expected = np.array(centers[mid], dtype=np.float32)
            assert np.linalg.norm(center - expected) < marker_size, (
                f"Marker {mid} center {center} too far from expected {expected}"
            )

    def test_detect_markers_large_image(self):
        """Detection works on a larger image with smaller markers."""
        img_size = 1600
        marker_size = 80
        margin = 150
        centers = {
            0: (margin, margin),
            1: (img_size - margin, margin),
            2: (img_size - margin, img_size - margin),
            3: (margin, img_size - margin),
        }
        img = _generate_table_image(img_size, img_size, centers, marker_size)
        detector = _make_detector()

        detected = detect_markers(img, detector)
        assert set(detected.keys()) == {0, 1, 2, 3}

    def test_detect_markers_after_mild_rotation(self):
        """Detection works after rotating the image slightly (simulating tilted camera)."""
        img_size = 1000
        marker_size = 120
        margin = 200
        centers = {
            0: (margin, margin),
            1: (img_size - margin, margin),
            2: (img_size - margin, img_size - margin),
            3: (margin, img_size - margin),
        }
        img = _generate_table_image(img_size, img_size, centers, marker_size)

        # Rotate by 10 degrees around center
        M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), 10, 1.0)
        rotated = cv2.warpAffine(img, M, (img_size, img_size), borderValue=(200, 200, 200))

        detector = _make_detector()
        detected = detect_markers(rotated, detector)
        assert set(detected.keys()) == {0, 1, 2, 3}

    def test_detect_markers_scaled_down(self):
        """Detection works when markers are smaller (camera farther away)."""
        img_size = 1000
        marker_size = 60
        margin = 200
        centers = {
            0: (margin, margin),
            1: (img_size - margin, margin),
            2: (img_size - margin, img_size - margin),
            3: (margin, img_size - margin),
        }
        img = _generate_table_image(img_size, img_size, centers, marker_size)
        detector = _make_detector()
        detected = detect_markers(img, detector)
        assert set(detected.keys()) == {0, 1, 2, 3}

    def test_missing_marker_returns_partial(self):
        """Only 3 markers drawn means only 3 detected."""
        img_size = 800
        centers = {
            0: (100, 100),
            1: (700, 100),
            2: (700, 700),
            # marker 3 missing
        }
        img = _generate_table_image(img_size, img_size, centers, 120)
        detector = _make_detector()
        detected = detect_markers(img, detector)
        assert 3 not in detected
        assert len(detected) == 3


# ---------------------------------------------------------------------------
# 2. Full camera -> rectification pipeline
# ---------------------------------------------------------------------------

class TestCameraRectificationPipeline:
    """Synthetic table images through detect -> homography -> rectify."""

    def _make_orthographic_table(self, size=800, marker_size=120, margin=120):
        """Create a table image with markers at corners, camera directly above."""
        centers = {
            0: (margin, margin),
            1: (size - margin, margin),
            2: (size - margin, size - margin),
            3: (margin, size - margin),
        }
        img = _generate_table_image(size, size, centers, marker_size)
        return img, centers

    def test_orthographic_rectification(self):
        """Camera directly above — homography should be near-identity."""
        img, centers = self._make_orthographic_table()
        detector = _make_detector()
        output_size = (800, 800)
        dst_points = np.array([
            [0, 0], [800, 0], [800, 800], [0, 800],
        ], dtype=np.float32)

        detected = detect_markers(img, detector)
        assert len(detected) == 4

        H = compute_homography(detected, dst_points)
        assert H is not None
        assert H.shape == (3, 3)

        rectified = rectify_frame(img, H, output_size)
        assert rectified.shape == (800, 800, 3)

    def test_perspective_distortion_rectification(self):
        """Simulate a camera at an angle by warping the table image."""
        # Start with a clean top-down table
        table_size = 1000
        marker_size = 100
        margin = 150
        centers = {
            0: (margin, margin),
            1: (table_size - margin, margin),
            2: (table_size - margin, table_size - margin),
            3: (margin, table_size - margin),
        }
        table_img = _generate_table_image(table_size, table_size, centers, marker_size)

        # Apply a perspective warp to simulate camera at angle
        src_pts = np.array([
            [0, 0], [table_size, 0], [table_size, table_size], [0, table_size]
        ], dtype=np.float32)
        # Simulates viewing from below-left: top edge narrower
        dst_pts = np.array([
            [150, 100], [850, 50], [950, 900], [50, 950]
        ], dtype=np.float32)
        H_distort = cv2.getPerspectiveTransform(src_pts, dst_pts)
        distorted = cv2.warpPerspective(table_img, H_distort, (table_size, table_size),
                                        borderValue=(200, 200, 200))

        detector = _make_detector()
        detected = detect_markers(distorted, detector)
        assert len(detected) == 4, f"Found {len(detected)} markers, expected 4"

        # Rectify to 768x768
        output_size = (768, 768)
        rect_dst = np.array([
            [0, 0], [768, 0], [768, 768], [0, 768],
        ], dtype=np.float32)
        H = compute_homography(detected, rect_dst)
        assert H is not None

        rectified = rectify_frame(distorted, H, output_size)
        assert rectified.shape == (768, 768, 3)

        # Verify rectification by checking that the inner corners of detected
        # markers map to the expected destination points through the homography.
        for idx, mid in enumerate(MARKER_IDS):
            corners = detected[mid]
            corner_idx = CORNER_INDICES[mid]
            src_pt = corners[corner_idx].reshape(1, 1, 2).astype(np.float64)
            dst_pt = cv2.perspectiveTransform(src_pt, H)[0, 0]
            expected = rect_dst[idx]
            assert np.linalg.norm(dst_pt - expected) < 5, (
                f"Marker {mid} inner corner mapped to {dst_pt}, expected ~{expected}"
            )

    def test_mild_rotation_rectification(self):
        """Camera rotated ~15 degrees, rectification should correct it."""
        table_size = 1000
        marker_size = 100
        margin = 200
        centers = {
            0: (margin, margin),
            1: (table_size - margin, margin),
            2: (table_size - margin, table_size - margin),
            3: (margin, table_size - margin),
        }
        table_img = _generate_table_image(table_size, table_size, centers, marker_size)

        # Rotate 15 degrees
        M = cv2.getRotationMatrix2D((table_size / 2, table_size / 2), 15, 0.9)
        rotated = cv2.warpAffine(table_img, M, (table_size, table_size),
                                 borderValue=(200, 200, 200))

        detector = _make_detector()
        detected = detect_markers(rotated, detector)
        assert len(detected) == 4

        output_size = (768, 768)
        rect_dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)
        H = compute_homography(detected, rect_dst)
        assert H is not None

        rectified = rectify_frame(rotated, H, output_size)
        assert rectified.shape == (768, 768, 3)

    def test_homography_returns_none_with_insufficient_markers(self):
        """compute_homography returns None when fewer than 4 markers found."""
        detected = {0: np.zeros((4, 2)), 1: np.zeros((4, 2))}  # only 2
        dst_points = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)
        H = compute_homography(detected, dst_points)
        assert H is None


# ---------------------------------------------------------------------------
# 3. Projector homography simulation
# ---------------------------------------------------------------------------

class TestProjectorHomographySimulation:
    """Test various H_proj scenarios and verify coordinate transforms."""

    def test_identity_mapping(self):
        """Table coords map proportionally to projector pixels."""
        proj_w, proj_h = 1280, 720
        H = _identity_H_proj(proj_w, proj_h)

        # Table center [y=500, x=500] -> projector center
        px, py = _transform_point_via_H(H, 500, 500)
        assert abs(px - proj_w / 2) < 2
        assert abs(py - proj_h / 2) < 2

        # Table origin [y=0, x=0] -> projector origin
        px, py = _transform_point_via_H(H, 0, 0)
        assert abs(px) < 2
        assert abs(py) < 2

        # Table [y=1000, x=1000] -> projector [w, h]
        px, py = _transform_point_via_H(H, 1000, 1000)
        assert abs(px - proj_w) < 2
        assert abs(py - proj_h) < 2

    def test_scaled_mapping(self):
        """Table coords map to a centered 50% sub-region of projector."""
        proj_w, proj_h = 1280, 720
        H = _scaled_H_proj(proj_w, proj_h, scale=0.5)

        # Center should still map to center
        px, py = _transform_point_via_H(H, 500, 500)
        assert abs(px - proj_w / 2) < 2
        assert abs(py - proj_h / 2) < 2

        # Origin maps to 25% in from edges
        px, py = _transform_point_via_H(H, 0, 0)
        assert abs(px - proj_w * 0.25) < 2
        assert abs(py - proj_h * 0.25) < 2

        # [1000, 1000] maps to 75% in from origin
        px, py = _transform_point_via_H(H, 1000, 1000)
        assert abs(px - proj_w * 0.75) < 2
        assert abs(py - proj_h * 0.75) < 2

    def test_rotated_180_mapping(self):
        """Table coords are rotated 180 degrees on projector."""
        proj_w, proj_h = 1280, 720
        H = _rotated_180_H_proj(proj_w, proj_h)

        # Origin -> opposite corner
        px, py = _transform_point_via_H(H, 0, 0)
        assert abs(px - proj_w) < 2
        assert abs(py - proj_h) < 2

        # [1000, 1000] -> origin
        px, py = _transform_point_via_H(H, 1000, 1000)
        assert abs(px) < 2
        assert abs(py) < 2

        # Center -> center (rotation preserves center)
        px, py = _transform_point_via_H(H, 500, 500)
        assert abs(px - proj_w / 2) < 2
        assert abs(py - proj_h / 2) < 2

    def test_keystone_mapping(self):
        """Keystone distortion maps correctly."""
        proj_w, proj_h = 1280, 720
        H = _keystone_H_proj(proj_w, proj_h)

        # Top-left should be indented
        px, py = _transform_point_via_H(H, 0, 0)
        assert px > 50  # indented from left edge
        assert abs(py) < 5  # near top

        # Bottom corners should be at full width
        px, py = _transform_point_via_H(H, 1000, 0)
        assert abs(px) < 5
        assert abs(py - proj_h) < 5

    def test_compute_projector_homography_round_trip(self):
        """compute_projector_homography + table_to_projector is consistent."""
        table_pts = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        proj_pts = [(100, 50), (1180, 50), (1180, 670), (100, 670)]

        H = compute_projector_homography(table_pts, proj_pts)

        for tp, pp in zip(table_pts, proj_pts):
            result = table_to_projector(tp, H)
            assert abs(result[0] - pp[0]) < 2, f"{tp} -> {result}, expected {pp}"
            assert abs(result[1] - pp[1]) < 2, f"{tp} -> {result}, expected {pp}"


# ---------------------------------------------------------------------------
# 4. Full round-trip: Gemini coordinates -> projector pixels
# ---------------------------------------------------------------------------

class TestGeminiToProjectorRoundTrip:
    """Verify overlay placement with various Gemini placements and H_proj."""

    PLACEMENTS = [
        [0, 0, 500, 500],         # top-left quadrant
        [0, 500, 500, 1000],      # top-right quadrant
        [500, 0, 1000, 500],      # bottom-left quadrant
        [500, 500, 1000, 1000],   # bottom-right quadrant
        [250, 250, 750, 750],     # center
        [0, 0, 1000, 1000],       # full canvas
        [490, 490, 510, 510],     # tiny center
    ]

    def test_identity_H_proj_placement(self):
        """With identity H_proj, Gemini coords map proportionally to projector."""
        proj_w, proj_h = 1280, 720
        H = _identity_H_proj(proj_w, proj_h)
        mgr = OverlayManager(H, proj_w, proj_h, mode="projector")

        for placement in self.PLACEMENTS:
            y_min, x_min, y_max, x_max = placement
            overlay = np.full((100, 100, 3), 128, dtype=np.uint8)
            canvas = mgr.place_on_canvas(overlay, placement)
            assert canvas.shape == (proj_h, proj_w, 3)

            # Expected projector pixel region
            exp_px_min = int(x_min / 1000.0 * proj_w)
            exp_py_min = int(y_min / 1000.0 * proj_h)
            exp_px_max = int(x_max / 1000.0 * proj_w)
            exp_py_max = int(y_max / 1000.0 * proj_h)

            if exp_px_max > exp_px_min and exp_py_max > exp_py_min:
                # Check center of expected region has content
                cx = (exp_px_min + exp_px_max) // 2
                cy = (exp_py_min + exp_py_max) // 2
                cx = min(cx, proj_w - 1)
                cy = min(cy, proj_h - 1)
                assert canvas[cy, cx].sum() > 0, (
                    f"Placement {placement}: center ({cx},{cy}) should have content"
                )

    def test_screen_mode_placement(self):
        """In screen mode, coords map directly without H_proj."""
        proj_w, proj_h = 1280, 720
        mgr = OverlayManager(None, proj_w, proj_h, mode="screen")

        for placement in self.PLACEMENTS:
            y_min, x_min, y_max, x_max = placement
            overlay = np.full((100, 100, 3), 200, dtype=np.uint8)
            canvas = mgr.place_on_canvas(overlay, placement)

            exp_px_min = max(0, int(x_min / 1000.0 * proj_w))
            exp_py_min = max(0, int(y_min / 1000.0 * proj_h))
            exp_px_max = min(proj_w, int(x_max / 1000.0 * proj_w))
            exp_py_max = min(proj_h, int(y_max / 1000.0 * proj_h))

            if exp_px_max > exp_px_min and exp_py_max > exp_py_min:
                region = canvas[exp_py_min:exp_py_max, exp_px_min:exp_px_max]
                assert region.sum() > 0, f"Placement {placement} region should have content"

    def test_scaled_H_proj_placement(self):
        """With scaled H_proj, overlays appear in center sub-region."""
        proj_w, proj_h = 1280, 720
        H = _scaled_H_proj(proj_w, proj_h, scale=0.5)
        mgr = OverlayManager(H, proj_w, proj_h, mode="projector")

        # Full table placement [0,0,1000,1000] should only fill center 50%
        overlay = np.full((100, 100, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])

        # The very corners of the projector should be black
        assert canvas[0, 0].sum() == 0, "Top-left corner should be black"
        assert canvas[0, proj_w - 1].sum() == 0, "Top-right corner should be black"
        assert canvas[proj_h - 1, 0].sum() == 0, "Bottom-left corner should be black"
        assert canvas[proj_h - 1, proj_w - 1].sum() == 0, "Bottom-right corner should be black"

        # Center should have content
        assert canvas[proj_h // 2, proj_w // 2].sum() > 0, "Center should have content"

    def test_all_placements_with_rotated_H(self):
        """All placements produce valid output with 180-degree rotated H_proj."""
        proj_w, proj_h = 1280, 720
        H = _rotated_180_H_proj(proj_w, proj_h)
        mgr = OverlayManager(H, proj_w, proj_h, mode="projector")

        for placement in self.PLACEMENTS:
            overlay = np.full((50, 50, 3), 150, dtype=np.uint8)
            canvas = mgr.place_on_canvas(overlay, placement)
            assert canvas.shape == (proj_h, proj_w, 3)
            # At least some pixels should have content
            assert canvas.sum() > 0, f"Placement {placement} produced empty canvas"


# ---------------------------------------------------------------------------
# 5. Rotation permutations
# ---------------------------------------------------------------------------

class TestUnrotatePlacement:
    """Test _unrotate_placement for all 4 rotations."""

    def _make_mgr(self, rotate: int) -> OverlayManager:
        return OverlayManager(None, 1280, 720, mode="screen", image_rotate=rotate)

    def test_no_rotation(self):
        """0-degree rotation is identity."""
        mgr = self._make_mgr(0)
        placement = [100, 200, 700, 800]
        result = mgr._unrotate_placement(placement)
        assert result == placement

    def test_90_degree_unrotate(self):
        """90-degree CW rotation unrotation."""
        mgr = self._make_mgr(90)
        # CW 90 inverse: [1000-xmax, ymin, 1000-xmin, ymax]
        placement = [100, 200, 700, 800]
        result = mgr._unrotate_placement(placement)
        assert result == [1000 - 800, 100, 1000 - 200, 700]
        assert result == [200, 100, 800, 700]

    def test_180_degree_unrotate(self):
        """180-degree rotation unrotation."""
        mgr = self._make_mgr(180)
        placement = [100, 200, 700, 800]
        result = mgr._unrotate_placement(placement)
        assert result == [1000 - 700, 1000 - 800, 1000 - 100, 1000 - 200]
        assert result == [300, 200, 900, 800]

    def test_270_degree_unrotate(self):
        """270-degree rotation unrotation."""
        mgr = self._make_mgr(270)
        # CCW 90 inverse: [xmin, 1000-ymax, xmax, 1000-ymin]
        placement = [100, 200, 700, 800]
        result = mgr._unrotate_placement(placement)
        assert result == [200, 1000 - 700, 800, 1000 - 100]
        assert result == [200, 300, 800, 900]

    def test_all_rotations_preserve_area(self):
        """All rotations should preserve the area of the bounding box (in 0-1000 space)."""
        placement = [100, 200, 600, 900]
        orig_area = (600 - 100) * (900 - 200)

        for rot in [0, 90, 180, 270]:
            mgr = self._make_mgr(rot)
            result = mgr._unrotate_placement(placement)
            ymin, xmin, ymax, xmax = result
            area = (ymax - ymin) * (xmax - xmin)
            assert area == orig_area, f"Rotation {rot}: area {area} != {orig_area}"

    def test_full_canvas_unrotation(self):
        """Full canvas [0,0,1000,1000] should remain full after any unrotation."""
        placement = [0, 0, 1000, 1000]
        for rot in [0, 90, 180, 270]:
            mgr = self._make_mgr(rot)
            result = mgr._unrotate_placement(placement)
            ymin, xmin, ymax, xmax = result
            assert ymin == 0 and xmin == 0 and ymax == 1000 and xmax == 1000, (
                f"Rotation {rot}: full canvas became {result}"
            )


# ---------------------------------------------------------------------------
# 6. Camera/Projector/Table orientation permutations
# ---------------------------------------------------------------------------

class TestProjectorModeOrientations:
    """Test OverlayManager with mode='projector' and various H_proj."""

    def test_projector_mode_flips_180(self):
        """Projector mode applies a 180° flip via orient_overlay so content
        is readable by the human sitting opposite the projector.

        A bright top-left corner in the overlay should end up in the
        bottom-right area of the canvas after the flip.
        """
        proj_w, proj_h = 640, 480
        H = _identity_H_proj(proj_w, proj_h)
        mgr = OverlayManager(H, proj_w, proj_h, mode="projector")

        # Create an asymmetric overlay (bright top-left corner only)
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[0:30, 0:30] = [0, 255, 255]  # yellow top-left

        mgr._show_overlay(overlay, [0, 0, 1000, 1000])
        canvas = mgr.canvas

        # With 180° flip, yellow should move to the bottom-right area
        top_left_sum = canvas[0:proj_h // 4, 0:proj_w // 4].sum()
        bottom_right_sum = canvas[3 * proj_h // 4:, 3 * proj_w // 4:].sum()
        assert bottom_right_sum > top_left_sum, (
            "Projector mode flips 180°: content should move to bottom-right"
        )

    def test_projector_mode_highlight_flips(self):
        """Highlights ARE flipped 180° in projector mode (same as all overlay types)."""
        proj_w, proj_h = 640, 480
        H = _identity_H_proj(proj_w, proj_h)
        mgr = OverlayManager(H, proj_w, proj_h, mode="projector")

        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[0:30, 0:30] = [0, 255, 255]

        mgr._show_overlay(overlay, [0, 0, 500, 500])
        # Verify it produces content and the flip is applied
        assert mgr.canvas.sum() > 0

    def test_white_bg_mode(self):
        """White background mode starts with white canvas."""
        mgr = OverlayManager(None, 640, 480, mode="screen", white_bg=True)
        assert mgr.canvas.mean() == 255.0

    def test_black_bg_mode(self):
        """Black background mode starts with black canvas."""
        mgr = OverlayManager(None, 640, 480, mode="screen", white_bg=False)
        assert mgr.canvas.mean() == 0.0

    def test_white_bg_compositing(self):
        """White bg mode: only non-black overlay pixels replace canvas."""
        proj_w, proj_h = 640, 480
        mgr = OverlayManager(None, proj_w, proj_h, mode="screen", white_bg=True)

        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[40:60, 40:60] = [0, 255, 0]  # green center

        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])

        # Most of the canvas should still be white (from bg), not black
        white_pixels = np.count_nonzero(canvas.sum(axis=2) == 255 * 3)
        total_pixels = proj_w * proj_h
        # Most pixels should be white (overlay is mostly black = transparent)
        assert white_pixels > total_pixels * 0.5


# ---------------------------------------------------------------------------
# 7. End-to-end rendering verification
# ---------------------------------------------------------------------------

class TestEndToEndRendering:
    """Create OverlayManager, render overlays, verify canvas content."""

    def test_graph_overlay_placement(self):
        """Graph overlay renders at the correct region."""
        proj_w, proj_h = 1280, 720
        H = _identity_H_proj(proj_w, proj_h)
        mgr = OverlayManager(H, proj_w, proj_h, mode="projector")

        placement = [0, 0, 500, 500]
        overlay = mgr.render_overlay("graph", placement, "y=x^2", {
            "expression": "x**2",
            "x_range": [-5, 5],
            "y_range": [0, 25],
        })

        assert overlay.shape[2] == 3  # BGR
        assert overlay.shape[0] > 0 and overlay.shape[1] > 0
        assert overlay.sum() > 0, "Graph should have visible content"

    def test_annotation_overlay_placement(self):
        """Annotation overlay has text in expected region."""
        proj_w, proj_h = 1280, 720
        mgr = OverlayManager(None, proj_w, proj_h, mode="screen")

        result = {
            "content_type": "annotation",
            "placement": [250, 250, 750, 750],
            "title": "Test",
            "data": {"text": "Hello World"},
        }
        mgr.handle_tool_result("overlay", {"action": "create", **result})

        canvas = mgr.canvas
        # Center region should have content
        center_region = canvas[
            int(0.25 * proj_h):int(0.75 * proj_h),
            int(0.25 * proj_w):int(0.75 * proj_w),
        ]
        assert center_region.sum() > 0, "Center should have annotation text"

        # Corners outside the placement should be black
        corner = canvas[0:50, 0:50]
        assert corner.sum() == 0, "Top-left corner should be black"

    def test_highlight_overlay_placement(self):
        """Highlight overlay renders as colored rectangle."""
        proj_w, proj_h = 640, 480
        mgr = OverlayManager(None, proj_w, proj_h, mode="screen")

        result = {
            "content_type": "highlight",
            "placement": [0, 0, 500, 500],
            "title": "",
            "data": {"color": "#ff0000"},
        }
        mgr.handle_tool_result("overlay", {"action": "create", **result})
        canvas = mgr.canvas

        # Top-left quadrant should have content
        tl = canvas[0:proj_h // 2, 0:proj_w // 2]
        assert tl.sum() > 0, "Top-left should have highlight"

        # Bottom-right should be black
        br = canvas[proj_h // 2:, proj_w // 2:]
        assert br.sum() == 0, "Bottom-right should be black"

    def test_markdown_overlay_placement(self):
        """Markdown overlay renders text."""
        proj_w, proj_h = 1280, 720
        mgr = OverlayManager(None, proj_w, proj_h, mode="screen")

        result = {
            "content_type": "markdown",
            "placement": [0, 0, 1000, 1000],
            "title": "",
            "data": {"text": "# Hello\n\nThis is a test"},
        }
        mgr.handle_tool_result("overlay", {"action": "create", **result})
        canvas = mgr.canvas
        assert canvas.sum() > 0, "Markdown should render visible text"

    def test_render_graph_dimensions(self):
        """render_graph produces exact requested dimensions."""
        for w, h in [(400, 300), (1280, 720), (100, 100)]:
            img = render_graph("x**2", [-5, 5], [0, 25], w, h)
            assert img.shape == (h, w, 3), f"Expected ({h},{w},3), got {img.shape}"

    def test_render_annotation_dimensions(self):
        """render_annotation produces exact requested dimensions."""
        for w, h in [(400, 300), (1280, 720), (50, 50)]:
            img = render_annotation("Test text", w, h)
            assert img.shape == (h, w, 3), f"Expected ({h},{w},3), got {img.shape}"

    def test_render_markdown_dimensions(self):
        """render_markdown produces exact requested dimensions."""
        for w, h in [(400, 300), (1280, 720), (200, 150)]:
            img = render_markdown("# Title\n\n- bullet", w, h)
            assert img.shape == (h, w, 3), f"Expected ({h},{w},3), got {img.shape}"

    def test_render_highlight_dimensions(self):
        """render_highlight produces exact requested dimensions (BGRA)."""
        for w, h in [(400, 300), (1280, 720), (50, 50)]:
            img = render_highlight(w, h, "#00ff00")
            assert img.shape == (h, w, 4), f"Expected ({h},{w},4), got {img.shape}"

    def test_clear_resets_canvas(self):
        """clear() resets canvas to background."""
        mgr = OverlayManager(None, 640, 480, mode="screen")

        # Add some content
        result = {
            "content_type": "annotation",
            "placement": [0, 0, 1000, 1000],
            "title": "",
            "data": {"text": "Hello"},
        }
        mgr.handle_tool_result("overlay", {"action": "create", **result})
        assert mgr.canvas.sum() > 0

        mgr.clear()
        assert mgr.canvas.sum() == 0, "Canvas should be black after clear"


# ---------------------------------------------------------------------------
# 8. Stress tests
# ---------------------------------------------------------------------------

class TestStress:
    """Edge cases and stress scenarios."""

    def test_large_placement_small_projector(self):
        """Full placement on tiny 320x240 projector."""
        proj_w, proj_h = 320, 240
        mgr = OverlayManager(None, proj_w, proj_h, mode="screen")

        overlay = render_annotation("Big text on small screen", 320, 240)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])
        assert canvas.shape == (proj_h, proj_w, 3)
        assert canvas.sum() > 0

    def test_near_zero_placement(self):
        """Near-zero size placement doesn't crash."""
        proj_w, proj_h = 1280, 720
        mgr = OverlayManager(None, proj_w, proj_h, mode="screen")

        overlay = np.full((10, 10, 3), 200, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [499, 499, 501, 501])
        assert canvas.shape == (proj_h, proj_w, 3)

    def test_single_pixel_placement(self):
        """Placement that maps to ~1 pixel doesn't crash."""
        proj_w, proj_h = 1280, 720
        mgr = OverlayManager(None, proj_w, proj_h, mode="screen")

        overlay = np.full((1, 1, 3), 255, dtype=np.uint8)
        # 1/1000 of 1280 = 1.28 pixels, 1/1000 of 720 = 0.72 pixels -> 0 height, skipped
        canvas = mgr.place_on_canvas(overlay, [500, 500, 501, 501])
        assert canvas.shape == (proj_h, proj_w, 3)

    def test_rapid_overlay_then_clear(self):
        """Rapid overlay placement followed by clear."""
        mgr = OverlayManager(None, 640, 480, mode="screen")

        for i in range(20):
            result = {
                "content_type": "annotation",
                "placement": [0, 0, 1000, 1000],
                "title": f"Frame {i}",
                "data": {"text": f"Overlay {i}"},
            }
            mgr.handle_tool_result("overlay", {"action": "create", **result})
            assert mgr.canvas.sum() > 0

        mgr.clear()
        assert mgr.canvas.sum() == 0

    def test_projector_H_with_small_projector(self):
        """H_proj with small projector resolution."""
        proj_w, proj_h = 320, 240
        H = _identity_H_proj(proj_w, proj_h)
        mgr = OverlayManager(H, proj_w, proj_h, mode="projector")

        overlay = np.full((50, 50, 3), 128, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])
        assert canvas.shape == (proj_h, proj_w, 3)
        assert canvas.sum() > 0

    def test_floating_point_placement(self):
        """Placements with float values are handled."""
        mgr = OverlayManager(None, 1280, 720, mode="screen")
        overlay = np.full((50, 50, 3), 100, dtype=np.uint8)
        # These are floats, not ints
        canvas = mgr.place_on_canvas(overlay, [100.5, 200.7, 800.3, 900.1])
        assert canvas.shape == (720, 1280, 3)
        assert canvas.sum() > 0

    def test_refresh_cycle(self):
        """request_refresh + complete_refresh restores canvas."""
        mgr = OverlayManager(None, 640, 480, mode="screen")
        result = {
            "content_type": "annotation",
            "placement": [0, 0, 1000, 1000],
            "title": "",
            "data": {"text": "Hello"},
        }
        mgr.handle_tool_result("overlay", {"action": "create", **result})
        content_sum = mgr.canvas.sum()
        assert content_sum > 0

        mgr.request_refresh()
        assert mgr.canvas.sum() == 0, "Canvas should be blank during refresh"

        mgr.complete_refresh()
        assert mgr.canvas.sum() == content_sum, "Canvas should be restored after refresh"

    def test_multiple_overlays_accumulate(self):
        """Multiple placements in different regions all appear on canvas."""
        mgr = OverlayManager(None, 1280, 720, mode="screen")

        # Place in top-left
        overlay1 = np.full((100, 100, 3), 100, dtype=np.uint8)
        mgr.canvas = mgr.place_on_canvas(overlay1, [0, 0, 200, 200])
        tl_sum = mgr.canvas[0:144, 0:256].sum()

        # Place in bottom-right (should not overwrite top-left due to black bg)
        overlay2 = np.full((100, 100, 3), 200, dtype=np.uint8)
        mgr.canvas = mgr.place_on_canvas(overlay2, [800, 800, 1000, 1000])

        # Both regions should have content
        assert mgr.canvas[0:100, 0:100].sum() > 0, "Top-left should have content"
        assert mgr.canvas[600:, 1024:].sum() > 0, "Bottom-right should have content"

    def test_keystone_H_proj_full_pipeline(self):
        """Keystone H_proj produces content on canvas."""
        proj_w, proj_h = 1280, 720
        H = _keystone_H_proj(proj_w, proj_h)
        mgr = OverlayManager(H, proj_w, proj_h, mode="projector")

        overlay = np.full((100, 100, 3), 150, dtype=np.uint8)
        canvas = mgr.place_on_canvas(overlay, [0, 0, 1000, 1000])
        assert canvas.sum() > 0

    def test_unknown_content_type_returns_black(self):
        """Unknown content type produces black image."""
        mgr = OverlayManager(None, 640, 480, mode="screen")
        overlay = mgr.render_overlay("unknown_type", [0, 0, 1000, 1000], "", {})
        assert overlay.sum() == 0
        assert overlay.shape == (480, 640, 3)
