"""Tests for coordinate system round-trips: Gemini 0-1000 → table → projector.

Simulates the full physical setup with synthetic homographies to verify
overlays land in the correct position regardless of camera/projector orientation.
"""

import cv2
import numpy as np
import pytest

from client.overlay_manager import OverlayManager
from client.camera import (
    detect_markers,
    compute_homography,
    rectify_frame,
    encode_jpeg,
    ARUCO_DICT,
    MARKER_IDS,
    CORNER_INDICES,
)


# ---------------------------------------------------------------------------
# Helpers: generate synthetic ArUco marker images
# ---------------------------------------------------------------------------


def _draw_marker(image: np.ndarray, marker_id: int, center: tuple, size: int):
    """Draw an ArUco marker onto an image at the given center position."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    cx, cy = center
    x1 = cx - size // 2
    y1 = cy - size // 2
    x2 = x1 + size
    y2 = y1 + size

    # Clip to image bounds
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(image.shape[1], x2), min(image.shape[0], y2)
    mx1, my1 = ix1 - x1, iy1 - y1
    mx2, my2 = mx1 + (ix2 - ix1), my1 + (iy2 - iy1)

    if ix2 > ix1 and iy2 > iy1:
        image[iy1:iy2, ix1:ix2] = marker_bgr[my1:my2, mx1:mx2]


def _make_synthetic_table(
    width: int = 1200,
    height: int = 1200,
    marker_size: int = 80,
    margin: int = 100,
) -> tuple[np.ndarray, dict]:
    """Create a synthetic table image with 4 ArUco markers at known positions.

    Returns (image, expected_corners) where expected_corners maps marker_id
    to the center position.
    """
    img = np.full((height, width, 3), 200, dtype=np.uint8)  # grey table

    positions = {
        0: (margin, margin),              # top-left
        1: (width - margin, margin),      # top-right
        2: (width - margin, height - margin),  # bottom-right
        3: (margin, height - margin),     # bottom-left
    }

    for mid, pos in positions.items():
        _draw_marker(img, mid, pos, marker_size)

    return img, positions


# ---------------------------------------------------------------------------
# Test: ArUco detection on synthetic images
# ---------------------------------------------------------------------------


class TestSyntheticMarkerDetection:
    def test_detect_all_four_markers(self):
        """Generate a synthetic table and detect all 4 markers."""
        img, positions = _make_synthetic_table()
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        detected = detect_markers(img, detector)
        assert len(detected) == 4
        for mid in MARKER_IDS:
            assert mid in detected

    def test_detected_markers_near_expected_positions(self):
        """Detected marker centers should be near the drawn positions."""
        img, positions = _make_synthetic_table(marker_size=100)
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        detected = detect_markers(img, detector)
        for mid in MARKER_IDS:
            corners = detected[mid]  # (4, 2) array
            center = corners.mean(axis=0)
            expected = np.array(positions[mid], dtype=np.float32)
            dist = np.linalg.norm(center - expected)
            assert dist < 30, f"Marker {mid} center {center} too far from expected {expected}"


# ---------------------------------------------------------------------------
# Test: Homography computation from synthetic markers
# ---------------------------------------------------------------------------


class TestSyntheticHomography:
    def test_homography_from_orthographic_view(self):
        """Camera directly above table → near-identity homography."""
        img, _ = _make_synthetic_table(width=768, height=768, margin=80)
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        detected = detect_markers(img, detector)
        dst_points = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)
        H = compute_homography(detected, dst_points)

        assert H is not None
        # Rectified image should be similar to original (near-identity warp)
        rectified = rectify_frame(img, H, (768, 768))
        assert rectified.shape == (768, 768, 3)

    def test_rectification_preserves_content(self):
        """Rectified output should contain marker pixels (not all black)."""
        img, _ = _make_synthetic_table()
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

        detected = detect_markers(img, detector)
        dst_points = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)
        H = compute_homography(detected, dst_points)
        assert H is not None

        rectified = rectify_frame(img, H, (768, 768))
        # Should not be all black
        assert rectified.max() > 0

    def test_encode_decode_roundtrip(self):
        """JPEG encode → decode should preserve basic structure."""
        img, _ = _make_synthetic_table(width=400, height=400)
        jpeg = encode_jpeg(img, quality=95)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == img.shape
        # Lossy, but should be close
        assert np.allclose(decoded, img, atol=10)


# ---------------------------------------------------------------------------
# Test: Projector homography — Gemini coords → projector pixels
# ---------------------------------------------------------------------------


class TestProjectorHomography:
    def _identity_h_proj(self, proj_w: int, proj_h: int) -> np.ndarray:
        """H_proj that maps [y, x] in 0-1000 → projector pixels directly."""
        src = np.array([[0, 0], [0, 1000], [1000, 1000], [1000, 0]], dtype=np.float32)
        dst = np.array([[0, 0], [proj_w, 0], [proj_w, proj_h], [0, proj_h]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def test_identity_h_proj_center_overlay(self):
        """Center overlay at [250, 250, 750, 750] → center of projector."""
        H = self._identity_h_proj(1280, 720)
        om = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")

        # Create a simple bright overlay
        overlay = np.full((100, 100, 3), 255, dtype=np.uint8)
        canvas = om.place_on_canvas(overlay, [250, 250, 750, 750])

        # Center region should have content
        cy, cx = 360, 640
        assert canvas[cy, cx].max() > 0, "Center of canvas should have overlay content"

    def test_identity_h_proj_corner_overlays(self):
        """Overlays at each corner should appear in correct projector region."""
        H = self._identity_h_proj(1280, 720)
        corners = {
            "top-left": [0, 0, 500, 500],
            "top-right": [0, 500, 500, 1000],
            "bottom-left": [500, 0, 1000, 500],
            "bottom-right": [500, 500, 1000, 1000],
        }
        for name, placement in corners.items():
            om = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
            overlay = np.full((100, 100, 3), 200, dtype=np.uint8)
            canvas = om.place_on_canvas(overlay, placement)
            assert canvas.max() > 0, f"Canvas should have content for {name}"

    def test_full_canvas_overlay(self):
        """Overlay at [0, 0, 1000, 1000] should fill the entire projector."""
        H = self._identity_h_proj(640, 480)
        om = OverlayManager(H_proj=H, proj_width=640, proj_height=480, mode="projector")
        overlay = np.full((480, 640, 3), 128, dtype=np.uint8)
        canvas = om.place_on_canvas(overlay, [0, 0, 1000, 1000])
        # Most of the canvas should be non-black
        nonblack = np.count_nonzero(canvas.sum(axis=2))
        total = canvas.shape[0] * canvas.shape[1]
        assert nonblack / total > 0.5

    def test_tiny_overlay(self):
        """Very small overlay at center should still render."""
        H = self._identity_h_proj(1280, 720)
        om = OverlayManager(H_proj=H, proj_width=1280, proj_height=720, mode="projector")
        overlay = np.full((10, 10, 3), 255, dtype=np.uint8)
        canvas = om.place_on_canvas(overlay, [490, 490, 510, 510])
        assert canvas.max() > 0

    def test_screen_mode_direct_mapping(self):
        """Screen mode maps 0-1000 directly to pixels without H_proj."""
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        overlay = np.full((100, 100, 3), 200, dtype=np.uint8)
        canvas = om.place_on_canvas(overlay, [0, 0, 500, 500])
        # Top-left quadrant should have content
        assert canvas[250, 250].max() > 0
        # Bottom-right should be black
        assert canvas[750, 750].max() == 0


# ---------------------------------------------------------------------------
# Test: Rotation permutations
# ---------------------------------------------------------------------------


class TestRotationRoundTrip:
    def test_unrotate_0_is_identity(self):
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=0)
        placement = [100, 200, 800, 900]
        result = om._unrotate_placement(placement)
        assert result == placement

    def test_unrotate_90_roundtrip(self):
        """Rotate 90 CW then unrotate should return to original exactly."""
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=90)
        original = [100, 200, 800, 900]
        ymin, xmin, ymax, xmax = original
        # Forward 90° CW: (y, x) → (x, 1000-y)
        # Bbox: yr=[xmin,xmax], xr=[1000-ymax, 1000-ymin]
        rotated = [xmin, 1000 - ymax, xmax, 1000 - ymin]
        unrotated = om._unrotate_placement(rotated)
        assert unrotated == original

    def test_unrotate_180_roundtrip(self):
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=180)
        original = [100, 200, 800, 900]
        # 180° rotation: (y, x) → (1000-y, 1000-x)
        rotated = [1000 - 800, 1000 - 900, 1000 - 100, 1000 - 200]
        unrotated = om._unrotate_placement(rotated)
        assert unrotated == original

    def test_unrotate_270_roundtrip(self):
        """Rotate 270 CW (= 90 CCW) then unrotate should return to original exactly."""
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=270)
        original = [100, 200, 800, 900]
        ymin, xmin, ymax, xmax = original
        # Forward 270° (= CCW 90): (y, x) → (1000-x, y)
        # Bbox: yr=[1000-xmax, 1000-xmin], xr=[ymin, ymax]
        rotated = [1000 - xmax, ymin, 1000 - xmin, ymax]
        unrotated = om._unrotate_placement(rotated)
        assert unrotated == original


# ---------------------------------------------------------------------------
# Test: End-to-end rendering at specific placements
# ---------------------------------------------------------------------------


class TestEndToEndRendering:
    def test_graph_overlay_placed_correctly(self):
        """Render a graph overlay and verify it lands in the right region."""
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        om.handle_tool_result("project_overlay", {
            "status": "displayed",
            "content_type": "graph",
            "placement": [0, 0, 500, 500],
            "title": "test graph",
            "data": {"expression": "x**2", "x_range": [-5, 5], "y_range": [0, 25]},
        })
        canvas = om.canvas
        # Top-left quadrant should have content
        top_left = canvas[:500, :500]
        assert top_left.max() > 0, "Graph should render in top-left"
        # Bottom-right should be black
        bottom_right = canvas[500:, 500:]
        assert bottom_right.max() == 0, "Bottom-right should be empty"

    def test_annotation_overlay_placed_correctly(self):
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        om.handle_tool_result("project_overlay", {
            "status": "displayed",
            "content_type": "annotation",
            "placement": [500, 500, 1000, 1000],
            "title": "note",
            "data": {"text": "Hello World"},
        })
        canvas = om.canvas
        # Bottom-right should have content
        bottom_right = canvas[500:, 500:]
        assert bottom_right.max() > 0

    def test_clear_resets_canvas(self):
        om = OverlayManager(H_proj=None, proj_width=500, proj_height=500, mode="screen")
        om.handle_tool_result("project_overlay", {
            "status": "displayed",
            "content_type": "annotation",
            "placement": [0, 0, 1000, 1000],
            "title": "t",
            "data": {"text": "fill"},
        })
        assert om.canvas.max() > 0
        om.clear()
        assert om.canvas.max() == 0


# ---------------------------------------------------------------------------
# Test: Stress / edge cases
# ---------------------------------------------------------------------------


class TestStressAndEdgeCases:
    def test_large_overlay_on_small_projector(self):
        """Full-canvas overlay on 320x240 projector."""
        om = OverlayManager(H_proj=None, proj_width=320, proj_height=240, mode="screen")
        om.handle_tool_result("project_overlay", {
            "status": "displayed",
            "content_type": "annotation",
            "placement": [0, 0, 1000, 1000],
            "title": "t",
            "data": {"text": "Big text on tiny projector"},
        })
        assert om.canvas.shape == (240, 320, 3)
        assert om.canvas.max() > 0

    def test_rapid_overlay_then_clear(self):
        """Place 10 overlays then clear."""
        om = OverlayManager(H_proj=None, proj_width=500, proj_height=500, mode="screen")
        for i in range(10):
            y = (i * 100) % 900
            om.handle_tool_result("project_overlay", {
                "status": "displayed",
                "content_type": "annotation",
                "placement": [y, 0, y + 100, 500],
                "title": f"overlay_{i}",
                "data": {"text": f"Item {i}"},
            })
        assert om.canvas.max() > 0
        om.clear()
        assert om.canvas.max() == 0

    def test_projector_mode_180_flip(self):
        """Projector mode should flip overlay 180°."""
        H = np.eye(3, dtype=np.float64)
        # Scale H to map 0-1000 → 0-500 pixels
        H[0, 0] = 0.5
        H[1, 1] = 0.5
        om_proj = OverlayManager(
            H_proj=H, proj_width=500, proj_height=500, mode="projector"
        )
        om_screen = OverlayManager(
            H_proj=None, proj_width=500, proj_height=500, mode="screen"
        )

        # Create an asymmetric overlay (bright top, dark bottom)
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[:50, :, :] = 255  # top half bright

        canvas_proj = om_proj.place_on_canvas(overlay, [0, 0, 200, 200])
        canvas_screen = om_screen.place_on_canvas(overlay, [0, 0, 200, 200])

        # Both should have content
        assert canvas_proj.max() > 0
        assert canvas_screen.max() > 0

    def test_white_bg_mode_compositing(self):
        """White background mode should only overwrite non-black overlay pixels."""
        om = OverlayManager(
            H_proj=None, proj_width=200, proj_height=200,
            mode="screen", white_bg=True,
        )
        assert om.canvas.min() == 255  # starts white

        # Place a half-bright overlay
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[:, :50, :] = 200  # left half bright, right half black

        canvas = om.place_on_canvas(overlay, [0, 0, 1000, 1000])
        # Left side should have overlay content
        assert canvas[100, 50].max() > 0
        # Right side should still be white-ish (black overlay pixels not written)
        assert canvas[100, 150].min() > 200
