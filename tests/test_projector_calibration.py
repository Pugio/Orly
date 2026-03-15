"""Tests for projector calibration functions (PoC 2)."""

import numpy as np
import cv2
import pytest

from calibration.projector_calibrate import (
    generate_calibration_grid,
    create_dot_image,
    find_bright_centroid,
    camera_to_table,
    compute_projector_homography,
    table_to_projector,
)


# ---------------------------------------------------------------------------
# generate_calibration_grid
# ---------------------------------------------------------------------------

class TestGenerateCalibrationGrid:
    def test_correct_number_of_points(self):
        pts = generate_calibration_grid(1280, 720, cols=4, rows=3)
        assert len(pts) == 12

    def test_correct_number_of_points_custom(self):
        pts = generate_calibration_grid(1280, 720, cols=5, rows=4)
        assert len(pts) == 20

    def test_all_within_bounds(self):
        pts = generate_calibration_grid(1280, 720, cols=4, rows=3, margin=0.1)
        for x, y in pts:
            assert 0 <= x <= 1280, f"x={x} out of bounds"
            assert 0 <= y <= 720, f"y={y} out of bounds"

    def test_respects_margin(self):
        w, h = 1000, 500
        margin = 0.2
        pts = generate_calibration_grid(w, h, cols=4, rows=3, margin=margin)
        for x, y in pts:
            assert x >= w * margin, f"x={x} violates left margin"
            assert x <= w * (1 - margin), f"x={x} violates right margin"
            assert y >= h * margin, f"y={y} violates top margin"
            assert y <= h * (1 - margin), f"y={y} violates bottom margin"

    def test_evenly_spaced_x(self):
        pts = generate_calibration_grid(1280, 720, cols=4, rows=3, margin=0.1)
        # Extract unique x values (first row)
        first_row = pts[:4]
        xs = [p[0] for p in first_row]
        # Spacing between consecutive x values should be equal
        spacings = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        for s in spacings:
            assert abs(s - spacings[0]) <= 1, f"Uneven x spacing: {spacings}"

    def test_evenly_spaced_y(self):
        pts = generate_calibration_grid(1280, 720, cols=4, rows=3, margin=0.1)
        # Extract unique y values (first column: indices 0, 4, 8 for 4-col grid)
        first_col = [pts[i * 4] for i in range(3)]
        ys = [p[1] for p in first_col]
        spacings = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
        for s in spacings:
            assert abs(s - spacings[0]) < 1, f"Uneven y spacing: {spacings}"

    def test_ordered_row_major(self):
        """Points should be ordered row-major: left-to-right, top-to-bottom."""
        pts = generate_calibration_grid(1280, 720, cols=4, rows=3)
        for row in range(3):
            row_pts = pts[row * 4 : (row + 1) * 4]
            xs = [p[0] for p in row_pts]
            assert xs == sorted(xs), "Points within a row should be left-to-right"
        # First point of each row should have increasing y
        col_ys = [pts[row * 4][1] for row in range(3)]
        assert col_ys == sorted(col_ys), "Rows should go top-to-bottom"

    def test_zero_margin_fills_display(self):
        pts = generate_calibration_grid(1280, 720, cols=2, rows=2, margin=0.0)
        xs = sorted(set(p[0] for p in pts))
        ys = sorted(set(p[1] for p in pts))
        assert xs[0] == 0
        assert xs[-1] == 1280
        assert ys[0] == 0
        assert ys[-1] == 720


# ---------------------------------------------------------------------------
# create_dot_image
# ---------------------------------------------------------------------------

class TestCreateDotImage:
    def test_output_shape(self):
        img = create_dot_image(1280, 720, (640, 360))
        assert img.shape == (720, 1280, 3)

    def test_dtype_uint8(self):
        img = create_dot_image(1280, 720, (640, 360))
        assert img.dtype == np.uint8

    def test_mostly_black(self):
        img = create_dot_image(1280, 720, (640, 360), dot_radius=15)
        total_pixels = 1280 * 720
        bright_pixels = (img.sum(axis=2) > 0).sum()
        # A circle of radius 15 has ~706 pixels. Much less than total.
        assert bright_pixels / total_pixels < 0.01

    def test_dot_at_correct_position(self):
        cx, cy = 400, 300
        img = create_dot_image(1280, 720, (cx, cy), dot_radius=10)
        # The center pixel should be white
        assert img[cy, cx].sum() > 700, "Center of dot should be bright white"

    def test_dot_away_from_center_is_black(self):
        img = create_dot_image(1280, 720, (100, 100), dot_radius=10)
        # A pixel far from the dot should be black
        assert img[600, 600].sum() == 0

    def test_custom_radius(self):
        img_small = create_dot_image(1280, 720, (640, 360), dot_radius=5)
        img_large = create_dot_image(1280, 720, (640, 360), dot_radius=30)
        bright_small = (img_small.sum(axis=2) > 0).sum()
        bright_large = (img_large.sum(axis=2) > 0).sum()
        assert bright_large > bright_small


# ---------------------------------------------------------------------------
# find_bright_centroid
# ---------------------------------------------------------------------------

class TestFindBrightCentroid:
    def test_finds_synthetic_blob(self):
        bg = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = bg.copy()
        cv2.circle(frame, (320, 240), 20, (255, 255, 255), -1)
        result = find_bright_centroid(frame, bg, threshold=50)
        assert result is not None
        x, y = result
        assert abs(x - 320) < 3, f"x centroid off: {x}"
        assert abs(y - 240) < 3, f"y centroid off: {y}"

    def test_returns_none_for_black_image(self):
        bg = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = find_bright_centroid(frame, bg, threshold=50)
        assert result is None

    def test_finds_off_center_blob(self):
        bg = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = bg.copy()
        cv2.circle(frame, (100, 400), 15, (255, 255, 255), -1)
        result = find_bright_centroid(frame, bg, threshold=50)
        assert result is not None
        x, y = result
        assert abs(x - 100) < 3
        assert abs(y - 400) < 3

    def test_background_subtraction(self):
        """If the bright region is also in the background, it should be cancelled out."""
        bg = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(bg, (320, 240), 20, (200, 200, 200), -1)
        frame = bg.copy()
        # No new bright region beyond background
        result = find_bright_centroid(frame, bg, threshold=50)
        assert result is None

    def test_noisy_background(self):
        """Should still find a clear bright dot against uniform noise."""
        rng = np.random.RandomState(42)
        bg = rng.randint(0, 30, (480, 640, 3), dtype=np.uint8)
        frame = bg.copy()
        cv2.circle(frame, (200, 150), 20, (255, 255, 255), -1)
        result = find_bright_centroid(frame, bg, threshold=50)
        assert result is not None
        x, y = result
        assert abs(x - 200) < 5
        assert abs(y - 150) < 5


# ---------------------------------------------------------------------------
# camera_to_table
# ---------------------------------------------------------------------------

class TestCameraToTable:
    def test_identity_homography(self):
        H = np.eye(3, dtype=np.float64)
        x, y = camera_to_table((100.0, 200.0), H)
        assert abs(x - 100.0) < 1e-6
        assert abs(y - 200.0) < 1e-6

    def test_scaling_homography(self):
        # Scale by 2x
        H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
        x, y = camera_to_table((50.0, 75.0), H)
        assert abs(x - 100.0) < 1e-6
        assert abs(y - 150.0) < 1e-6

    def test_translation_homography(self):
        # Translate by (10, 20)
        H = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=np.float64)
        x, y = camera_to_table((5.0, 5.0), H)
        assert abs(x - 15.0) < 1e-6
        assert abs(y - 25.0) < 1e-6

    def test_known_perspective(self):
        """Use cv2.findHomography with known correspondences, verify transform."""
        src = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        dst = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        x, y = camera_to_table((50.0, 50.0), H)
        assert abs(x - 250.0) < 1.0
        assert abs(y - 250.0) < 1.0


# ---------------------------------------------------------------------------
# compute_projector_homography
# ---------------------------------------------------------------------------

class TestComputeProjectorHomography:
    def test_returns_3x3_matrix(self):
        table_pts = [(0, 0), (1000, 0), (1000, 1000), (0, 1000),
                     (500, 0), (500, 1000), (0, 500), (1000, 500)]
        proj_pts = [(0, 0), (1280, 0), (1280, 720), (0, 720),
                    (640, 0), (640, 720), (0, 360), (1280, 360)]
        H = compute_projector_homography(table_pts, proj_pts)
        assert H.shape == (3, 3)

    def test_round_trip(self):
        """If we compute H from known correspondences, applying it should recover projector points."""
        table_pts = [(100, 100), (900, 100), (900, 600), (100, 600),
                     (500, 350), (100, 350), (900, 350), (500, 100)]
        proj_pts = [(128, 72), (1152, 72), (1152, 648), (128, 648),
                    (640, 360), (128, 360), (1152, 360), (640, 72)]
        H = compute_projector_homography(table_pts, proj_pts)
        for tp, pp in zip(table_pts, proj_pts):
            result = table_to_projector(tp, H)
            assert abs(result[0] - pp[0]) < 2, f"x mismatch: {result[0]} vs {pp[0]}"
            assert abs(result[1] - pp[1]) < 2, f"y mismatch: {result[1]} vs {pp[1]}"

    def test_minimum_four_points(self):
        """Homography needs at least 4 correspondences."""
        table_pts = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        proj_pts = [(0, 0), (1280, 0), (1280, 720), (0, 720)]
        H = compute_projector_homography(table_pts, proj_pts)
        assert H is not None
        assert H.shape == (3, 3)


# ---------------------------------------------------------------------------
# table_to_projector
# ---------------------------------------------------------------------------

class TestTableToProjector:
    def test_identity_homography(self):
        H = np.eye(3, dtype=np.float64)
        x, y = table_to_projector((100.0, 200.0), H)
        assert abs(x - 100) < 1
        assert abs(y - 200) < 1

    def test_scaling_homography(self):
        H = np.array([[1.28, 0, 0], [0, 0.72, 0], [0, 0, 1]], dtype=np.float64)
        x, y = table_to_projector((1000.0, 1000.0), H)
        assert abs(x - 1280) < 1
        assert abs(y - 720) < 1

    def test_returns_integers(self):
        H = np.eye(3, dtype=np.float64)
        result = table_to_projector((123.7, 456.2), H)
        assert isinstance(result[0], (int, np.integer))
        assert isinstance(result[1], (int, np.integer))

    def test_boundary_point_origin(self):
        H = np.eye(3, dtype=np.float64)
        x, y = table_to_projector((0.0, 0.0), H)
        assert x == 0
        assert y == 0
