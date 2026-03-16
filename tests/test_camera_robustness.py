"""Robustness tests for client/camera.py — stream simulation, caching,
dropped frames, extreme angles, rotation, JPEG quality, full pipeline,
and stats/staleness tracking.

Uses MockCaptureSource from camera_helpers to simulate capture behaviour
without any real camera or network connection.
"""

import time

import cv2
import numpy as np

from client.camera import (
    CameraCapture,
    compute_homography,
    detect_markers,
    encode_jpeg,
    rectify_frame,
)

from tests.camera_helpers import (
    make_blank_image,
    make_camera_with_mock,
    make_detector,
    make_table_image,
)


# ---------------------------------------------------------------------------
# Homography caching through frame sequences
# ---------------------------------------------------------------------------


class TestHomographyCaching:
    def test_cache_used_when_markers_lost(self):
        """Frame 1: markers present -> H cached.
        Frame 2: no markers -> cached H used, JPEG still returned.
        Frame 3: markers again -> H refreshed."""
        good = make_table_image()
        blank = make_blank_image()
        cam = make_camera_with_mock([good, blank, good])

        jpeg1, _, H1 = cam.get_rectified_frame()
        assert jpeg1 is not None
        assert H1 is not None

        jpeg2, _, H2 = cam.get_rectified_frame()
        assert jpeg2 is not None
        np.testing.assert_array_equal(H2, H1)

        jpeg3, _, H3 = cam.get_rectified_frame()
        assert jpeg3 is not None
        assert H3 is not None

    def test_cache_survives_multiple_lost_frames(self):
        """Frame 1 has markers; frames 2-10 have none; frame 11 has markers."""
        good = make_table_image()
        blank = make_blank_image()
        cam = make_camera_with_mock([good] + [blank] * 9 + [good])

        jpeg, _, H_first = cam.get_rectified_frame()
        assert jpeg is not None

        for _ in range(9):
            jpeg, _, H = cam.get_rectified_frame()
            assert jpeg is not None
            np.testing.assert_array_equal(H, H_first)

        jpeg, _, H_last = cam.get_rectified_frame()
        assert jpeg is not None
        assert H_last is not None

    def test_no_cache_no_markers_returns_none(self):
        """No cached H and no markers -> (None, None, None)."""
        blank = make_blank_image()
        cam = make_camera_with_mock([blank] * 5)

        for _ in range(5):
            jpeg, _, H = cam.get_rectified_frame()
            assert jpeg is None
            assert H is None

    def test_cache_not_polluted_by_bad_detection(self):
        """Good markers -> cache H1. Different marker positions -> new H2 != H1."""
        good1 = make_table_image(margin=120)
        good2 = make_table_image(margin=200)
        cam = make_camera_with_mock([good1, good2])

        _, _, H1 = cam.get_rectified_frame()
        _, _, H2 = cam.get_rectified_frame()
        assert H1 is not None and H2 is not None
        assert not np.allclose(H1, H2, atol=1e-3)


# ---------------------------------------------------------------------------
# Dropped frames and stream failures
# ---------------------------------------------------------------------------


class TestDroppedFrames:
    def test_dropped_frame_returns_none(self):
        """read() returning None -> get_rectified_frame returns (None, None, None)."""
        cam = make_camera_with_mock([None])
        jpeg, _, H = cam.get_rectified_frame()
        assert jpeg is None
        assert H is None

    def test_recovery_after_dropped_frames(self):
        """Frames 1-3 dropped, frame 4 valid with markers."""
        good = make_table_image()
        cam = make_camera_with_mock([None, None, None, good])

        for _ in range(3):
            jpeg, _, H = cam.get_rectified_frame()
            assert jpeg is None

        jpeg, _, H = cam.get_rectified_frame()
        assert jpeg is not None
        assert H is not None

    def test_source_read_called_per_frame(self):
        """Verify each get_rectified_frame() consumes exactly one source frame."""
        good = make_table_image()
        sentinel = np.full((800, 800, 3), 42, dtype=np.uint8)  # no markers
        cam = make_camera_with_mock([sentinel, good])

        # First call gets sentinel (no markers, no cached H => None)
        jpeg1, _, _ = cam.get_rectified_frame()
        assert jpeg1 is None

        # Second call gets good (markers found)
        jpeg2, _, H = cam.get_rectified_frame()
        assert jpeg2 is not None
        assert H is not None

    def test_exhausted_source_returns_none(self):
        """Once all frames are consumed, returns (None, None, None)."""
        good = make_table_image()
        cam = make_camera_with_mock([good])

        jpeg, _, _ = cam.get_rectified_frame()
        assert jpeg is not None

        jpeg, _, _ = cam.get_rectified_frame()
        assert jpeg is None


# ---------------------------------------------------------------------------
# Extreme camera angles
# ---------------------------------------------------------------------------


def _perspective_warp(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply a synthetic perspective transform simulating viewing from an angle."""
    h, w = image.shape[:2]
    t = np.clip(angle_deg / 90.0, 0.0, 0.7)
    inset_top = int(w * t * 0.3)
    inset_bottom = 0
    src = np.array([
        [0, 0], [w, 0], [w, h], [0, h],
    ], dtype=np.float32)
    dst = np.array([
        [inset_top, int(h * t * 0.15)],
        [w - inset_top, int(h * t * 0.15)],
        [w - inset_bottom, h - int(h * t * 0.05)],
        [inset_bottom, h - int(h * t * 0.05)],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), borderValue=(200, 200, 200))


class TestExtremeCameraAngles:
    def test_markers_at_30_degree_angle(self):
        table = make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 30)
        detected = detect_markers(warped, make_detector())
        assert len(detected) == 4

    def test_markers_at_45_degree_angle(self):
        table = make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 45)
        detected = detect_markers(warped, make_detector())
        assert len(detected) >= 3

    def test_markers_at_60_degree_angle(self):
        """Very steep angle — detection may fail gracefully."""
        table = make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 60)
        detected = detect_markers(warped, make_detector())
        assert isinstance(detected, dict)

    def test_markers_with_scale_reduction(self):
        """Markers appear small (camera far away)."""
        table = make_table_image(size=1000, marker_size=60, margin=200)
        small = cv2.resize(table, (500, 500))
        detected = detect_markers(small, make_detector())
        assert isinstance(detected, dict)

    def test_markers_with_barrel_distortion(self):
        """Slight barrel distortion (simulating phone lens)."""
        table = make_table_image(size=1000, marker_size=120, margin=200)
        h, w = table.shape[:2]
        cx, cy = w / 2, h / 2
        k1 = -0.05
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                dx, dy = (x - cx) / cx, (y - cy) / cy
                r2 = dx * dx + dy * dy
                factor = 1 + k1 * r2
                map_x[y, x] = cx + dx * factor * cx
                map_y[y, x] = cy + dy * factor * cy
        distorted = cv2.remap(table, map_x, map_y, cv2.INTER_LINEAR,
                              borderValue=(200, 200, 200))
        detected = detect_markers(distorted, make_detector())
        assert len(detected) == 4


# ---------------------------------------------------------------------------
# Rotation parameter testing
# ---------------------------------------------------------------------------


class TestRotation:
    def _get_rectified_for_rotation(self, rotate: int) -> bytes:
        good = make_table_image()
        cam = make_camera_with_mock([good], rotate=rotate)
        jpeg, *_ = cam.get_rectified_frame()
        assert jpeg is not None
        return jpeg

    def test_rotation_0_no_change(self):
        jpeg = self._get_rectified_for_rotation(0)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[0] == img.shape[1]

    def test_rotation_90_applied(self):
        jpeg = self._get_rectified_for_rotation(90)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_rotation_180_applied(self):
        """180 rotation should flip content."""
        good = make_table_image()
        cam0 = make_camera_with_mock([good.copy()], rotate=0)
        cam180 = make_camera_with_mock([good.copy()], rotate=180)

        jpeg0, *_ = cam0.get_rectified_frame()
        jpeg180, *_ = cam180.get_rectified_frame()

        img0 = cv2.imdecode(np.frombuffer(jpeg0, np.uint8), cv2.IMREAD_COLOR)
        img180 = cv2.imdecode(np.frombuffer(jpeg180, np.uint8), cv2.IMREAD_COLOR)

        img0_rotated = cv2.rotate(img0, cv2.ROTATE_180)
        diff = np.abs(img180.astype(float) - img0_rotated.astype(float)).mean()
        assert diff < 5

    def test_rotation_270_applied(self):
        jpeg = self._get_rectified_for_rotation(270)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None

    def test_rotation_preserves_jpeg_validity(self):
        """All rotations produce valid JPEG bytes."""
        for rot in [0, 90, 180, 270]:
            jpeg = self._get_rectified_for_rotation(rot)
            assert jpeg[:2] == b"\xff\xd8", f"Rotation {rot} produced invalid JPEG"
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            assert img is not None


# ---------------------------------------------------------------------------
# JPEG quality and size
# ---------------------------------------------------------------------------


class TestJPEGQuality:
    def test_quality_70_smaller_than_85(self):
        frame = make_table_image()
        j70 = encode_jpeg(frame, quality=70)
        j85 = encode_jpeg(frame, quality=85)
        assert len(j70) < len(j85)

    def test_quality_70_still_readable(self):
        frame = make_table_image()
        j70 = encode_jpeg(frame, quality=70)
        decoded = cv2.imdecode(np.frombuffer(j70, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == frame.shape

    def test_quality_affects_marker_detection(self):
        """Encode at quality 70, decode, run detection — should still find markers."""
        frame = make_table_image()
        j70 = encode_jpeg(frame, quality=70)
        decoded = cv2.imdecode(np.frombuffer(j70, np.uint8), cv2.IMREAD_COLOR)
        detected = detect_markers(decoded, make_detector())
        assert len(detected) == 4


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_full_pipeline_overhead_view(self):
        """Overhead camera -> detect -> rectify -> JPEG -> decode -> verify."""
        table = make_table_image(size=1000, marker_size=120, margin=150)
        detector = make_detector()
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)

        detected = detect_markers(table, detector)
        assert len(detected) == 4

        H = compute_homography(detected, dst)
        assert H is not None

        rectified = rectify_frame(table, H, (768, 768))
        jpeg = encode_jpeg(rectified)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (768, 768, 3)
        assert decoded.max() > 0
        center = decoded[300:468, 300:468]
        assert center.mean() > 100

    def test_full_pipeline_angled_view(self):
        """Camera at 30 deg -> detect -> rectify -> JPEG -> decode."""
        table = make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 30)
        detector = make_detector()
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)

        detected = detect_markers(warped, detector)
        assert len(detected) == 4

        H = compute_homography(detected, dst)
        assert H is not None

        rectified = rectify_frame(warped, H, (768, 768))
        jpeg = encode_jpeg(rectified, quality=70)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (768, 768, 3)
        assert decoded.max() > 0

    def test_full_pipeline_with_rotation(self):
        """Camera at angle + 180 rotation via CameraCapture."""
        table = make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 25)
        cam = make_camera_with_mock([warped], rotate=180)

        jpeg, _, H = cam.get_rectified_frame()
        assert jpeg is not None
        assert H is not None
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (768, 768, 3)


# ---------------------------------------------------------------------------
# Stats and staleness tracking
# ---------------------------------------------------------------------------


class TestStatsTracking:
    def test_stats_increment_on_capture(self):
        good = make_table_image()
        cam = make_camera_with_mock([good] * 3)

        cam.get_rectified_frame()
        assert cam.stats["frames_captured"] == 1
        assert cam.stats["frames_with_markers"] == 1

        cam.get_rectified_frame()
        assert cam.stats["frames_captured"] == 2
        assert cam.stats["frames_with_markers"] == 2

    def test_stats_increment_on_cache_use(self):
        good = make_table_image()
        blank = make_blank_image()
        cam = make_camera_with_mock([good, blank])

        cam.get_rectified_frame()
        assert cam.stats["frames_using_cache"] == 0

        cam.get_rectified_frame()
        assert cam.stats["frames_using_cache"] == 1

    def test_marker_staleness_increases(self):
        good = make_table_image()
        cam = make_camera_with_mock([good])

        assert cam.marker_staleness_seconds is None

        cam.get_rectified_frame()
        s1 = cam.marker_staleness_seconds
        assert s1 is not None
        assert s1 >= 0

        time.sleep(0.05)
        s2 = cam.marker_staleness_seconds
        assert s2 > s1

    def test_consecutive_failures_tracked(self):
        cam = make_camera_with_mock([None] * 5)

        for i in range(1, 6):
            cam.get_rectified_frame()
            assert cam.stats["consecutive_failures"] == i

    def test_consecutive_failures_reset_on_success(self):
        good = make_table_image()
        cam = make_camera_with_mock([None, None, good])

        cam.get_rectified_frame()
        assert cam.stats["consecutive_failures"] == 1

        cam.get_rectified_frame()
        assert cam.stats["consecutive_failures"] == 2

        cam.get_rectified_frame()
        assert cam.stats["consecutive_failures"] == 0

    def test_stats_starts_at_zero(self):
        cam = CameraCapture(url="http://fake:8080")
        assert cam.stats["frames_captured"] == 0
        assert cam.stats["frames_with_markers"] == 0
        assert cam.stats["frames_using_cache"] == 0
        assert cam.stats["consecutive_failures"] == 0

    def test_marker_staleness_none_before_detection(self):
        cam = CameraCapture(url="http://fake:8080")
        assert cam.marker_staleness_seconds is None
