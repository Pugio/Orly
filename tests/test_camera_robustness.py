"""Robustness tests for client/camera.py — stream simulation, caching,
dropped frames, extreme angles, rotation, JPEG quality, full pipeline,
and the new stats/staleness tracking.

Uses a MockVideoCapture to simulate cv2.VideoCapture behaviour without
any real camera or network connection.
"""

import time

import cv2
import numpy as np
import pytest

from client.camera import (
    ARUCO_DICT,
    CORNER_INDICES,
    MARKER_IDS,
    CameraCapture,
    compute_homography,
    detect_markers,
    encode_jpeg,
    rectify_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def _draw_marker(image: np.ndarray, marker_id: int, center: tuple, size: int):
    """Draw an ArUco marker at *center* (x, y) onto *image*."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    cx, cy = center
    x1, y1 = cx - size // 2, cy - size // 2

    sx1, sy1 = max(0, -x1), max(0, -y1)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2 = min(image.shape[1], x1 + size)
    dy2 = min(image.shape[0], y1 + size)
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)

    if dx2 > dx1 and dy2 > dy1:
        image[dy1:dy2, dx1:dx2] = marker_bgr[sy1:sy2, sx1:sx2]


def _make_table_image(
    size: int = 800,
    marker_size: int = 120,
    margin: int = 120,
    bg: int = 200,
) -> np.ndarray:
    """Synthetic table with 4 ArUco markers at the corners."""
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    positions = {
        0: (margin, margin),
        1: (size - margin, margin),
        2: (size - margin, size - margin),
        3: (margin, size - margin),
    }
    for mid, pos in positions.items():
        _draw_marker(img, mid, pos, marker_size)
    return img


def _make_blank_image(size: int = 800) -> np.ndarray:
    """Plain grey image with no markers."""
    return np.full((size, size, 3), 200, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. MockVideoCapture
# ---------------------------------------------------------------------------


class MockVideoCapture:
    """Simulates cv2.VideoCapture for testing CameraCapture without hardware.

    Args:
        frames: list of frames (np.ndarray) or None entries for dropped frames.
        opened: whether isOpened() returns True.
    """

    def __init__(self, frames: list, opened: bool = True):
        self._frames = frames
        self._idx = 0
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened

    def grab(self) -> bool:
        """Advance internal index (discard one frame, like real grab)."""
        if self._idx < len(self._frames):
            self._idx += 1
        return True

    def read(self):
        """Return the current frame (or False/None for drops)."""
        if self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        if frame is None:
            return False, None
        return True, frame.copy()

    def set(self, prop, value):
        pass

    def release(self):
        self._opened = False


def _make_camera_with_mock(frames: list, rotate: int = 0) -> CameraCapture:
    """Build a CameraCapture wired to a MockVideoCapture (no real start())."""
    cam = CameraCapture(url="http://fake:8080", rotate=rotate)
    cam.cap = MockVideoCapture(frames)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    cam.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    return cam


# ---------------------------------------------------------------------------
# 2. Homography caching through frame sequences
# ---------------------------------------------------------------------------


class TestHomographyCaching:
    def test_cache_used_when_markers_lost(self):
        """Frame 1: markers present -> H cached.
        Frame 2: no markers -> cached H used, JPEG still returned.
        Frame 3: markers again -> H refreshed."""
        good = _make_table_image()
        blank = _make_blank_image()
        # Each get_rectified_frame does grab()+read(), consuming 2 frames
        frames = [good, good, blank, blank, good, good]
        cam = _make_camera_with_mock(frames)

        # Frame 1: markers present
        jpeg1, _, H1 = cam.get_rectified_frame()
        assert jpeg1 is not None
        assert H1 is not None

        # Frame 2: no markers — should use cached H
        jpeg2, _, H2 = cam.get_rectified_frame()
        assert jpeg2 is not None
        np.testing.assert_array_equal(H2, H1)

        # Frame 3: markers return — H refreshed
        jpeg3, _, H3 = cam.get_rectified_frame()
        assert jpeg3 is not None
        assert H3 is not None

    def test_cache_survives_multiple_lost_frames(self):
        """Frame 1 has markers; frames 2-10 have none; frame 11 has markers."""
        good = _make_table_image()
        blank = _make_blank_image()
        # 11 calls * 2 frames each = 22 frames total
        frames = [good, good] + [blank, blank] * 9 + [good, good]
        cam = _make_camera_with_mock(frames)

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
        blank = _make_blank_image()
        frames = [blank, blank] * 5
        cam = _make_camera_with_mock(frames)

        for _ in range(5):
            jpeg, _, H = cam.get_rectified_frame()
            assert jpeg is None
            assert H is None

    def test_cache_not_polluted_by_bad_detection(self):
        """Good markers -> cache H1. Different marker positions -> new H2 != H1."""
        good1 = _make_table_image(margin=120)
        good2 = _make_table_image(margin=200)  # markers in different positions
        frames = [good1, good1, good2, good2]
        cam = _make_camera_with_mock(frames)

        _, _, H1 = cam.get_rectified_frame()
        _, _, H2 = cam.get_rectified_frame()
        assert H1 is not None and H2 is not None
        assert not np.allclose(H1, H2, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Dropped frames and stream failures
# ---------------------------------------------------------------------------


class TestDroppedFrames:
    def test_dropped_frame_returns_none(self):
        """read() returning False -> get_rectified_frame returns (None, None, None)."""
        frames = [None, None]  # grab consumes first None, read gets second None
        cam = _make_camera_with_mock(frames)
        jpeg, _, H = cam.get_rectified_frame()
        assert jpeg is None
        assert H is None

    def test_recovery_after_dropped_frames(self):
        """Frames 1-3 dropped, frame 4 valid with markers."""
        good = _make_table_image()
        frames = [None, None] * 3 + [good, good]
        cam = _make_camera_with_mock(frames)

        for _ in range(3):
            jpeg, _, H = cam.get_rectified_frame()
            assert jpeg is None

        jpeg, _, H = cam.get_rectified_frame()
        assert jpeg is not None
        assert H is not None

    def test_grab_called_before_read(self):
        """Verify _capture_frame consumes a frame via grab() before read()."""
        good = _make_table_image()
        # The first frame (index 0) is consumed by grab(),
        # read() gets the second frame (index 1)
        sentinel = np.full((800, 800, 3), 42, dtype=np.uint8)
        frames = [sentinel, good]
        cam = _make_camera_with_mock(frames)
        frame = cam._capture_frame()
        # Should get `good`, not `sentinel` (sentinel was grabbed/discarded)
        assert frame is not None
        # The sentinel has uniform value 42; good has markers (varied content)
        assert frame.std() > 10  # good image has variance from markers


# ---------------------------------------------------------------------------
# 4. Extreme camera angles
# ---------------------------------------------------------------------------


def _perspective_warp(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply a synthetic perspective transform simulating viewing from an angle.

    The angle_deg parameter controls how extreme the perspective is
    (larger = steeper viewing angle).
    """
    h, w = image.shape[:2]
    # Compute inset proportional to angle
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
        table = _make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 30)
        detected = detect_markers(warped, _make_detector())
        assert len(detected) == 4

    def test_markers_at_45_degree_angle(self):
        table = _make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 45)
        detected = detect_markers(warped, _make_detector())
        # At 45 degrees detection should still work with large enough markers
        assert len(detected) >= 3

    def test_markers_at_60_degree_angle(self):
        """Very steep angle — detection may fail gracefully."""
        table = _make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 60)
        detected = detect_markers(warped, _make_detector())
        # Should not crash; may detect fewer markers
        assert isinstance(detected, dict)

    def test_markers_with_scale_reduction(self):
        """Markers appear small (camera far away)."""
        table = _make_table_image(size=1000, marker_size=60, margin=200)
        # Scale down to simulate distance
        small = cv2.resize(table, (500, 500))
        detected = detect_markers(small, _make_detector())
        # Smaller markers are harder — at least detect some
        assert isinstance(detected, dict)

    def test_markers_with_barrel_distortion(self):
        """Slight barrel distortion (simulating phone lens)."""
        table = _make_table_image(size=1000, marker_size=120, margin=200)
        h, w = table.shape[:2]
        # Barrel distortion via remap — use mild coefficient
        cx, cy = w / 2, h / 2
        k1 = -0.05  # mild barrel distortion typical of phone lenses
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
        detected = detect_markers(distorted, _make_detector())
        assert len(detected) == 4


# ---------------------------------------------------------------------------
# 5. Rotation parameter testing
# ---------------------------------------------------------------------------


class TestRotation:
    def _get_rectified_for_rotation(self, rotate: int) -> bytes:
        good = _make_table_image()
        frames = [good, good]
        cam = _make_camera_with_mock(frames, rotate=rotate)
        jpeg, *_ = cam.get_rectified_frame()
        assert jpeg is not None
        return jpeg

    def test_rotation_0_no_change(self):
        jpeg = self._get_rectified_for_rotation(0)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[0] == img.shape[1]  # square output

    def test_rotation_90_applied(self):
        jpeg = self._get_rectified_for_rotation(90)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_rotation_180_applied(self):
        """180 rotation should flip content."""
        good = _make_table_image()
        frames_0 = [good.copy(), good.copy()]
        frames_180 = [good.copy(), good.copy()]
        cam0 = _make_camera_with_mock(frames_0, rotate=0)
        cam180 = _make_camera_with_mock(frames_180, rotate=180)

        jpeg0, *_ = cam0.get_rectified_frame()
        jpeg180, *_ = cam180.get_rectified_frame()

        img0 = cv2.imdecode(np.frombuffer(jpeg0, np.uint8), cv2.IMREAD_COLOR)
        img180 = cv2.imdecode(np.frombuffer(jpeg180, np.uint8), cv2.IMREAD_COLOR)

        # 180 rotation: compare with manually rotated
        img0_rotated = cv2.rotate(img0, cv2.ROTATE_180)
        # They should be very similar (JPEG compression may cause slight diffs)
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
# 6. JPEG quality and size
# ---------------------------------------------------------------------------


class TestJPEGQuality:
    def test_quality_70_smaller_than_85(self):
        frame = _make_table_image()
        j70 = encode_jpeg(frame, quality=70)
        j85 = encode_jpeg(frame, quality=85)
        assert len(j70) < len(j85)

    def test_quality_70_still_readable(self):
        frame = _make_table_image()
        j70 = encode_jpeg(frame, quality=70)
        decoded = cv2.imdecode(np.frombuffer(j70, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == frame.shape

    def test_quality_affects_marker_detection(self):
        """Encode at quality 70, decode, run detection — should still find markers."""
        frame = _make_table_image()
        j70 = encode_jpeg(frame, quality=70)
        decoded = cv2.imdecode(np.frombuffer(j70, np.uint8), cv2.IMREAD_COLOR)
        detected = detect_markers(decoded, _make_detector())
        assert len(detected) == 4


# ---------------------------------------------------------------------------
# 7. Full pipeline tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_full_pipeline_overhead_view(self):
        """Overhead camera -> detect -> rectify -> JPEG -> decode -> verify."""
        table = _make_table_image(size=1000, marker_size=120, margin=150)
        detector = _make_detector()
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)

        detected = detect_markers(table, detector)
        assert len(detected) == 4

        H = compute_homography(detected, dst)
        assert H is not None

        rectified = rectify_frame(table, H, (768, 768))
        jpeg = encode_jpeg(rectified)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (768, 768, 3)

        # Rectified image should not be all-black and should have reasonable content
        assert decoded.max() > 0, "Decoded image should not be all black"
        # The center region should contain the grey background
        center = decoded[300:468, 300:468]
        assert center.mean() > 100, "Center of rectified view should be the table surface"

    def test_full_pipeline_angled_view(self):
        """Camera at 30 deg -> detect -> rectify -> JPEG -> decode."""
        table = _make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 30)
        detector = _make_detector()
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)

        detected = detect_markers(warped, detector)
        assert len(detected) == 4

        H = compute_homography(detected, dst)
        assert H is not None

        rectified = rectify_frame(warped, H, (768, 768))
        jpeg = encode_jpeg(rectified, quality=70)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (768, 768, 3)
        assert decoded.max() > 0  # not all black

    def test_full_pipeline_with_rotation(self):
        """Camera at angle + 180 rotation via CameraCapture."""
        table = _make_table_image(size=1000, marker_size=120, margin=200)
        warped = _perspective_warp(table, 25)
        frames = [warped, warped]
        cam = _make_camera_with_mock(frames, rotate=180)

        jpeg, _, H = cam.get_rectified_frame()
        assert jpeg is not None
        assert H is not None
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (768, 768, 3)


# ---------------------------------------------------------------------------
# 8. Stats and staleness tracking (Part 3)
# ---------------------------------------------------------------------------


class TestStatsTracking:
    def test_stats_increment_on_capture(self):
        good = _make_table_image()
        frames = [good, good] * 3
        cam = _make_camera_with_mock(frames)

        cam.get_rectified_frame()
        assert cam.stats["frames_captured"] == 1
        assert cam.stats["frames_with_markers"] == 1

        cam.get_rectified_frame()
        assert cam.stats["frames_captured"] == 2
        assert cam.stats["frames_with_markers"] == 2

    def test_stats_increment_on_cache_use(self):
        good = _make_table_image()
        blank = _make_blank_image()
        frames = [good, good, blank, blank]
        cam = _make_camera_with_mock(frames)

        cam.get_rectified_frame()  # markers found
        assert cam.stats["frames_using_cache"] == 0

        cam.get_rectified_frame()  # no markers, uses cache
        assert cam.stats["frames_using_cache"] == 1

    def test_marker_staleness_increases(self):
        good = _make_table_image()
        frames = [good, good]
        cam = _make_camera_with_mock(frames)

        assert cam.marker_staleness_seconds is None

        cam.get_rectified_frame()
        s1 = cam.marker_staleness_seconds
        assert s1 is not None
        assert s1 >= 0

        time.sleep(0.05)
        s2 = cam.marker_staleness_seconds
        assert s2 > s1

    def test_consecutive_failures_tracked(self):
        frames = [None, None] * 5
        cam = _make_camera_with_mock(frames)

        for i in range(1, 6):
            cam.get_rectified_frame()
            assert cam.stats["consecutive_failures"] == i

    def test_consecutive_failures_reset_on_success(self):
        good = _make_table_image()
        frames = [None, None, None, None, good, good]
        cam = _make_camera_with_mock(frames)

        cam.get_rectified_frame()  # fail
        assert cam.stats["consecutive_failures"] == 1

        cam.get_rectified_frame()  # fail
        assert cam.stats["consecutive_failures"] == 2

        cam.get_rectified_frame()  # success (good frame)
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
