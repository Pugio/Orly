"""Extended tests for client/camera.py — pure functions with real cv2/numpy."""

import cv2
import numpy as np
import pytest

from client.camera import (
    ARUCO_DICT,
    CORNER_INDICES,
    MARKER_IDS,
    compute_homography,
    detect_markers,
    encode_jpeg,
    rectify_frame,
)


# ---------------------------------------------------------------------------
# Helper: generate synthetic ArUco marker images
# ---------------------------------------------------------------------------


def _make_marker_image(marker_ids, size=600, marker_size=100, margin=50):
    """Generate a synthetic image with ArUco markers at four corners."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img = np.ones((size, size, 3), dtype=np.uint8) * 200

    positions = [
        (margin, margin),
        (size - margin - marker_size, margin),
        (size - margin - marker_size, size - margin - marker_size),
        (margin, size - margin - marker_size),
    ]

    for mid, (x, y) in zip(marker_ids, positions):
        marker_img = cv2.aruco.generateImageMarker(dictionary, mid, marker_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        img[y:y + marker_size, x:x + marker_size] = marker_bgr

    return img


def _make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


# ---------------------------------------------------------------------------
# detect_markers
# ---------------------------------------------------------------------------


class TestDetectMarkersExtended:
    def test_detects_subset_of_markers(self):
        """Should detect only the markers that are present."""
        img = _make_marker_image([0, 1, 2, 3])
        # Blank out marker 2 area
        img[450:550, 450:550] = 200
        detector = _make_detector()
        result = detect_markers(img, detector)
        # Should detect at least markers 0, 1, 3 (marker 2 was blanked)
        assert 0 in result
        assert 1 in result
        assert 3 in result

    def test_corner_format_float32(self):
        img = _make_marker_image([0, 1, 2, 3])
        detector = _make_detector()
        result = detect_markers(img, detector)
        for mid, corners in result.items():
            assert corners.dtype == np.float32

    def test_different_marker_ids(self):
        """Works with non-default marker IDs."""
        img = _make_marker_image([10, 11, 12, 13])
        detector = _make_detector()
        result = detect_markers(img, detector)
        assert set(result.keys()) == {10, 11, 12, 13}

    def test_grayscale_input_fails(self):
        """detect_markers expects BGR input — grayscale should raise."""
        gray = np.zeros((400, 400), dtype=np.uint8)
        detector = _make_detector()
        with pytest.raises(cv2.error):
            detect_markers(gray, detector)


# ---------------------------------------------------------------------------
# compute_homography
# ---------------------------------------------------------------------------


class TestComputeHomographyExtended:
    def test_missing_one_marker_returns_none(self):
        detected = {
            0: np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=np.float32),
            1: np.array([[540, 10], [590, 10], [590, 60], [540, 60]], dtype=np.float32),
            2: np.array([[540, 540], [590, 540], [590, 590], [540, 590]], dtype=np.float32),
        }
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)
        assert compute_homography(detected, dst) is None

    def test_missing_all_markers_returns_none(self):
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)
        assert compute_homography({}, dst) is None

    def test_uses_correct_inner_corners(self):
        """Verify CORNER_INDICES mapping: the computed H should use inner corners."""
        # Create markers where inner corners are at known positions
        detected = {
            0: np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32),
            1: np.array([[700, 0], [800, 0], [800, 100], [700, 100]], dtype=np.float32),
            2: np.array([[700, 700], [800, 700], [800, 800], [700, 800]], dtype=np.float32),
            3: np.array([[0, 700], [100, 700], [100, 800], [0, 800]], dtype=np.float32),
        }
        # Inner corners: marker0[2]=(100,100), marker1[3]=(700,100),
        #                marker2[0]=(700,700), marker3[1]=(100,800)
        dst = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)
        H = compute_homography(detected, dst)
        assert H is not None

        # Verify inner corner of marker 0 (index 2 = (100,100)) maps near dst[0] = (0,0)
        pt = np.array([[[100.0, 100.0]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)
        assert abs(result[0, 0, 0] - 0) < 5
        assert abs(result[0, 0, 1] - 0) < 5

    def test_homography_dtype_float64(self):
        detected = {
            0: np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=np.float32),
            1: np.array([[540, 10], [590, 10], [590, 60], [540, 60]], dtype=np.float32),
            2: np.array([[540, 540], [590, 540], [590, 590], [540, 590]], dtype=np.float32),
            3: np.array([[10, 540], [60, 540], [60, 590], [10, 590]], dtype=np.float32),
        }
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)
        H = compute_homography(detected, dst)
        assert H.dtype == np.float64

    def test_extra_markers_ignored(self):
        """Extra detected markers should not affect the result."""
        detected = {
            0: np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=np.float32),
            1: np.array([[540, 10], [590, 10], [590, 60], [540, 60]], dtype=np.float32),
            2: np.array([[540, 540], [590, 540], [590, 590], [540, 590]], dtype=np.float32),
            3: np.array([[10, 540], [60, 540], [60, 590], [10, 590]], dtype=np.float32),
            99: np.array([[300, 300], [350, 300], [350, 350], [300, 350]], dtype=np.float32),
        }
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)
        H = compute_homography(detected, dst)
        assert H is not None
        assert H.shape == (3, 3)


# ---------------------------------------------------------------------------
# rectify_frame
# ---------------------------------------------------------------------------


class TestRectifyFrameExtended:
    def test_identity_preserves_center_pixel(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[240, 320] = (0, 255, 0)
        H = np.eye(3, dtype=np.float64)
        result = rectify_frame(frame, H, (640, 480))
        assert result[240, 320, 1] == 255

    def test_scaling_homography(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[50, 50] = (255, 0, 0)
        # Scale by 2x
        H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
        result = rectify_frame(frame, H, (200, 200))
        assert result.shape == (200, 200, 3)

    def test_different_output_sizes(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        for w, h in [(100, 100), (1920, 1080), (768, 768)]:
            result = rectify_frame(frame, H, (w, h))
            assert result.shape == (h, w, 3)

    def test_dtype_preserved(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        result = rectify_frame(frame, H, (100, 100))
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# encode_jpeg
# ---------------------------------------------------------------------------


class TestEncodeJpegExtended:
    def test_jpeg_magic_bytes(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        data = encode_jpeg(frame)
        assert data[:2] == b"\xff\xd8"

    def test_different_frame_sizes(self):
        for h, w in [(50, 50), (1080, 1920), (1, 1)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            data = encode_jpeg(frame)
            assert isinstance(data, bytes)
            assert len(data) > 0

    def test_decode_roundtrip(self):
        """Encode then decode should produce same dimensions."""
        frame = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        data = encode_jpeg(frame, quality=100)
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == frame.shape

    def test_quality_parameter(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        sizes = {}
        for q in [10, 50, 95]:
            sizes[q] = len(encode_jpeg(frame, quality=q))
        assert sizes[10] < sizes[95]

    def test_all_black_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = encode_jpeg(frame)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_all_white_frame(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        data = encode_jpeg(frame)
        assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# CameraCapture class — test init only (no real camera)
# ---------------------------------------------------------------------------


class TestCameraCaptureInit:
    def test_default_output_size(self):
        from client.camera import CameraCapture
        cam = CameraCapture(url="http://example.com:8080")
        assert cam.output_size == (768, 768)

    def test_custom_output_size(self):
        from client.camera import CameraCapture
        cam = CameraCapture(url="http://example.com:8080", output_size=(1024, 1024))
        assert cam.output_size == (1024, 1024)

    def test_dst_points_match_output_size(self):
        from client.camera import CameraCapture
        cam = CameraCapture(url="http://example.com:8080", output_size=(500, 400))
        expected = np.array([[0, 0], [500, 0], [500, 400], [0, 400]], dtype=np.float32)
        np.testing.assert_array_equal(cam.dst_points, expected)

    def test_rotate_stored(self):
        from client.camera import CameraCapture
        cam = CameraCapture(url="http://example.com:8080", rotate=180)
        assert cam.rotate == 180

    def test_h_cached_starts_none(self):
        from client.camera import CameraCapture
        cam = CameraCapture(url="http://example.com:8080")
        assert cam.H_cached is None

    def test_no_url_or_webcam_raises(self):
        from client.camera import CameraCapture
        cam = CameraCapture()
        with pytest.raises(ValueError, match="url or webcam"):
            cam.start()
