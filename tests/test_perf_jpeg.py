"""Tests for Perf 1: Eliminate JPEG round-trip in video hot path."""

import numpy as np
import pytest

from client.camera import CameraCapture


class TestCameraReturnsRawFrame:
    def test_get_rectified_frame_returns_3_tuple(self):
        """get_rectified_frame should return (jpeg_bytes, raw_ndarray, H_cam)."""
        # Create a camera with a fake capture source
        cam = CameraCapture.__new__(CameraCapture)
        cam.cap = None
        cam.H_cached = np.eye(3)
        cam.output_size = (640, 480)
        cam.rotate = 0
        cam.stats = {"frames_captured": 0, "frames_with_markers": 0,
                     "frames_using_cache": 0, "consecutive_failures": 0}
        cam._consecutive_failures = 0
        cam.detector = None
        cam.dst_points = None
        cam._last_detection_time = 0

        # Monkey-patch _capture_frame to return a test frame
        test_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cam._capture_frame = lambda: test_frame

        # Monkey-patch detect_markers to return empty (use cached H)
        import client.camera as cam_mod
        original_detect = cam_mod.detect_markers
        original_rectify = cam_mod.rectify_frame
        cam_mod.detect_markers = lambda frame, det: {}
        cam_mod.rectify_frame = lambda frame, H, size: frame

        try:
            result = cam.get_rectified_frame()
            assert len(result) == 3, "Should return 3-tuple (jpeg, raw, H)"
            jpeg_bytes, raw_frame, H = result
            assert isinstance(jpeg_bytes, bytes)
            assert isinstance(raw_frame, np.ndarray)
            assert raw_frame.shape == (480, 640, 3)
            assert H is not None
        finally:
            cam_mod.detect_markers = original_detect
            cam_mod.rectify_frame = original_rectify
