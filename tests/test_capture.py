"""Tests for client/capture.py — CaptureSource ABC, _VideoCaptureSource base,
USBWebcamSource, IPWebcamSource, and CameraCapture integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from client.capture import (
    CaptureSource,
    IPWebcamSource,
    USBWebcamSource,
    _VideoCaptureSource,
)
from tests.camera_helpers import MockCaptureSource, make_blank_image, make_table_image


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestCaptureSourceABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CaptureSource()

    def test_cannot_instantiate_video_capture_source(self):
        """_VideoCaptureSource is still abstract (_create_capture not implemented)."""
        with pytest.raises(TypeError):
            _VideoCaptureSource()

    def test_concrete_subclass_works(self):
        """A minimal concrete implementation satisfies the ABC."""
        class DummySource(CaptureSource):
            def open(self): pass
            def read(self): return None
            def close(self): pass

        src = DummySource()
        src.open()
        assert src.read() is None
        src.close()
        assert src.name == "DummySource"

    def test_name_default_is_class_name(self):
        class MyCustomSource(CaptureSource):
            def open(self): pass
            def read(self): return None
            def close(self): pass

        assert MyCustomSource().name == "MyCustomSource"


# ---------------------------------------------------------------------------
# USBWebcamSource
# ---------------------------------------------------------------------------


class TestUSBWebcamSource:
    def test_init_defaults(self):
        src = USBWebcamSource()
        assert src.index == 0
        assert src.buffer_size == 1

    def test_init_custom_index(self):
        src = USBWebcamSource(index=2, buffer_size=3)
        assert src.index == 2
        assert src.buffer_size == 3

    def test_name(self):
        src = USBWebcamSource(index=1)
        assert src.name == "USBWebcam(1)"

    def test_read_before_open_returns_none(self):
        src = USBWebcamSource(index=99)
        assert src.read() is None

    def test_close_before_open_is_safe(self):
        src = USBWebcamSource(index=99)
        src.close()  # should not raise

    def test_close_is_idempotent(self):
        src = USBWebcamSource(index=99)
        src.close()
        src.close()  # second close should not raise

    def test_open_invalid_index_raises(self):
        src = USBWebcamSource(index=99)
        with pytest.raises(RuntimeError, match="Failed to open capture source"):
            src.open()

    def test_open_sets_cap(self):
        """Verify open() with a valid mock sets _cap."""
        src = USBWebcamSource(index=0)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        with patch("client.capture.cv2.VideoCapture", return_value=mock_cap):
            src.open()
        assert src._cap is mock_cap
        mock_cap.set.assert_called_once_with(cv2.CAP_PROP_BUFFERSIZE, 1)

    def test_double_open_releases_previous(self):
        """Calling open() twice should release the first capture."""
        src = USBWebcamSource(index=0)
        mock_cap1 = MagicMock()
        mock_cap1.isOpened.return_value = True
        mock_cap2 = MagicMock()
        mock_cap2.isOpened.return_value = True

        with patch("client.capture.cv2.VideoCapture", return_value=mock_cap1):
            src.open()
        with patch("client.capture.cv2.VideoCapture", return_value=mock_cap2):
            src.open()

        mock_cap1.release.assert_called_once()
        assert src._cap is mock_cap2

    def test_read_delegates_grab_then_read(self):
        """read() should grab (discard stale) then read."""
        src = USBWebcamSource(index=0)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        with patch("client.capture.cv2.VideoCapture", return_value=mock_cap):
            src.open()

        result = src.read()
        mock_cap.grab.assert_called_once()
        mock_cap.read.assert_called_once()
        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_read_returns_none_on_failure(self):
        """read() returns None when VideoCapture.read() fails."""
        src = USBWebcamSource(index=0)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        with patch("client.capture.cv2.VideoCapture", return_value=mock_cap):
            src.open()

        assert src.read() is None

    def test_close_releases_and_nulls_cap(self):
        src = USBWebcamSource(index=0)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        with patch("client.capture.cv2.VideoCapture", return_value=mock_cap):
            src.open()

        src.close()
        mock_cap.release.assert_called_once()
        assert src._cap is None


# ---------------------------------------------------------------------------
# IPWebcamSource
# ---------------------------------------------------------------------------


class TestIPWebcamSource:
    def test_init_strips_trailing_slash(self):
        src = IPWebcamSource("http://192.168.1.100:8080/")
        assert src.url == "http://192.168.1.100:8080"

    def test_name(self):
        src = IPWebcamSource("http://10.0.0.1:8080")
        assert src.name == "IPWebcam(http://10.0.0.1:8080)"

    def test_read_before_open_returns_none(self):
        src = IPWebcamSource("http://fake:9999")
        assert src.read() is None

    def test_close_before_open_is_safe(self):
        src = IPWebcamSource("http://fake:9999")
        src.close()  # should not raise

    def test_close_is_idempotent(self):
        src = IPWebcamSource("http://fake:9999")
        src.close()
        src.close()

    def test_creates_capture_with_video_suffix(self):
        """_create_capture appends /video to the base URL."""
        src = IPWebcamSource("http://192.168.1.100:8080")
        with patch("client.capture.cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_vc.return_value = mock_cap
            src.open()
        mock_vc.assert_called_once_with("http://192.168.1.100:8080/video")

    def test_double_open_releases_previous(self):
        """Calling open() twice should release the first capture."""
        src = IPWebcamSource("http://fake:8080")
        mock_cap1 = MagicMock()
        mock_cap1.isOpened.return_value = True
        mock_cap2 = MagicMock()
        mock_cap2.isOpened.return_value = True

        with patch("client.capture.cv2.VideoCapture", return_value=mock_cap1):
            src.open()
        with patch("client.capture.cv2.VideoCapture", return_value=mock_cap2):
            src.open()

        mock_cap1.release.assert_called_once()
        assert src._cap is mock_cap2


# ---------------------------------------------------------------------------
# CameraCapture integration with CaptureSource
# ---------------------------------------------------------------------------


class TestCameraCaptureWithSource:
    def test_source_based_capture(self):
        """CameraCapture works with an explicit CaptureSource."""
        from client.camera import CameraCapture

        good = make_table_image()
        src = MockCaptureSource([good] * 4)
        cam = CameraCapture(source=src)
        cam.start()

        jpeg, raw, H = cam.get_rectified_frame()
        assert jpeg is not None
        assert H is not None
        assert cam.stats["frames_captured"] >= 1

        cam.stop()

    def test_source_with_dropped_frames(self):
        """Source returning None should be handled gracefully."""
        from client.camera import CameraCapture

        good = make_table_image()
        src = MockCaptureSource([None, None, good])
        cam = CameraCapture(source=src)
        cam.start()

        jpeg, _, _ = cam.get_rectified_frame()
        assert jpeg is None

        jpeg, _, _ = cam.get_rectified_frame()
        assert jpeg is None

        jpeg, _, H = cam.get_rectified_frame()
        assert jpeg is not None
        assert H is not None

        cam.stop()

    def test_legacy_url_still_creates_source(self):
        """Passing url= should auto-create IPWebcamSource."""
        from client.camera import CameraCapture
        cam = CameraCapture(url="http://example.com:8080")
        assert isinstance(cam.source, IPWebcamSource)
        assert cam.source.url == "http://example.com:8080"

    def test_legacy_webcam_still_creates_source(self):
        """Passing webcam= should auto-create USBWebcamSource."""
        from client.camera import CameraCapture
        cam = CameraCapture(webcam=0)
        assert isinstance(cam.source, USBWebcamSource)
        assert cam.source.index == 0

    def test_explicit_source_wins_over_url(self):
        """Explicit source= takes precedence over url=."""
        from client.camera import CameraCapture
        src = MockCaptureSource([])
        cam = CameraCapture(source=src, url="http://example.com:8080")
        assert cam.source is src

    def test_homography_caching_with_source(self):
        """Homography caching works through the CaptureSource path."""
        from client.camera import CameraCapture

        good = make_table_image()
        blank = make_blank_image()
        src = MockCaptureSource([good, blank])
        cam = CameraCapture(source=src)
        cam.start()

        jpeg1, _, H1 = cam.get_rectified_frame()
        assert jpeg1 is not None
        assert H1 is not None

        jpeg2, _, H2 = cam.get_rectified_frame()
        assert jpeg2 is not None
        np.testing.assert_array_equal(H2, H1)

        cam.stop()

    def test_start_calls_source_open(self):
        """start() should delegate to source.open()."""
        from client.camera import CameraCapture
        src = MockCaptureSource([])
        cam = CameraCapture(source=src)
        cam.start()
        assert src.open_count == 1

    def test_stop_calls_source_close(self):
        """stop() should delegate to source.close()."""
        from client.camera import CameraCapture
        src = MockCaptureSource([])
        cam = CameraCapture(source=src)
        cam.start()
        cam.stop()
        assert src.close_count == 1

    def test_stats_tracked_through_source(self):
        """Stats should work correctly through the CaptureSource path."""
        from client.camera import CameraCapture

        good = make_table_image()
        blank = make_blank_image()
        src = MockCaptureSource([None, good, blank])
        cam = CameraCapture(source=src)
        cam.start()

        cam.get_rectified_frame()  # None — failure
        assert cam.stats["consecutive_failures"] == 1
        assert cam.stats["frames_captured"] == 0

        cam.get_rectified_frame()  # good — markers found
        assert cam.stats["consecutive_failures"] == 0
        assert cam.stats["frames_captured"] == 1
        assert cam.stats["frames_with_markers"] == 1

        cam.get_rectified_frame()  # blank — no markers, use cache
        assert cam.stats["frames_captured"] == 2
        assert cam.stats["frames_using_cache"] == 1

        cam.stop()

    def test_stop_without_start_is_safe(self):
        """stop() on an unstarted CameraCapture should not raise."""
        from client.camera import CameraCapture
        cam = CameraCapture(url="http://fake:8080")
        cam.stop()  # should not raise
