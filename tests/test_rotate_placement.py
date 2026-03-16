"""Tests for Bug 3: --rotate doesn't apply _unrotate_placement().

Verifies that handle_tool_result calls _unrotate_placement before rendering.
"""

import numpy as np
import pytest
from unittest.mock import patch

from client.overlay_manager import OverlayManager


class TestUnrotatePlacement:
    def test_rotate_0_no_change(self):
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, image_rotate=0)
        placement = [100, 200, 500, 600]
        result = mgr._unrotate_placement(placement)
        assert result == [100, 200, 500, 600]

    def test_rotate_90_unrotates_placement(self):
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, image_rotate=90)
        # CW 90 inverse: [1000-xmax, ymin, 1000-xmin, ymax]
        placement = [100, 200, 500, 600]
        result = mgr._unrotate_placement(placement)
        assert result == [400, 100, 800, 500]

    def test_rotate_180_unrotates(self):
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, image_rotate=180)
        placement = [100, 200, 500, 600]
        result = mgr._unrotate_placement(placement)
        assert result == [500, 400, 900, 800]

    def test_rotate_270_unrotates(self):
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, image_rotate=270)
        # CCW 90 inverse: [xmin, 1000-ymax, xmax, 1000-ymin]
        placement = [100, 200, 500, 600]
        result = mgr._unrotate_placement(placement)
        assert result == [200, 500, 600, 900]


class TestUnrotateImage:
    """Locally-rendered overlays must be un-rotated to match the human's viewing angle."""

    def _make_asymmetric_image(self):
        """Create an image with a recognizable corner so rotation is detectable."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        # Red square in top-left corner only
        img[0:20, 0:20] = [0, 0, 255]
        return img

    def test_rotate_0_no_change(self):
        mgr = OverlayManager(H_proj=None, image_rotate=0)
        img = self._make_asymmetric_image()
        result = mgr._unrotate_image(img)
        assert result.shape == img.shape
        assert np.array_equal(result, img)

    def test_rotate_270_applies_90_cw(self):
        """image_rotate=270 means frame was rotated 270 CW. Undo = rotate 90 CW."""
        mgr = OverlayManager(H_proj=None, image_rotate=270)
        img = self._make_asymmetric_image()  # 100x200, red in top-left
        result = mgr._unrotate_image(img)
        # 90 CW: (100, 200) -> (200, 100), red moves to top-right
        assert result.shape == (200, 100, 3)
        # Top-right corner should be red
        assert result[0, 99, 2] == 255  # red channel
        # Top-left should be black
        assert result[0, 0, 2] == 0

    def test_rotate_90_applies_270_cw(self):
        mgr = OverlayManager(H_proj=None, image_rotate=90)
        img = self._make_asymmetric_image()
        result = mgr._unrotate_image(img)
        assert result.shape == (200, 100, 3)
        # 270 CW (= 90 CCW): red moves to bottom-left
        assert result[199, 0, 2] == 255

    def test_rotate_180_flips(self):
        mgr = OverlayManager(H_proj=None, image_rotate=180)
        img = self._make_asymmetric_image()
        result = mgr._unrotate_image(img)
        assert result.shape == img.shape
        # 180: red moves to bottom-right
        assert result[99, 199, 2] == 255
        assert result[0, 0, 2] == 0


class TestHandleToolResultCallsUnrotate:
    def test_handle_tool_result_applies_unrotate(self):
        """handle_tool_result must call _unrotate_placement when image_rotate != 0."""
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, image_rotate=90)
        called_with = []
        original_unrotate = mgr._unrotate_placement

        def spy_unrotate(placement):
            result = original_unrotate(placement)
            called_with.append((placement, result))
            return result

        mgr._unrotate_placement = spy_unrotate

        result = {
            "content_type": "annotation",
            "placement": [100, 200, 500, 600],
            "title": "test",
            "data": {"text": "hello"},
        }
        mgr.handle_tool_result("overlay", {"action": "create", **result})
        assert len(called_with) == 1, "_unrotate_placement must be called"

    def test_rotate_0_still_calls_unrotate(self):
        """Even with rotate=0, _unrotate_placement should be called (it's a no-op)."""
        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, image_rotate=0)
        called = []
        original = mgr._unrotate_placement
        mgr._unrotate_placement = lambda p: (called.append(True), original(p))[1]

        result = {
            "content_type": "annotation",
            "placement": [100, 200, 500, 600],
            "title": "test",
            "data": {"text": "hello"},
        }
        mgr.handle_tool_result("overlay", {"action": "create", **result})
        assert len(called) == 1
