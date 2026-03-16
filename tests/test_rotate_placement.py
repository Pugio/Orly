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
        mgr.handle_tool_result("project_overlay", result)
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
        mgr.handle_tool_result("project_overlay", result)
        assert len(called) == 1
