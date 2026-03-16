"""Tests for the central CoordinateTransform module.

CoordinateTransform is the SINGLE authority for all spatial operations.
These tests verify each transformation in isolation and in combination.
"""

import cv2
import numpy as np
import pytest

from client.coordinate_transform import CoordinateTransform
from calibration.projector_calibrate import compute_projector_homography


def _identity_H(proj_w, proj_h):
    """H_proj that maps table (x,y) 0-1000 → projector pixels linearly."""
    table = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    proj = [(0, 0), (proj_w, 0), (proj_w, proj_h), (0, proj_h)]
    return compute_projector_homography(table, proj)


def _inverted_H(proj_w, proj_h):
    """H_proj for a projector mounted 180° rotated (across the table)."""
    table = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    proj = [(proj_w, proj_h), (0, proj_h), (0, 0), (proj_w, 0)]
    return compute_projector_homography(table, proj)


class TestGeminiToTable:
    """Verify coordinate un-rotation for all rotation values."""

    def test_rotate_0_identity(self):
        ct = CoordinateTransform(rotate=0)
        assert ct.gemini_to_table([100, 200, 500, 600]) == [100, 200, 500, 600]

    @pytest.mark.parametrize("rotate", [90, 180, 270])
    def test_roundtrip(self, rotate):
        """Forward rotate then gemini_to_table should return original."""
        ct = CoordinateTransform(rotate=rotate)
        original = [150, 250, 700, 850]
        ymin, xmin, ymax, xmax = original

        if rotate == 90:
            rotated = [xmin, 1000 - ymax, xmax, 1000 - ymin]
        elif rotate == 180:
            rotated = [1000 - ymax, 1000 - xmax, 1000 - ymin, 1000 - xmin]
        elif rotate == 270:
            rotated = [1000 - xmax, ymin, 1000 - xmin, ymax]

        assert ct.gemini_to_table(rotated) == original


class TestOrientOverlay:
    """Verify overlay rotation for all rotation values."""

    def _make_asymmetric(self):
        """100x200 image with red pixel at (0,0)."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[0, 0] = [0, 0, 255]
        return img

    def test_rotate_0_no_change(self):
        ct = CoordinateTransform(rotate=0)
        img = self._make_asymmetric()
        result = ct.orient_overlay(img)
        assert np.array_equal(result, img)

    def test_rotate_270_applies_90_cw(self):
        ct = CoordinateTransform(rotate=270)
        img = self._make_asymmetric()
        result = ct.orient_overlay(img)
        assert result.shape == (200, 100, 3)
        # 90 CW: pixel (0,0) → (0, h-1) = (0, 99) in the rotated image
        assert result[0, 99, 2] == 255

    def test_rotate_180_flips(self):
        ct = CoordinateTransform(rotate=180)
        img = self._make_asymmetric()
        result = ct.orient_overlay(img)
        assert result.shape == img.shape
        assert result[99, 199, 2] == 255


class TestPlaceOnCanvas:
    """Verify canvas placement in both screen and projector modes."""

    def test_screen_mode_basic(self):
        ct = CoordinateTransform(mode="screen", display_width=1000, display_height=1000)
        canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        result = ct.place_on_canvas(canvas, overlay, [0, 0, 500, 500])
        assert result[:500, :500].max() > 0
        assert result[500:, 500:].max() == 0

    def test_projector_mode_identity_H(self):
        H = _identity_H(1280, 720)
        ct = CoordinateTransform(mode="projector", H_proj=H,
                                 display_width=1280, display_height=720)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        result = ct.place_on_canvas(canvas, overlay, [400, 400, 600, 600])
        # Center should have content
        assert result[360, 640].sum() > 0

    def test_projector_mode_inverted_H(self):
        """With 180° inverted projector, content at table TL → projector BR."""
        H = _inverted_H(1280, 720)
        ct = CoordinateTransform(mode="projector", H_proj=H,
                                 display_width=1280, display_height=720)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        overlay = np.full((50, 50, 3), 200, dtype=np.uint8)
        result = ct.place_on_canvas(canvas, overlay, [0, 0, 200, 200])
        # With inverted H, TL table → BR projector
        br = result[540:, 960:]
        tl = result[:180, :256]
        assert br.sum() > 0, "Inverted: table TL should map to projector BR"
        assert tl.sum() == 0, "Inverted: projector TL should be empty"

    def test_white_bg_compositing(self):
        ct = CoordinateTransform(mode="screen", display_width=100, display_height=100)
        canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
        overlay = np.zeros((50, 50, 3), dtype=np.uint8)
        overlay[10:20, 10:20] = (0, 255, 0)
        result = ct.place_on_canvas(canvas, overlay, [0, 0, 500, 500], white_bg=True)
        # White should be preserved where overlay is black
        assert result[0, 0, 0] == 255
        # Green should be written
        assert result[:50, :50, 1].max() > 200


class TestPlaceOnCanvasNoFlip:
    """place_on_canvas does NOT flip — it only does spatial mapping.

    The 180° projector flip lives in orient_overlay (called by _show_overlay),
    which runs BEFORE place_on_canvas. These tests verify that place_on_canvas
    itself applies only the H_proj warp, no additional rotation.
    """

    def test_identity_H_preserves_orientation(self):
        """With identity H_proj, overlay TL stays at canvas TL (no warp effect)."""
        H = _identity_H(640, 480)
        ct = CoordinateTransform(mode="projector", H_proj=H,
                                 display_width=640, display_height=480)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[0:30, 0:30] = [0, 255, 255]  # yellow TL

        result = ct.place_on_canvas(canvas, overlay, [0, 0, 1000, 1000])
        tl = result[:120, :160].sum()
        br = result[360:, 480:].sum()
        assert tl > br, "Identity H: no flip, yellow stays in TL"

    def test_inverted_H_flips_via_warp(self):
        """With inverted H_proj, the WARP (not a manual flip) moves content."""
        H = _inverted_H(640, 480)
        ct = CoordinateTransform(mode="projector", H_proj=H,
                                 display_width=640, display_height=480)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay[0:30, 0:30] = [0, 255, 255]  # yellow TL

        result = ct.place_on_canvas(canvas, overlay, [0, 0, 1000, 1000])
        tl = result[:120, :160].sum()
        br = result[360:, 480:].sum()
        assert br > tl, "Inverted H: warp moves yellow to BR (via H_proj, not manual flip)"


class TestShowSceneUnrotatesPlacement:
    """show_scene receives placement in Gemini space (rotated) and must unrotate."""

    def test_show_scene_unrotates_placement(self):
        """show_scene should call _unrotate_placement on the placement."""
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000,
                              mode="screen", image_rotate=270)
        fake_scene = np.full((100, 100, 3), 200, dtype=np.uint8)
        mgr.scenes["test"] = fake_scene

        unrotate_calls = []
        original = mgr._unrotate_placement

        def spy(placement):
            result = original(placement)
            unrotate_calls.append((placement, result))
            return result

        mgr._unrotate_placement = spy
        mgr.handle_tool_result("overlay", {
            "action": "show_scene",
            "scene_name": "test",
            "placement": [0, 0, 500, 500],
        })
        assert len(unrotate_calls) == 1, "show_scene must unrotate placement"
        # With rotate=270, [0, 0, 500, 500] → [0, 500, 500, 1000]
        assert unrotate_calls[0][1] == [0, 500, 500, 1000]


class TestOrientContractEnforced:
    """Verify that _show_overlay automatically orients content.

    The key enforcement: _show_overlay calls orient_overlay internally,
    so callers can't forget. We verify by spying on transform.orient_overlay.
    """

    def test_show_overlay_calls_orient(self):
        """_show_overlay must call transform.orient_overlay internally."""
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=200, proj_height=200,
                              mode="screen", image_rotate=270)
        orient_calls = []
        original = mgr.transform.orient_overlay

        def spy(overlay):
            orient_calls.append(True)
            return original(overlay)

        mgr.transform.orient_overlay = spy
        overlay = np.zeros((50, 50, 3), dtype=np.uint8)
        mgr._show_overlay(overlay, [0, 0, 500, 500])
        assert len(orient_calls) == 1, "_show_overlay must call orient_overlay"

    def test_show_preoriented_skips_orient(self):
        """_show_preoriented must NOT call orient_overlay."""
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=200, proj_height=200,
                              mode="screen", image_rotate=270)
        orient_calls = []
        original = mgr.transform.orient_overlay

        def spy(overlay):
            orient_calls.append(True)
            return original(overlay)

        mgr.transform.orient_overlay = spy
        overlay = np.zeros((50, 50, 3), dtype=np.uint8)
        mgr._show_preoriented(overlay, [0, 0, 500, 500])
        assert len(orient_calls) == 0, "_show_preoriented must skip orient"

    def test_handle_tool_result_orients_via_show_overlay(self):
        """handle_tool_result → _show_overlay → orient is called."""
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=200, proj_height=200,
                              mode="screen", image_rotate=270)
        orient_calls = []
        original = mgr.transform.orient_overlay

        def spy(overlay):
            orient_calls.append(True)
            return original(overlay)

        mgr.transform.orient_overlay = spy
        mgr.handle_tool_result("overlay", {
            "action": "create",
            "content_type": "annotation",
            "placement": [0, 0, 1000, 1000],
            "title": "t",
            "data": {"text": "x"},
        })
        assert len(orient_calls) == 1, "orient must be called via _show_overlay"

    def test_session_restore_orients_non_images(self):
        """session restore calls _show_overlay (which orients) for non-images."""
        from client.session_restore import restore_session_state
        from unittest.mock import MagicMock

        mgr = MagicMock()
        mgr.scenes = {}
        store = MagicMock()
        store.load_overlay_state.return_value = {
            "overlays": [{
                "name": "t", "content_type": "annotation",
                "placement": [0, 0, 500, 500], "title": "t",
                "data": {"text": "x"},
            }],
        }
        store.load_scene_order.return_value = []

        restore_session_state(store, mgr)
        assert mgr.render_overlay.call_count == 1
        # Non-images go through _show_overlay (which orients internally)
        assert mgr._show_overlay.call_count == 1
        # Images go through _show_preoriented (already oriented)
        assert mgr._show_preoriented.call_count == 0

    def test_session_restore_uses_preoriented_for_images(self):
        """session restore calls _show_preoriented for images."""
        from client.session_restore import restore_session_state
        from unittest.mock import MagicMock

        mgr = MagicMock()
        mgr.scenes = {}
        store = MagicMock()
        store.load_image.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        store.load_overlay_state.return_value = {
            "overlays": [{
                "name": "pic", "content_type": "image",
                "placement": [0, 0, 500, 500], "title": "pic",
                "data": {},
            }],
        }
        store.load_scene_order.return_value = []

        restore_session_state(store, mgr)
        # Images use _show_preoriented (already oriented from disk)
        assert mgr._show_preoriented.call_count == 1
        assert mgr._show_overlay.call_count == 0

    def test_program_runtime_orients_for_storage(self):
        """TableAPI.place_overlay orients via transform.orient_overlay for storage."""
        from client.overlay_state import OverlayStateManager
        from client.program_runtime import TableAPI
        from unittest.mock import MagicMock
        from client.coordinate_transform import CoordinateTransform

        om = MagicMock()
        om.transform = CoordinateTransform(rotate=270)
        om.render_overlay.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        om.place_on_canvas.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        om._make_bg.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        om._has_content = False
        om.canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        om.mode = "screen"

        osm = OverlayStateManager(om)
        api = TableAPI(
            overlay_state_manager=osm,
            object_tracker=MagicMock(),
            session_store=MagicMock(),
            notify_fn=lambda msg: None,
            get_frame_fn=lambda: None,
        )
        api.place_overlay("test", "annotation", [0, 0, 500, 500], {"text": "x"})
        assert om.render_overlay.call_count == 1
        # Stored image should be rotated (50x50 → 50x50 for 90° CW)
        entry = osm.get("test")
        assert entry is not None
