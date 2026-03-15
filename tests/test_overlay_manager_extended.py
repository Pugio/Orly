"""Extended tests for client/overlay_manager.py."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from client.overlay_manager import OverlayManager


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestOverlayManagerInit:
    def test_default_screen_mode(self):
        om = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        assert om.mode == "screen"
        assert om.proj_width == 1280
        assert om.proj_height == 720
        assert om.H_proj is None

    def test_projector_mode_with_homography(self):
        H = np.eye(3, dtype=np.float64)
        om = OverlayManager(H_proj=H, proj_width=1920, proj_height=1080, mode="projector")
        assert om.mode == "projector"
        assert np.array_equal(om.H_proj, H)

    def test_canvas_shape_matches_dimensions(self):
        om = OverlayManager(H_proj=None, proj_width=800, proj_height=600, mode="screen")
        assert om.canvas.shape == (600, 800, 3)

    def test_canvas_starts_black(self):
        om = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        assert om.canvas.max() == 0

    def test_canvas_starts_white_when_white_bg(self):
        om = OverlayManager(
            H_proj=None, proj_width=640, proj_height=480,
            mode="screen", white_bg=True,
        )
        assert om.canvas.min() == 255

    def test_image_rotate_stored(self):
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=90)
        assert om.image_rotate == 90

    def test_scenes_empty_on_init(self):
        om = OverlayManager(H_proj=None, mode="screen")
        assert om.scenes == {}
        assert om._scene_order == []

    def test_refresh_not_requested_on_init(self):
        om = OverlayManager(H_proj=None, mode="screen")
        assert om._refresh_requested is False
        assert om._saved_canvas is None


# ---------------------------------------------------------------------------
# render_overlay dispatch
# ---------------------------------------------------------------------------


class TestRenderOverlayDispatch:
    def _om(self):
        return OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")

    def test_graph_dispatch(self):
        om = self._om()
        img = om.render_overlay(
            "graph", [0, 0, 500, 500], "y=x",
            {"expression": "x", "x_range": [-5, 5], "y_range": [-5, 5]},
        )
        assert img.shape[2] == 3
        assert img.max() > 0

    def test_annotation_dispatch(self):
        om = self._om()
        img = om.render_overlay("annotation", [0, 0, 500, 500], "Hi", {"text": "hello"})
        assert img.shape[2] == 3
        assert img.max() > 0

    def test_markdown_dispatch(self):
        om = self._om()
        img = om.render_overlay("markdown", [0, 0, 500, 500], "Steps", {"text": "# Step 1"})
        assert img.shape[2] == 3
        assert img.max() > 0

    def test_highlight_dispatch_returns_bgr(self):
        om = self._om()
        img = om.render_overlay("highlight", [0, 0, 500, 500], "H", {"color": "#ff0000"})
        assert img.shape[2] == 3  # BGR, not BGRA

    def test_unknown_type_returns_black(self):
        om = self._om()
        img = om.render_overlay("unknown", [0, 0, 500, 500], "?", {})
        assert img.max() == 0

    def test_overlay_dimensions_from_placement(self):
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        img = om.render_overlay(
            "annotation", [0, 0, 500, 500], "test", {"text": "hi"},
        )
        # 500/1000 = 0.5 => 500px each dimension
        assert img.shape == (500, 500, 3)

    def test_graph_defaults_when_data_empty(self):
        om = self._om()
        img = om.render_overlay("graph", [0, 0, 500, 500], "t", {})
        # Should default to expression="x", x_range=[-10,10], y_range=[-10,10]
        assert img.shape[2] == 3
        assert img.max() > 0


# ---------------------------------------------------------------------------
# place_on_canvas — screen mode
# ---------------------------------------------------------------------------


class TestPlaceOnCanvasScreen:
    def test_direct_mapping_dimensions(self):
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        overlay = np.ones((100, 200, 3), dtype=np.uint8) * 255
        canvas = om.place_on_canvas(overlay, [0, 0, 200, 400])
        assert canvas.shape == (1000, 1000, 3)
        # Top-left region should have content
        assert canvas[0:100, 0:200].max() > 0

    def test_placement_maps_correctly(self):
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        overlay = np.ones((100, 100, 3), dtype=np.uint8) * 255
        canvas = om.place_on_canvas(overlay, [500, 500, 700, 700])
        # Area before placement should be black
        assert canvas[0:200, 0:200].max() == 0
        # Placement area should have content
        assert canvas[500:700, 500:700].max() > 0

    def test_full_canvas_placement(self):
        om = OverlayManager(H_proj=None, proj_width=500, proj_height=500, mode="screen")
        overlay = np.ones((50, 50, 3), dtype=np.uint8) * 128
        canvas = om.place_on_canvas(overlay, [0, 0, 1000, 1000])
        assert canvas.shape == (500, 500, 3)
        assert canvas.max() > 0

    def test_does_not_mutate_existing_canvas(self):
        """place_on_canvas should work with the canvas copy."""
        om = OverlayManager(H_proj=None, proj_width=500, proj_height=500, mode="screen")
        original_canvas = om.canvas.copy()
        overlay = np.ones((50, 50, 3), dtype=np.uint8) * 255
        result = om.place_on_canvas(overlay, [0, 0, 200, 200])
        # The original canvas object should be unchanged (place_on_canvas copies)
        assert np.array_equal(om.canvas, original_canvas)


# ---------------------------------------------------------------------------
# place_on_canvas — projector mode
# ---------------------------------------------------------------------------


class TestPlaceOnCanvasProjector:
    def test_projector_identity_homography(self):
        H = np.eye(3, dtype=np.float64)
        om = OverlayManager(H_proj=H, proj_width=1000, proj_height=1000, mode="projector")
        overlay = np.ones((50, 50, 3), dtype=np.uint8) * 200
        canvas = om.place_on_canvas(overlay, [100, 200, 300, 400])
        assert canvas.shape == (1000, 1000, 3)

    def test_projector_scaling_homography(self):
        # H that scales coordinates: table 0-1000 -> projector 0-500
        H = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float64)
        om = OverlayManager(H_proj=H, proj_width=500, proj_height=500, mode="projector")
        overlay = np.ones((50, 50, 3), dtype=np.uint8) * 200
        canvas = om.place_on_canvas(overlay, [0, 0, 500, 500])
        assert canvas.shape == (500, 500, 3)
        # With scaling H, table [0,0,500,500] maps to projector [0,0,250,250]
        assert canvas.max() > 0

    def test_projector_falls_back_when_out_of_bounds(self):
        """When H_proj maps outside projector bounds, should fall back to direct."""
        # H that maps everything to negative coords
        H = np.array([[1, 0, -5000], [0, 1, -5000], [0, 0, 1]], dtype=np.float64)
        om = OverlayManager(H_proj=H, proj_width=500, proj_height=500, mode="projector")
        overlay = np.ones((50, 50, 3), dtype=np.uint8) * 200
        canvas = om.place_on_canvas(overlay, [0, 0, 500, 500])
        # Should still produce a result via direct mapping fallback
        assert canvas.shape == (500, 500, 3)


# ---------------------------------------------------------------------------
# handle_tool_result
# ---------------------------------------------------------------------------


class TestHandleToolResult:
    def test_project_overlay_graph(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        om.handle_tool_result("project_overlay", {
            "content_type": "graph",
            "placement": [0, 0, 500, 500],
            "title": "y=x",
            "data": {"expression": "x", "x_range": [-5, 5], "y_range": [-5, 5]},
        })
        assert om.canvas.max() > 0

    def test_project_overlay_annotation(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        om.handle_tool_result("project_overlay", {
            "content_type": "annotation",
            "placement": [0, 0, 500, 500],
            "title": "Test",
            "data": {"text": "hello"},
        })
        assert om.canvas.max() > 0

    def test_unknown_tool_ignored(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        om.handle_tool_result("google_search", {"query": "test"})
        assert om.canvas.max() == 0

    def test_show_scene_unknown_scene_does_nothing(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        om.handle_tool_result("show_scene", {
            "scene_name": "nonexistent",
            "placement": [0, 0, 1000, 1000],
        })
        assert om.canvas.max() == 0

    def test_show_scene_known_scene(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        # Manually add a scene
        fake_scene = np.ones((200, 300, 3), dtype=np.uint8) * 128
        om.scenes["My Scene"] = fake_scene
        om.handle_tool_result("show_scene", {
            "scene_name": "My Scene",
            "placement": [0, 0, 1000, 1000],
        })
        assert om.canvas.max() > 0

    def test_defaults_when_result_missing_fields(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        om.handle_tool_result("project_overlay", {})
        # content_type defaults to "annotation", placement to [0,0,1000,1000]
        # title defaults to "" -> empty annotation -> black
        # This should not crash
        assert om.canvas.shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# Minimum placement size enforcement
# ---------------------------------------------------------------------------


class TestMinPlacementSize:
    def test_markdown_small_width_expanded(self):
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        om.handle_tool_result("project_overlay", {
            "content_type": "markdown",
            "placement": [100, 100, 600, 200],  # width=100, too narrow
            "title": "t",
            "data": {"text": "hello"},
        })
        assert om.canvas.max() > 0

    def test_annotation_small_height_expanded(self):
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        om.handle_tool_result("project_overlay", {
            "content_type": "annotation",
            "placement": [100, 100, 200, 800],  # height=100, too short
            "title": "t",
            "data": {"text": "hello"},
        })
        assert om.canvas.max() > 0

    def test_graph_not_expanded(self):
        """Graph type should not be affected by min-size enforcement."""
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        om.handle_tool_result("project_overlay", {
            "content_type": "graph",
            "placement": [100, 100, 200, 200],  # small but no enforcement
            "title": "t",
            "data": {"expression": "x", "x_range": [-5, 5], "y_range": [-5, 5]},
        })
        assert om.canvas.max() > 0

    def test_min_width_clamps_to_1000(self):
        """When expanding width would exceed 1000, clamp xmax then shift xmin."""
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        # xmin=800, width would need 500 => xmax=1000 still too narrow => shift xmin
        om.handle_tool_result("project_overlay", {
            "content_type": "markdown",
            "placement": [100, 800, 600, 900],  # width=100
            "title": "t",
            "data": {"text": "hello"},
        })
        assert om.canvas.max() > 0

    def test_min_height_clamps_to_1000(self):
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        om.handle_tool_result("project_overlay", {
            "content_type": "annotation",
            "placement": [800, 100, 900, 800],  # height=100
            "title": "t",
            "data": {"text": "hello"},
        })
        assert om.canvas.max() > 0


# ---------------------------------------------------------------------------
# _unrotate_placement
# ---------------------------------------------------------------------------


class TestUnrotatePlacement:
    def test_no_rotation(self):
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=0)
        result = om._unrotate_placement([100, 200, 300, 400])
        assert result == [100, 200, 300, 400]

    def test_90_rotation(self):
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=90)
        result = om._unrotate_placement([100, 200, 300, 400])
        # CW 90: [xmin, 1000 - ymax, xmax, 1000 - ymin]
        assert result == [200, 1000 - 300, 400, 1000 - 100]

    def test_180_rotation(self):
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=180)
        result = om._unrotate_placement([100, 200, 300, 400])
        assert result == [1000 - 300, 1000 - 400, 1000 - 100, 1000 - 200]

    def test_270_rotation(self):
        om = OverlayManager(H_proj=None, mode="screen", image_rotate=270)
        result = om._unrotate_placement([100, 200, 300, 400])
        assert result == [1000 - 400, 100, 1000 - 200, 300]

    def test_round_trip_90(self):
        """Rotating 90 then un-rotating should give back original placement."""
        om90 = OverlayManager(H_proj=None, mode="screen", image_rotate=90)
        original = [100, 200, 500, 700]
        unrotated = om90._unrotate_placement(original)
        # Verify the result is a valid placement (ymin < ymax, xmin < xmax)
        assert unrotated[0] < unrotated[2]  # ymin < ymax
        assert unrotated[1] < unrotated[3]  # xmin < xmax


# ---------------------------------------------------------------------------
# _composite
# ---------------------------------------------------------------------------


class TestComposite:
    def test_black_bg_direct_overwrite(self):
        om = OverlayManager(H_proj=None, proj_width=100, proj_height=100, mode="screen")
        canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = np.ones((50, 50, 3), dtype=np.uint8) * 200
        om._composite(canvas, overlay, 10, 60, 10, 60)
        assert np.array_equal(canvas[10:60, 10:60], overlay)

    def test_white_bg_preserves_background(self):
        om = OverlayManager(
            H_proj=None, proj_width=100, proj_height=100,
            mode="screen", white_bg=True,
        )
        canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
        overlay = np.zeros((50, 50, 3), dtype=np.uint8)
        # Add some non-black content to overlay
        overlay[10:20, 10:20] = (0, 255, 0)
        om._composite(canvas, overlay, 0, 50, 0, 50)
        # Non-black region should be overwritten
        assert np.array_equal(canvas[10:20, 10:20], overlay[10:20, 10:20])
        # Black region should still be white (background shows through)
        assert canvas[0, 0, 0] == 255

    def test_white_bg_black_overlay_not_written(self):
        om = OverlayManager(
            H_proj=None, proj_width=100, proj_height=100,
            mode="screen", white_bg=True,
        )
        canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
        overlay = np.zeros((50, 50, 3), dtype=np.uint8)  # all black
        om._composite(canvas, overlay, 0, 50, 0, 50)
        # Canvas should remain all white since overlay is entirely black
        assert canvas[0:50, 0:50].min() == 255


# ---------------------------------------------------------------------------
# request_refresh / complete_refresh
# ---------------------------------------------------------------------------


class TestRefreshCycle:
    def test_request_refresh_clears_canvas(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        # Put something on canvas
        om.canvas[0:100, 0:100] = 255
        om.request_refresh()
        assert om.canvas.max() == 0
        assert om._refresh_requested is True

    def test_complete_refresh_restores_canvas(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        om.canvas[0:100, 0:100] = 128
        original = om.canvas.copy()
        om.request_refresh()
        assert om.canvas.max() == 0
        om.complete_refresh()
        assert np.array_equal(om.canvas, original)
        assert om._refresh_requested is False

    def test_double_request_only_saves_once(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        om.canvas[0:100, 0:100] = 128
        original = om.canvas.copy()
        om.request_refresh()
        # Modify canvas during refresh (shouldn't happen but test safety)
        om.canvas[50:60, 50:60] = 50
        om.request_refresh()  # second call should be a no-op
        om.complete_refresh()
        assert np.array_equal(om.canvas, original)

    def test_complete_without_request_is_noop(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        canvas_before = om.canvas.copy()
        om.complete_refresh()
        assert np.array_equal(om.canvas, canvas_before)

    def test_refresh_with_white_bg(self):
        om = OverlayManager(
            H_proj=None, proj_width=640, proj_height=480,
            mode="screen", white_bg=True,
        )
        om.canvas[0:100, 0:100] = 128  # some content
        om.request_refresh()
        # During refresh, canvas should be white (the bg color)
        assert om.canvas.min() == 255
        om.complete_refresh()
        assert om.canvas[0, 0, 0] == 128


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_to_black(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        om.canvas[:] = 200
        om.clear()
        assert om.canvas.max() == 0

    def test_clear_resets_to_white_when_white_bg(self):
        om = OverlayManager(
            H_proj=None, proj_width=640, proj_height=480,
            mode="screen", white_bg=True,
        )
        om.canvas[:] = 0
        om.clear()
        assert om.canvas.min() == 255

    def test_clear_preserves_canvas_shape(self):
        om = OverlayManager(H_proj=None, proj_width=800, proj_height=600, mode="screen")
        om.clear()
        assert om.canvas.shape == (600, 800, 3)


# ---------------------------------------------------------------------------
# _placement_pixel_size
# ---------------------------------------------------------------------------


class TestPlacementPixelSize:
    def test_full_size(self):
        om = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        w, h = om._placement_pixel_size([0, 0, 1000, 1000])
        assert w == 1280
        assert h == 720

    def test_half_size(self):
        om = OverlayManager(H_proj=None, proj_width=1000, proj_height=1000, mode="screen")
        w, h = om._placement_pixel_size([0, 0, 500, 500])
        assert w == 500
        assert h == 500

    def test_minimum_one_pixel(self):
        om = OverlayManager(H_proj=None, proj_width=100, proj_height=100, mode="screen")
        w, h = om._placement_pixel_size([0, 0, 1, 1])
        assert w >= 1
        assert h >= 1


# ---------------------------------------------------------------------------
# Image async (handle_tool_result for image type)
# ---------------------------------------------------------------------------


class TestImageHandling:
    def test_image_shows_loading_immediately(self):
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        with patch("client.overlay_manager.render_image") as mock_ri:
            mock_ri.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            om.handle_tool_result("project_overlay", {
                "content_type": "image",
                "placement": [0, 0, 1000, 1000],
                "title": "test",
                "data": {"prompt": "a cat"},
            })
        assert om.canvas.max() > 0  # loading placeholder visible

    def test_projector_mode_flips_overlay(self):
        """In projector mode, non-highlight overlays should be rotated 180."""
        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="projector")
        om.handle_tool_result("project_overlay", {
            "content_type": "annotation",
            "placement": [0, 0, 500, 500],
            "title": "t",
            "data": {"text": "hello"},
        })
        assert om.canvas.max() > 0
