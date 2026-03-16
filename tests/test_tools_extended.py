"""Extended tests for backend/tools.py — show_scene, refresh_view, edge cases."""

import pytest

from backend.tools import overlay, query


# ---------------------------------------------------------------------------
# show_scene (via overlay action="show_scene")
# ---------------------------------------------------------------------------


class TestShowScene:
    def test_returns_showing_scene_status(self):
        result = overlay(action="show_scene", scene_name="Scene 1", placement=[0, 0, 1000, 1000])
        assert result["status"] == "showing_scene"

    def test_returns_scene_name(self):
        result = overlay(action="show_scene", scene_name="My Picture", placement=[100, 200, 500, 700])
        assert result["scene_name"] == "My Picture"

    def test_show_scene_status(self):
        result = overlay(action="show_scene", scene_name="X", placement=[50, 50, 800, 800])
        assert result["status"] == "showing_scene"
        assert result["scene_name"] == "X"

    def test_empty_scene_name(self):
        result = overlay(action="show_scene", scene_name="", placement=[0, 0, 1000, 1000])
        assert result["status"] == "error"  # empty scene_name is now an error


# ---------------------------------------------------------------------------
# refresh_view — additional (via query target="fresh_view")
# ---------------------------------------------------------------------------


class TestRefreshViewExtended:
    def test_empty_reason(self):
        result = query(target="fresh_view", reason="")
        assert result["status"] == "refreshing"
        assert result["reason"] == ""

    def test_long_reason_preserved(self):
        reason = "x" * 500
        result = query(target="fresh_view", reason=reason)
        assert result["reason"] == reason

    def test_description_field_present(self):
        result = query(target="fresh_view", reason="test")
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0


# ---------------------------------------------------------------------------
# overlay create — additional edge cases
# ---------------------------------------------------------------------------


class TestProjectOverlayEdgeCases:
    def test_markdown_type_accepted(self):
        result = overlay(
            action="create",
            content_type="markdown",
            placement=[100, 100, 600, 600],
            title="Steps",
            data={"text": "# Step 1\nDo the thing."},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "markdown"

    def test_image_type_accepted(self):
        result = overlay(
            action="create",
            content_type="image",
            placement=[0, 0, 1000, 1000],
            title="A cat",
            data={"prompt": "a cute cat"},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "image"

    def test_empty_data_dict_accepted(self):
        result = overlay(
            action="create",
            content_type="annotation",
            placement=[100, 100, 500, 500],
            title="Test",
            data={},
        )
        assert result["status"] == "displayed"

    def test_empty_title_accepted(self):
        """Title is not validated by the tool — only placement/type are."""
        result = overlay(
            action="create",
            content_type="annotation",
            placement=[100, 100, 500, 500],
            title="",
            data={"text": "hi"},
        )
        assert result["status"] == "displayed"
        assert result["title"] == ""

    def test_placement_boundary_exact_zero(self):
        result = overlay(
            action="create",
            content_type="annotation",
            placement=[0, 0, 100, 100],
            title="t",
            data={},
        )
        assert result["status"] == "displayed"

    def test_placement_boundary_exact_1000(self):
        result = overlay(
            action="create",
            content_type="annotation",
            placement=[900, 900, 1000, 1000],
            title="t",
            data={},
        )
        assert result["status"] == "displayed"

    def test_placement_all_zero_fails(self):
        result = overlay(
            action="create",
            content_type="annotation",
            placement=[0, 0, 0, 0],
            title="t",
            data={},
        )
        assert result["status"] == "error"

    def test_very_small_region_valid(self):
        result = overlay(
            action="create",
            content_type="annotation",
            placement=[500, 500, 501, 501],
            title="t",
            data={},
        )
        assert result["status"] == "displayed"

    def test_placement_float_values(self):
        result = overlay(
            action="create",
            content_type="annotation",
            placement=[100.5, 200.7, 500.3, 700.1],
            title="t",
            data={},
        )
        assert result["status"] == "displayed"

    def test_error_message_includes_invalid_type(self):
        result = overlay(
            action="create",
            content_type="3d_model",
            placement=[100, 200, 500, 700],
            title="t",
            data={},
        )
        assert result["status"] == "error"
        assert "3d_model" in result["message"]

    def test_error_placement_identifies_bad_index(self):
        result = overlay(
            action="create",
            content_type="graph",
            placement=[100, 200, 500, 1500],
            title="t",
            data={},
        )
        assert result["status"] == "error"
        assert "3" in result["message"]  # index 3

    def test_placement_empty_list_error(self):
        result = overlay(
            action="create",
            content_type="graph",
            placement=[],
            title="t",
            data={},
        )
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# overlay remove
# ---------------------------------------------------------------------------


class TestOverlayRemove:
    def test_remove_returns_status(self):
        result = overlay(action="remove", overlay_name="X")
        assert result["status"] == "removing"
        assert result["overlay_name"] == "X"

    def test_remove_without_name_errors(self):
        result = overlay(action="remove")
        assert result["status"] == "error"
        assert "overlay_name" in result["message"]


# ---------------------------------------------------------------------------
# overlay clear
# ---------------------------------------------------------------------------


class TestOverlayClear:
    def test_clear_returns_status(self):
        result = overlay(action="clear")
        assert result["status"] == "clearing"
