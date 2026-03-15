"""Extended tests for backend/tools.py — show_scene, refresh_view, edge cases."""

import pytest

from backend.tools import project_overlay, refresh_view, show_scene


# ---------------------------------------------------------------------------
# show_scene
# ---------------------------------------------------------------------------


class TestShowScene:
    def test_returns_showing_scene_status(self):
        result = show_scene("Scene 1", [0, 0, 1000, 1000])
        assert result["status"] == "showing_scene"

    def test_returns_scene_name(self):
        result = show_scene("My Picture", [100, 200, 500, 700])
        assert result["scene_name"] == "My Picture"

    def test_returns_placement(self):
        result = show_scene("X", [50, 50, 800, 800])
        assert result["placement"] == [50, 50, 800, 800]

    def test_empty_scene_name(self):
        result = show_scene("", [0, 0, 1000, 1000])
        assert result["status"] == "showing_scene"
        assert result["scene_name"] == ""


# ---------------------------------------------------------------------------
# refresh_view — additional
# ---------------------------------------------------------------------------


class TestRefreshViewExtended:
    def test_empty_reason(self):
        result = refresh_view(reason="")
        assert result["status"] == "refreshing"
        assert result["reason"] == ""

    def test_long_reason_preserved(self):
        reason = "x" * 500
        result = refresh_view(reason=reason)
        assert result["reason"] == reason

    def test_description_field_present(self):
        result = refresh_view(reason="test")
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0


# ---------------------------------------------------------------------------
# project_overlay — additional edge cases
# ---------------------------------------------------------------------------


class TestProjectOverlayEdgeCases:
    def test_markdown_type_accepted(self):
        result = project_overlay(
            content_type="markdown",
            placement=[100, 100, 600, 600],
            title="Steps",
            data={"text": "# Step 1\nDo the thing."},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "markdown"

    def test_image_type_accepted(self):
        result = project_overlay(
            content_type="image",
            placement=[0, 0, 1000, 1000],
            title="A cat",
            data={"prompt": "a cute cat"},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "image"

    def test_empty_data_dict_accepted(self):
        result = project_overlay(
            content_type="diagram",
            placement=[100, 100, 500, 500],
            title="Test",
            data={},
        )
        assert result["status"] == "displayed"

    def test_empty_title_accepted(self):
        """Title is not validated by the tool — only placement/type are."""
        result = project_overlay(
            content_type="annotation",
            placement=[100, 100, 500, 500],
            title="",
            data={"text": "hi"},
        )
        assert result["status"] == "displayed"
        assert result["title"] == ""

    def test_placement_boundary_exact_zero(self):
        result = project_overlay(
            content_type="annotation",
            placement=[0, 0, 100, 100],
            title="t",
            data={},
        )
        assert result["status"] == "displayed"

    def test_placement_boundary_exact_1000(self):
        result = project_overlay(
            content_type="annotation",
            placement=[900, 900, 1000, 1000],
            title="t",
            data={},
        )
        assert result["status"] == "displayed"

    def test_placement_all_zero_fails(self):
        result = project_overlay(
            content_type="annotation",
            placement=[0, 0, 0, 0],
            title="t",
            data={},
        )
        assert result["status"] == "error"

    def test_very_small_region_valid(self):
        result = project_overlay(
            content_type="annotation",
            placement=[500, 500, 501, 501],
            title="t",
            data={},
        )
        assert result["status"] == "displayed"

    def test_placement_float_values(self):
        result = project_overlay(
            content_type="annotation",
            placement=[100.5, 200.7, 500.3, 700.1],
            title="t",
            data={},
        )
        assert result["status"] == "displayed"

    def test_error_message_includes_invalid_type(self):
        result = project_overlay(
            content_type="3d_model",
            placement=[100, 200, 500, 700],
            title="t",
            data={},
        )
        assert result["status"] == "error"
        assert "3d_model" in result["message"]

    def test_error_placement_identifies_bad_index(self):
        result = project_overlay(
            content_type="graph",
            placement=[100, 200, 500, 1500],
            title="t",
            data={},
        )
        assert result["status"] == "error"
        assert "3" in result["message"]  # index 3

    def test_placement_empty_list_error(self):
        result = project_overlay(
            content_type="graph",
            placement=[],
            title="t",
            data={},
        )
        assert result["status"] == "error"
