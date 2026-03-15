"""Tests for backend.tools.project_overlay."""

import pytest

from backend.tools import project_overlay


# --- Happy path ---


class TestProjectOverlayValid:
    def test_graph_returns_success(self):
        result = project_overlay(
            content_type="graph",
            placement=[100.0, 200.0, 500.0, 700.0],
            title="y = x^2",
            data={"expression": "x**2 - 3*x + 2", "x_range": [-5, 5], "y_range": [-5, 10]},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "graph"
        assert result["placement"] == [100.0, 200.0, 500.0, 700.0]
        assert result["title"] == "y = x^2"

    def test_annotation_returns_success(self):
        result = project_overlay(
            content_type="annotation",
            placement=[0.0, 0.0, 200.0, 300.0],
            title="Hint",
            data={"text": "Remember to carry the one!"},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "annotation"

    def test_highlight_returns_success(self):
        result = project_overlay(
            content_type="highlight",
            placement=[50.0, 50.0, 150.0, 150.0],
            title="Look here",
            data={"color": "#00ffff", "target": [50, 50, 150, 150]},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "highlight"

    def test_diagram_returns_success(self):
        result = project_overlay(
            content_type="diagram",
            placement=[200.0, 200.0, 800.0, 800.0],
            title="Number line",
            data={"elements": ["0", "1", "2", "3"]},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "diagram"

    def test_boundary_placement_zero_and_thousand(self):
        result = project_overlay(
            content_type="annotation",
            placement=[0.0, 0.0, 1000.0, 1000.0],
            title="Full table",
            data={"text": "Covers entire surface"},
        )
        assert result["status"] == "displayed"


# --- Invalid content_type ---


class TestProjectOverlayInvalidContentType:
    def test_unknown_type_returns_error(self):
        result = project_overlay(
            content_type="video",
            placement=[100.0, 200.0, 500.0, 700.0],
            title="Bad",
            data={},
        )
        assert result["status"] == "error"
        assert "content_type" in result["message"]


# --- Placement validation ---


class TestProjectOverlayInvalidPlacement:
    def test_out_of_range_above_1000(self):
        result = project_overlay(
            content_type="graph",
            placement=[100.0, 200.0, 1001.0, 700.0],
            title="Too big",
            data={},
        )
        assert result["status"] == "error"
        assert "placement" in result["message"].lower() or "range" in result["message"].lower()

    def test_out_of_range_negative(self):
        result = project_overlay(
            content_type="graph",
            placement=[-1.0, 200.0, 500.0, 700.0],
            title="Negative",
            data={},
        )
        assert result["status"] == "error"

    def test_wrong_number_of_values_too_few(self):
        result = project_overlay(
            content_type="graph",
            placement=[100.0, 200.0, 500.0],
            title="Three values",
            data={},
        )
        assert result["status"] == "error"
        assert "4" in result["message"] or "placement" in result["message"].lower()

    def test_wrong_number_of_values_too_many(self):
        result = project_overlay(
            content_type="graph",
            placement=[100.0, 200.0, 500.0, 700.0, 900.0],
            title="Five values",
            data={},
        )
        assert result["status"] == "error"

    def test_ymin_equals_ymax(self):
        result = project_overlay(
            content_type="graph",
            placement=[300.0, 200.0, 300.0, 700.0],
            title="Zero height",
            data={},
        )
        assert result["status"] == "error"

    def test_ymin_greater_than_ymax(self):
        result = project_overlay(
            content_type="graph",
            placement=[500.0, 200.0, 300.0, 700.0],
            title="Inverted Y",
            data={},
        )
        assert result["status"] == "error"

    def test_xmin_equals_xmax(self):
        result = project_overlay(
            content_type="graph",
            placement=[100.0, 500.0, 300.0, 500.0],
            title="Zero width",
            data={},
        )
        assert result["status"] == "error"

    def test_xmin_greater_than_xmax(self):
        result = project_overlay(
            content_type="graph",
            placement=[100.0, 700.0, 500.0, 200.0],
            title="Inverted X",
            data={},
        )
        assert result["status"] == "error"
