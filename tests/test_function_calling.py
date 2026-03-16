"""Tests for PoC 5 — Gemini function calling for overlay generation."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock classes for Gemini API responses
# ---------------------------------------------------------------------------


class MockFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class MockPart:
    def __init__(self, function_call=None, text=None):
        self.function_call = function_call
        self.text = text


class MockCandidate:
    def __init__(self, parts):
        self.content = type("Content", (), {"parts": parts})()


class MockResponse:
    def __init__(self, candidates):
        self.candidates = candidates


# ---------------------------------------------------------------------------
# Imports from the module under test
# ---------------------------------------------------------------------------

from poc.poc5_function_calling import (
    get_system_prompt,
    get_tool_declaration,
    parse_tool_calls,
    render_overlay_preview,
    validate_overlay_call,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_graph_call():
    return {
        "name": "project_overlay",
        "args": {
            "content_type": "graph",
            "placement": [100, 200, 500, 700],
            "title": "y = 2x + 1",
            "data": {"expression": "2*x + 1", "x_range": [-5, 5], "y_range": [-10, 10]},
        },
    }


def _valid_annotation_call():
    return {
        "name": "project_overlay",
        "args": {
            "content_type": "annotation",
            "placement": [50, 50, 300, 400],
            "title": "Hint",
            "data": {"text": "Remember to isolate x first."},
        },
    }


def _valid_highlight_call():
    return {
        "name": "project_overlay",
        "args": {
            "content_type": "highlight",
            "placement": [200, 300, 400, 600],
            "title": "Focus here",
            "data": {"color": "#00ffff"},
        },
    }


# ---------------------------------------------------------------------------
# get_tool_declaration
# ---------------------------------------------------------------------------


class TestGetToolDeclaration:
    def test_returns_dict_with_name(self):
        decl = get_tool_declaration()
        assert isinstance(decl, dict)
        assert decl["name"] == "project_overlay"

    def test_has_parameters_with_required_fields(self):
        decl = get_tool_declaration()
        params = decl["parameters"]
        assert "properties" in params
        assert "required" in params

    def test_properties_include_all_fields(self):
        decl = get_tool_declaration()
        props = decl["parameters"]["properties"]
        for field in ("content_type", "placement", "title", "data"):
            assert field in props, f"Missing property: {field}"


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------


class TestGetSystemPrompt:
    def test_returns_nonempty_string(self):
        prompt = get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_contains_lumi(self):
        prompt = get_system_prompt()
        assert "Orly" in prompt


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_extracts_single_function_call(self):
        fc = MockFunctionCall("project_overlay", {"content_type": "graph", "placement": [0, 0, 500, 500], "title": "Test", "data": {}})
        part = MockPart(function_call=fc)
        candidate = MockCandidate(parts=[part])
        response = MockResponse(candidates=[candidate])

        calls = parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "project_overlay"
        assert calls[0]["args"]["content_type"] == "graph"

    def test_extracts_multiple_function_calls(self):
        fc1 = MockFunctionCall("project_overlay", {"content_type": "graph", "placement": [0, 0, 500, 500], "title": "A", "data": {}})
        fc2 = MockFunctionCall("project_overlay", {"content_type": "annotation", "placement": [500, 0, 1000, 500], "title": "B", "data": {"text": "hi"}})
        parts = [MockPart(function_call=fc1), MockPart(function_call=fc2)]
        candidate = MockCandidate(parts=parts)
        response = MockResponse(candidates=[candidate])

        calls = parse_tool_calls(response)
        assert len(calls) == 2

    def test_no_tool_calls_returns_empty(self):
        candidate = MockCandidate(parts=[])
        response = MockResponse(candidates=[candidate])
        assert parse_tool_calls(response) == []

    def test_text_only_returns_empty(self):
        part = MockPart(text="Here is some explanation.")
        candidate = MockCandidate(parts=[part])
        response = MockResponse(candidates=[candidate])
        assert parse_tool_calls(response) == []


# ---------------------------------------------------------------------------
# validate_overlay_call
# ---------------------------------------------------------------------------


class TestValidateOverlayCall:
    def test_valid_graph_no_errors(self):
        errors = validate_overlay_call(_valid_graph_call())
        assert errors == []

    def test_valid_annotation_no_errors(self):
        errors = validate_overlay_call(_valid_annotation_call())
        assert errors == []

    def test_valid_highlight_no_errors(self):
        errors = validate_overlay_call(_valid_highlight_call())
        assert errors == []

    def test_invalid_content_type(self):
        call = _valid_graph_call()
        call["args"]["content_type"] = "pie_chart"
        errors = validate_overlay_call(call)
        assert len(errors) > 0
        assert any("content_type" in e for e in errors)

    def test_placement_out_of_range(self):
        call = _valid_graph_call()
        call["args"]["placement"] = [0, 0, 1500, 500]
        errors = validate_overlay_call(call)
        assert len(errors) > 0
        assert any("placement" in e.lower() or "range" in e.lower() for e in errors)

    def test_placement_ymin_not_less_than_ymax(self):
        call = _valid_graph_call()
        call["args"]["placement"] = [500, 200, 100, 700]  # ymin > ymax
        errors = validate_overlay_call(call)
        assert len(errors) > 0

    def test_placement_xmin_not_less_than_xmax(self):
        call = _valid_graph_call()
        call["args"]["placement"] = [100, 700, 500, 200]  # xmin > xmax
        errors = validate_overlay_call(call)
        assert len(errors) > 0

    def test_placement_wrong_length(self):
        call = _valid_graph_call()
        call["args"]["placement"] = [100, 200, 500]
        errors = validate_overlay_call(call)
        assert len(errors) > 0

    def test_missing_title(self):
        call = _valid_graph_call()
        call["args"]["title"] = ""
        errors = validate_overlay_call(call)
        assert len(errors) > 0
        assert any("title" in e for e in errors)

    def test_missing_title_key(self):
        call = _valid_graph_call()
        del call["args"]["title"]
        errors = validate_overlay_call(call)
        assert len(errors) > 0

    def test_graph_missing_expression(self):
        call = _valid_graph_call()
        call["args"]["data"] = {"x_range": [-5, 5]}
        errors = validate_overlay_call(call)
        assert len(errors) > 0
        assert any("expression" in e for e in errors)

    def test_annotation_missing_text(self):
        call = _valid_annotation_call()
        call["args"]["data"] = {}
        errors = validate_overlay_call(call)
        assert len(errors) > 0
        assert any("text" in e for e in errors)

    def test_highlight_missing_color(self):
        call = _valid_highlight_call()
        call["args"]["data"] = {}
        errors = validate_overlay_call(call)
        assert len(errors) > 0
        assert any("color" in e for e in errors)

    def test_missing_data(self):
        call = _valid_graph_call()
        del call["args"]["data"]
        errors = validate_overlay_call(call)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# render_overlay_preview
# ---------------------------------------------------------------------------


class TestRenderOverlayPreview:
    def _make_image(self, h=600, w=800):
        """Create a test image (white background)."""
        return np.full((h, w, 3), 200, dtype=np.uint8)

    def test_output_same_dimensions(self):
        img = self._make_image(600, 800)
        result = render_overlay_preview(_valid_graph_call(), img)
        assert result.shape[0] == 600
        assert result.shape[1] == 800

    def test_graph_overlay_changes_image(self):
        img = self._make_image()
        result = render_overlay_preview(_valid_graph_call(), img)
        # The overlay should modify some pixels
        assert not np.array_equal(result, img)

    def test_annotation_overlay_changes_image(self):
        img = self._make_image()
        result = render_overlay_preview(_valid_annotation_call(), img)
        assert not np.array_equal(result, img)

    def test_highlight_overlay_changes_image(self):
        img = self._make_image()
        result = render_overlay_preview(_valid_highlight_call(), img)
        assert not np.array_equal(result, img)

    def test_placement_maps_to_correct_region(self):
        """The overlay should affect pixels within the placement region."""
        img = self._make_image(1000, 1000)
        call = _valid_annotation_call()
        # placement: [50, 50, 300, 400] in 0-1000 coords
        # On a 1000x1000 image, that maps to pixels y:50-300, x:50-400
        result = render_overlay_preview(call, img)

        # Region inside placement should differ
        region = result[50:300, 50:400]
        original_region = img[50:300, 50:400]
        assert not np.array_equal(region, original_region)
