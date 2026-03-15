"""Tests for PoC 4 — Gemini spatial localization."""

import json

import numpy as np
import pytest

from poc.poc4_gemini_localize import (
    build_localization_prompt,
    draw_detections,
    parse_gemini_response,
    validate_detections,
)


# ---------------------------------------------------------------------------
# parse_gemini_response
# ---------------------------------------------------------------------------


class TestParseGeminiResponse:
    def test_valid_json_array(self):
        raw = json.dumps([
            {"label": "equation", "box_2d": [100, 200, 300, 400]},
            {"label": "graph", "box_2d": [500, 600, 700, 800]},
        ])
        result = parse_gemini_response(raw)
        assert len(result) == 2
        assert result[0]["label"] == "equation"
        assert result[0]["box_2d"] == [100, 200, 300, 400]
        assert result[1]["label"] == "graph"

    def test_json_in_markdown_code_fences(self):
        raw = '```json\n[{"label": "text", "box_2d": [10, 20, 30, 40]}]\n```'
        result = parse_gemini_response(raw)
        assert len(result) == 1
        assert result[0]["label"] == "text"
        assert result[0]["box_2d"] == [10, 20, 30, 40]

    def test_invalid_json_returns_empty(self):
        result = parse_gemini_response("this is not json at all")
        assert result == []

    def test_empty_string_returns_empty(self):
        result = parse_gemini_response("")
        assert result == []

    def test_extra_fields_preserved(self):
        raw = json.dumps([
            {"label": "diagram", "box_2d": [0, 0, 500, 500], "confidence": 0.95},
        ])
        result = parse_gemini_response(raw)
        assert len(result) == 1
        assert result[0]["label"] == "diagram"
        assert result[0]["box_2d"] == [0, 0, 500, 500]

    def test_code_fence_without_language_tag(self):
        raw = '```\n[{"label": "note", "box_2d": [1, 2, 3, 4]}]\n```'
        result = parse_gemini_response(raw)
        assert len(result) == 1
        assert result[0]["label"] == "note"


# ---------------------------------------------------------------------------
# validate_detections
# ---------------------------------------------------------------------------


class TestValidateDetections:
    def test_valid_detections_pass_through(self):
        detections = [
            {"label": "equation", "box_2d": [100, 200, 300, 400]},
            {"label": "graph", "box_2d": [0, 0, 1000, 1000]},
        ]
        result = validate_detections(detections)
        assert len(result) == 2

    def test_out_of_range_filtered(self):
        detections = [
            {"label": "bad", "box_2d": [-10, 200, 300, 400]},
            {"label": "also_bad", "box_2d": [100, 200, 300, 1100]},
        ]
        result = validate_detections(detections)
        assert len(result) == 0

    def test_ymin_gte_ymax_filtered(self):
        detections = [
            {"label": "inverted", "box_2d": [500, 200, 300, 400]},
            {"label": "equal", "box_2d": [300, 200, 300, 400]},
        ]
        result = validate_detections(detections)
        assert len(result) == 0

    def test_xmin_gte_xmax_filtered(self):
        detections = [
            {"label": "inverted_x", "box_2d": [100, 500, 300, 400]},
        ]
        result = validate_detections(detections)
        assert len(result) == 0

    def test_missing_label_filtered(self):
        detections = [
            {"label": "", "box_2d": [100, 200, 300, 400]},
            {"box_2d": [100, 200, 300, 400]},
        ]
        result = validate_detections(detections)
        assert len(result) == 0

    def test_wrong_number_of_box_values_filtered(self):
        detections = [
            {"label": "short", "box_2d": [100, 200, 300]},
            {"label": "long", "box_2d": [100, 200, 300, 400, 500]},
            {"label": "missing", "box_2d": []},
        ]
        result = validate_detections(detections)
        assert len(result) == 0

    def test_missing_box_2d_filtered(self):
        detections = [{"label": "no_box"}]
        result = validate_detections(detections)
        assert len(result) == 0

    def test_mixed_valid_and_invalid(self):
        detections = [
            {"label": "good", "box_2d": [100, 200, 300, 400]},
            {"label": "bad_range", "box_2d": [-1, 200, 300, 400]},
            {"label": "also_good", "box_2d": [0, 0, 500, 500]},
        ]
        result = validate_detections(detections)
        assert len(result) == 2
        assert result[0]["label"] == "good"
        assert result[1]["label"] == "also_good"


# ---------------------------------------------------------------------------
# draw_detections
# ---------------------------------------------------------------------------


class TestDrawDetections:
    def _make_image(self, w=640, h=480):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_output_same_shape(self):
        img = self._make_image()
        detections = [{"label": "test", "box_2d": [100, 100, 400, 400]}]
        out = draw_detections(img, detections)
        assert out.shape == img.shape

    def test_detections_modify_image(self):
        img = self._make_image()
        detections = [{"label": "test", "box_2d": [100, 100, 400, 400]}]
        out = draw_detections(img, detections)
        assert not np.array_equal(out, img)

    def test_empty_detections_unchanged(self):
        img = self._make_image()
        out = draw_detections(img, [])
        assert np.array_equal(out, img)

    def test_does_not_mutate_input(self):
        img = self._make_image()
        original = img.copy()
        detections = [{"label": "test", "box_2d": [100, 100, 400, 400]}]
        draw_detections(img, detections)
        assert np.array_equal(img, original)


# ---------------------------------------------------------------------------
# build_localization_prompt
# ---------------------------------------------------------------------------


class TestBuildLocalizationPrompt:
    def test_returns_nonempty_string(self):
        prompt = build_localization_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_key_terms(self):
        prompt = build_localization_prompt()
        for term in ["JSON", "box_2d", "label", "1000"]:
            assert term in prompt, f"Prompt missing key term: {term}"
