"""Tests for number line renderer."""

import numpy as np
from client.renderer.number_line import _render_number_line_impl as render_number_line


def test_number_line_output_shape():
    result = render_number_line(-5, 5, [], [], 640, 200)
    assert result.shape == (200, 640, 3)
    assert result.dtype == np.uint8


def test_number_line_black_background():
    result = render_number_line(0, 10, [], [], 640, 200)
    # Corners should be black
    assert result[0, 0].sum() < 30
    assert result[0, -1].sum() < 30


def test_number_line_has_content():
    result = render_number_line(-5, 5, [], [], 640, 200)
    assert result.sum() > 0  # axis line + ticks


def test_number_line_with_point():
    points = [{"value": 3, "label": "x", "color": "#00ff00"}]
    result = render_number_line(-5, 5, points, [], 640, 200)
    assert result[:, :, 1].max() > 100  # green channel has content


def test_number_line_with_range():
    ranges = [{"start": -1, "end": 2, "color": "#ffff00", "label": "S"}]
    result = render_number_line(-5, 5, [], ranges, 640, 200)
    assert result.sum() > 0


def test_number_line_invalid_range_ignored():
    # Point out of range shouldn't crash
    points = [{"value": 15, "label": "x"}]
    result = render_number_line(0, 10, points, [], 640, 200)
    assert result.shape == (200, 640, 3)


def test_number_line_min_equals_max():
    result = render_number_line(5, 5, [], [], 640, 200)
    assert result.shape == (200, 640, 3)


def test_number_line_negative_range():
    result = render_number_line(-100, -50, [], [], 640, 200)
    assert result.sum() > 0
