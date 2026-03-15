"""Extended tests for renderers: graph, annotation, highlight."""

import numpy as np
import pytest

from client.renderer.graph import render_graph
from client.renderer.annotation import render_annotation
from client.renderer.highlight import render_highlight


# ---------------------------------------------------------------------------
# Graph renderer — extended
# ---------------------------------------------------------------------------


class TestRenderGraphExtended:
    def test_sin_expression(self):
        img = render_graph("np.sin(x)", x_range=[0, 6.28], y_range=[-1.5, 1.5],
                           width=640, height=480)
        assert img.shape == (480, 640, 3)
        assert img.max() > 0

    def test_implicit_multiplication(self):
        """'7x' should be auto-converted to '7*x'."""
        img = render_graph("7x", x_range=[-5, 5], y_range=[-35, 35],
                           width=640, height=480)
        assert img.shape == (480, 640, 3)
        assert img.max() > 0

    def test_small_dimensions(self):
        img = render_graph("x", x_range=[-1, 1], y_range=[-1, 1],
                           width=50, height=50)
        assert img.shape == (50, 50, 3)

    def test_large_dimensions(self):
        img = render_graph("x**2", x_range=[-10, 10], y_range=[0, 100],
                           width=1920, height=1080)
        assert img.shape == (1080, 1920, 3)

    def test_constant_expression(self):
        """Constant via numpy broadcast: 0*x + 5 works; bare '5' does not."""
        img = render_graph("0*x + 5", x_range=[-10, 10], y_range=[0, 10],
                           width=400, height=300)
        assert img.shape == (300, 400, 3)
        assert img.max() > 0

    def test_negative_range(self):
        img = render_graph("x", x_range=[-100, -50], y_range=[-100, -50],
                           width=400, height=300)
        assert img.shape == (300, 400, 3)
        assert img.max() > 0


# ---------------------------------------------------------------------------
# Annotation renderer — extended
# ---------------------------------------------------------------------------


class TestRenderAnnotationExtended:
    def test_single_character(self):
        img = render_annotation("A", width=400, height=200)
        assert img.max() > 0

    def test_very_long_word_doesnt_crash(self):
        img = render_annotation("x" * 200, width=400, height=200)
        assert img.shape == (200, 400, 3)

    def test_multiline_implicit(self):
        """Long text should word-wrap."""
        long_text = " ".join(["word"] * 50)
        img = render_annotation(long_text, width=400, height=400)
        assert img.shape == (400, 400, 3)
        assert img.max() > 0

    def test_custom_font_scale(self):
        img_small = render_annotation("Test", width=400, height=200, font_scale=1.0)
        img_large = render_annotation("Test", width=400, height=200, font_scale=3.0)
        # Larger font should produce more bright pixels
        bright_small = (img_small.sum(axis=2) > 50).sum()
        bright_large = (img_large.sum(axis=2) > 50).sum()
        assert bright_large > bright_small

    def test_blue_color(self):
        img = render_annotation("Test", width=400, height=200, color=(255, 0, 0))
        bright_mask = img.sum(axis=2) > 50
        if bright_mask.any():
            bright_px = img[bright_mask]
            # Blue channel (index 0) should dominate
            assert bright_px[:, 0].mean() > bright_px[:, 1].mean()
            assert bright_px[:, 0].mean() > bright_px[:, 2].mean()

    def test_white_color(self):
        img = render_annotation("Test", width=400, height=200, color=(255, 255, 255))
        bright_mask = img.sum(axis=2) > 100
        assert bright_mask.any()

    def test_minimum_dimensions(self):
        img = render_annotation("A", width=1, height=1)
        assert img.shape == (1, 1, 3)


# ---------------------------------------------------------------------------
# Highlight renderer — extended
# ---------------------------------------------------------------------------


class TestRenderHighlightExtended:
    def test_green_hex(self):
        img = render_highlight(200, 100, color_hex="#00ff00")
        assert img[:, :, 1].mean() > 200  # green channel high
        assert img[:, :, 0].mean() < 10   # blue channel low
        assert img[:, :, 2].mean() < 10   # red channel low

    def test_white_hex(self):
        img = render_highlight(200, 100, color_hex="#ffffff")
        assert img[:, :, 0].mean() > 200
        assert img[:, :, 1].mean() > 200
        assert img[:, :, 2].mean() > 200

    def test_alpha_zero(self):
        img = render_highlight(200, 100, alpha=0.0)
        assert img[:, :, 3].max() == 0

    def test_alpha_one(self):
        img = render_highlight(200, 100, alpha=1.0)
        assert img[:, :, 3].min() == 255

    def test_small_dimensions(self):
        img = render_highlight(1, 1)
        assert img.shape == (1, 1, 4)

    def test_large_dimensions(self):
        img = render_highlight(1920, 1080)
        assert img.shape == (1080, 1920, 4)

    def test_hex_with_hash(self):
        img = render_highlight(100, 100, color_hex="#abcdef")
        assert img.shape == (100, 100, 4)

    def test_hex_without_hash(self):
        """Should handle hex without leading #."""
        img = render_highlight(100, 100, color_hex="abcdef")
        assert img.shape == (100, 100, 4)
