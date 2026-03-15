"""Tests for overlay renderer modules (graph, annotation, highlight)."""

import numpy as np
import pytest

from client.renderer.graph import render_graph
from client.renderer.annotation import render_annotation
from client.renderer.highlight import render_highlight


# ---------------------------------------------------------------------------
# Graph renderer tests
# ---------------------------------------------------------------------------

class TestRenderGraph:
    def test_output_shape(self):
        img = render_graph("x**2", x_range=[-5, 5], y_range=[0, 25], width=640, height=480)
        assert img.shape == (480, 640, 3), "Expected BGR image with shape (H, W, 3)"

    def test_background_is_mostly_black(self):
        img = render_graph("x**2", x_range=[-5, 5], y_range=[0, 25], width=640, height=480)
        # Majority of pixels should be near black (sum of BGR channels < 30)
        pixel_brightness = img.sum(axis=2)
        black_fraction = (pixel_brightness < 30).mean()
        assert black_fraction > 0.5, f"Expected >50% black pixels, got {black_fraction:.1%}"

    def test_content_exists(self):
        img = render_graph("x**2", x_range=[-5, 5], y_range=[0, 25], width=640, height=480)
        # Some pixels should be non-black (the graph line, axes, labels)
        pixel_brightness = img.sum(axis=2)
        bright_fraction = (pixel_brightness > 50).mean()
        assert bright_fraction > 0.01, "Expected some non-black pixels for the graph content"

    def test_different_expressions_produce_different_images(self):
        img1 = render_graph("x**2", x_range=[-5, 5], y_range=[-10, 25], width=640, height=480)
        img2 = render_graph("np.sin(x)", x_range=[-5, 5], y_range=[-2, 2], width=640, height=480)
        # Images should not be identical
        assert not np.array_equal(img1, img2), "Different expressions should produce different images"

    def test_custom_dimensions(self):
        img = render_graph("x", x_range=[0, 10], y_range=[0, 10], width=800, height=600)
        assert img.shape == (600, 800, 3)

    def test_dtype_is_uint8(self):
        img = render_graph("x**2", x_range=[-5, 5], y_range=[0, 25], width=640, height=480)
        assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Annotation renderer tests
# ---------------------------------------------------------------------------

class TestRenderAnnotation:
    def test_output_shape(self):
        img = render_annotation("Hello", width=400, height=200)
        assert img.shape == (200, 400, 3), "Expected BGR image with shape (H, W, 3)"

    def test_background_is_mostly_black(self):
        img = render_annotation("Hi", width=400, height=200)
        pixel_brightness = img.sum(axis=2)
        black_fraction = (pixel_brightness < 30).mean()
        assert black_fraction > 0.5, f"Expected >50% black pixels, got {black_fraction:.1%}"

    def test_content_exists(self):
        img = render_annotation("Hello World", width=400, height=200)
        pixel_brightness = img.sum(axis=2)
        bright_fraction = (pixel_brightness > 50).mean()
        assert bright_fraction > 0.001, "Expected some non-black pixels for text"

    def test_empty_text_produces_all_black(self):
        img = render_annotation("", width=400, height=200)
        assert img.max() == 0, "Empty text should produce an all-black image"

    def test_custom_color(self):
        img = render_annotation("Test", width=400, height=200, color=(0, 255, 0))
        # Any bright pixel should be in the green channel
        bright_mask = img.sum(axis=2) > 50
        if bright_mask.any():
            bright_pixels = img[bright_mask]
            # Green channel should dominate
            assert bright_pixels[:, 1].mean() > bright_pixels[:, 0].mean(), (
                "Green channel should dominate when color is green"
            )
            assert bright_pixels[:, 1].mean() > bright_pixels[:, 2].mean(), (
                "Green channel should dominate when color is green"
            )

    def test_dtype_is_uint8(self):
        img = render_annotation("Test", width=400, height=200)
        assert img.dtype == np.uint8

    def test_word_wrap_long_text(self):
        long_text = "This is a very long sentence that should be wrapped across multiple lines in the annotation renderer"
        img = render_annotation(long_text, width=400, height=300)
        # Should still produce valid output without crashing
        assert img.shape == (300, 400, 3)
        # Should have content
        assert img.max() > 0, "Long wrapped text should produce visible content"


# ---------------------------------------------------------------------------
# Highlight renderer tests
# ---------------------------------------------------------------------------

class TestRenderHighlight:
    def test_output_shape_bgra(self):
        img = render_highlight(width=300, height=200)
        assert img.shape == (200, 300, 4), "Expected BGRA image with shape (H, W, 4)"

    def test_alpha_channel_value(self):
        img = render_highlight(width=300, height=200, alpha=0.3)
        # Alpha channel should be approximately 0.3 * 255 = 76
        expected_alpha = int(0.3 * 255)
        alpha_channel = img[:, :, 3]
        assert np.allclose(alpha_channel, expected_alpha, atol=2), (
            f"Expected alpha ~{expected_alpha}, got mean {alpha_channel.mean():.0f}"
        )

    def test_alpha_channel_value_different(self):
        img = render_highlight(width=300, height=200, alpha=0.7)
        expected_alpha = int(0.7 * 255)
        alpha_channel = img[:, :, 3]
        assert np.allclose(alpha_channel, expected_alpha, atol=2), (
            f"Expected alpha ~{expected_alpha}, got mean {alpha_channel.mean():.0f}"
        )

    def test_color_hex_cyan(self):
        img = render_highlight(width=300, height=200, color_hex="#00ffff")
        # BGR order: cyan = (255, 255, 0) in BGR
        b, g, r = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
        assert b > 200 and g > 200, "Cyan should have high B and G channels"
        assert r < 50, "Cyan should have low R channel"

    def test_color_hex_red(self):
        img = render_highlight(width=300, height=200, color_hex="#ff0000")
        # BGR order: red = (0, 0, 255) in BGR
        b, g, r = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
        assert r > 200, "Red highlight should have high R channel in BGR"
        assert b < 50 and g < 50, "Red highlight should have low B and G channels"

    def test_dtype_is_uint8(self):
        img = render_highlight(width=300, height=200)
        assert img.dtype == np.uint8
