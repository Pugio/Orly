"""Tests for the PaintCanvas direct pixel drawing system."""

import numpy as np
import pytest

from client.paint_canvas import PaintCanvas


class TestPaintCanvasBasic:
    def test_create_canvas(self):
        """Canvas initialises to black (transparent)."""
        c = PaintCanvas(640, 480)
        assert c.width == 640
        assert c.height == 480
        img = c.get_image()
        assert img.shape == (480, 640, 3)
        assert not c.has_content()

    def test_circle_draws_pixels(self):
        c = PaintCanvas(100, 100)
        c.circle(500, 500, 100, (0, 0, 255))  # red circle at center
        assert c.has_content()
        img = c.get_image()
        # Center pixel should be red
        assert img[50, 50, 2] > 200  # red channel

    def test_line_draws_pixels(self):
        c = PaintCanvas(100, 100)
        c.line(0, 0, 1000, 1000, (0, 255, 0), thickness=30)
        assert c.has_content()

    def test_rectangle_draws_pixels(self):
        c = PaintCanvas(100, 100)
        c.rectangle(200, 200, 800, 800, (255, 0, 0))
        assert c.has_content()
        img = c.get_image()
        # Center should be filled
        assert img[50, 50, 0] > 200  # blue channel

    def test_text_draws_pixels(self):
        c = PaintCanvas(200, 100)
        c.text("Hi", 500, 200, (255, 255, 255), scale=1.0)
        assert c.has_content()

    def test_stamp_draws_filled_circle(self):
        c = PaintCanvas(100, 100)
        c.stamp(500, 500, 50, (0, 255, 255))
        img = c.get_image()
        assert img[50, 50, 1] > 200  # green channel

    def test_clear_resets_to_black(self):
        c = PaintCanvas(100, 100)
        c.circle(500, 500, 200, (255, 255, 255))
        assert c.has_content()
        c.clear()
        assert not c.has_content()

    def test_visibility_toggle(self):
        c = PaintCanvas(100, 100)
        assert c.visible is True
        c.visible = False
        assert c.visible is False


class TestPaintCanvasComposite:
    def test_composite_onto_target(self):
        """Non-black pixels overwrite target."""
        c = PaintCanvas(100, 100)
        c.rectangle(0, 0, 1000, 1000, (0, 0, 255))  # full red

        target = np.zeros((100, 100, 3), dtype=np.uint8)
        result = c.composite_onto(target)
        # Target should now be red everywhere
        assert result[50, 50, 2] > 200

    def test_composite_black_is_transparent(self):
        """Black pixels on canvas don't overwrite target."""
        c = PaintCanvas(100, 100)
        # Only draw a small circle
        c.circle(200, 200, 50, (0, 255, 0))

        target = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = c.composite_onto(target)
        # Far from circle should still be 128
        assert result[80, 80, 0] == 128

    def test_composite_invisible_canvas_is_noop(self):
        c = PaintCanvas(100, 100)
        c.rectangle(0, 0, 1000, 1000, (255, 255, 255))
        c.visible = False

        target = np.zeros((100, 100, 3), dtype=np.uint8)
        result = c.composite_onto(target)
        assert not np.any(result > 0)

    def test_composite_resizes_if_different_dimensions(self):
        """Canvas resizes to target dims during compositing."""
        c = PaintCanvas(200, 200)
        c.rectangle(0, 0, 1000, 1000, (0, 0, 255))

        target = np.zeros((100, 100, 3), dtype=np.uint8)
        result = c.composite_onto(target)
        assert result.shape == (100, 100, 3)
        assert result[50, 50, 2] > 200


class TestPaintCanvasCoordinates:
    def test_to_px_converts_normalized_to_pixels(self):
        c = PaintCanvas(640, 480)
        # (500, 500) normalized should be center
        px, py = c._to_px(500, 500)
        assert px == 320
        assert py == 240

    def test_to_px_corners(self):
        c = PaintCanvas(1000, 1000)
        assert c._to_px(0, 0) == (0, 0)
        assert c._to_px(1000, 1000) == (1000, 1000)

    def test_size_to_px(self):
        c = PaintCanvas(100, 100)
        # 100 out of 1000 = 10% of avg(100, 100) = 10
        assert c._size_to_px(100) == 10
