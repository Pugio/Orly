"""Tests for client.renderer.steps — render_steps."""

import numpy as np

from client.renderer.steps import render_steps


class TestRenderSteps:
    def test_render_steps_zero_visible(self):
        steps = [{"title": "Step 1", "content": "Do something"}]
        result = render_steps(steps, visible_count=0, width=400, height=300)
        assert np.all(result == 0)

    def test_render_steps_one_visible(self):
        steps = [
            {"title": "Step 1", "content": "First thing"},
            {"title": "Step 2", "content": "Second thing"},
        ]
        result = render_steps(steps, visible_count=1, width=400, height=300)
        # Top portion should have non-black pixels
        h = result.shape[0]
        top_half = result[: h // 2, :, :]
        assert np.any(top_half > 0)

    def test_render_steps_all_visible(self):
        steps = [
            {"title": "Step 1", "content": "First"},
            {"title": "Step 2", "content": "Second"},
        ]
        result = render_steps(steps, visible_count=2, width=400, height=300)
        # Non-black pixels throughout
        assert np.any(result > 0)

    def test_render_steps_empty_list(self):
        result = render_steps([], visible_count=0, width=400, height=300)
        assert np.all(result == 0)

    def test_render_steps_output_dimensions(self):
        steps = [{"title": "Step 1", "content": "Hello"}]
        result = render_steps(steps, visible_count=1, width=640, height=480)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_render_steps_visible_count_clamped(self):
        steps = [{"title": "Step 1", "content": "Only one"}]
        # visible_count > len(steps) should not crash
        result = render_steps(steps, visible_count=5, width=400, height=300)
        assert result.shape == (300, 400, 3)
        assert np.any(result > 0)
