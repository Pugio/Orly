"""Tests for client.renderer.transition — crossfade, slide_in, build_up."""

import numpy as np
import pytest

from client.renderer.transition import crossfade, slide_in, build_up


# --- crossfade ---


class TestCrossfade:
    def test_crossfade_alpha_zero_returns_old(self):
        old = np.full((100, 100, 3), 200, dtype=np.uint8)
        new = np.full((100, 100, 3), 50, dtype=np.uint8)
        result = crossfade(old, new, 0.0)
        np.testing.assert_array_equal(result, old)

    def test_crossfade_alpha_one_returns_new(self):
        old = np.full((100, 100, 3), 200, dtype=np.uint8)
        new = np.full((100, 100, 3), 50, dtype=np.uint8)
        result = crossfade(old, new, 1.0)
        np.testing.assert_array_equal(result, new)

    def test_crossfade_alpha_half_blends(self):
        old = np.full((100, 100, 3), 0, dtype=np.uint8)
        new = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = crossfade(old, new, 0.5)
        # 0*0.5 + 255*0.5 = 127.5 → clipped to 127 or 128
        assert np.all((result >= 127) & (result <= 128))

    def test_crossfade_different_sizes_raises(self):
        old = np.zeros((100, 100, 3), dtype=np.uint8)
        new = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            crossfade(old, new, 0.5)


# --- slide_in ---


class TestSlideIn:
    def test_slide_in_progress_zero_all_black(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = slide_in(img, "left", 0.0)
        assert np.all(result == 0)

    def test_slide_in_progress_one_full_image(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = slide_in(img, "left", 1.0)
        np.testing.assert_array_equal(result, img)

    def test_slide_in_left_half_progress(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = slide_in(img, "left", 0.5)
        h, w = result.shape[:2]
        mid = w // 2
        # Left half should be black (not yet slid in)
        assert np.all(result[:, :mid, :] == 0)
        # Right half should be white (the leading edge of the image)
        assert np.all(result[:, mid:, :] == 255)

    def test_slide_in_up_half_progress(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = slide_in(img, "up", 0.5)
        h, w = result.shape[:2]
        mid = h // 2
        # Top half should be black
        assert np.all(result[:mid, :, :] == 0)
        # Bottom half should be white
        assert np.all(result[mid:, :, :] == 255)

    def test_slide_in_invalid_direction_raises(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Invalid direction"):
            slide_in(img, "diagonal", 0.5)


# --- build_up ---


class TestBuildUp:
    def test_build_up_step_zero_all_black(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = build_up(img, 0, 4)
        assert np.all(result == 0)

    def test_build_up_all_steps_returns_full(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = build_up(img, 4, 4)
        np.testing.assert_array_equal(result, img)

    def test_build_up_half_shows_top_half(self):
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        result = build_up(img, 2, 4)
        h = img.shape[0]
        mid = h // 2
        # Top half visible
        assert np.all(result[:mid, :, :] == 255)
        # Bottom half black
        assert np.all(result[mid:, :, :] == 0)

    def test_build_up_single_step_of_three(self):
        img = np.full((90, 200, 3), 255, dtype=np.uint8)
        result = build_up(img, 1, 3)
        # 1/3 of 90 = 30 rows visible
        assert np.all(result[:30, :, :] == 255)
        assert np.all(result[30:, :, :] == 0)
