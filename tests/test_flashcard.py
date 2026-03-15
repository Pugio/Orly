"""Tests for flashcard renderer and tools."""
import numpy as np
import pytest


class TestFlashcardRenderer:
    def test_render_front(self):
        from client.renderer.flashcard import render_flashcard
        data = {"front": "What is 2+2?", "back": "4", "show_back": False}
        img = render_flashcard(data, 400, 300)
        assert img.shape == (300, 400, 3)
        assert img.dtype == np.uint8
        assert img.max() > 0  # has content

    def test_render_back(self):
        from client.renderer.flashcard import render_flashcard
        data = {"front": "What is 2+2?", "back": "4", "show_back": True}
        img = render_flashcard(data, 400, 300)
        assert img.shape == (300, 400, 3)
        assert img.max() > 0

    def test_render_front_and_back_differ(self):
        from client.renderer.flashcard import render_flashcard
        data_front = {"front": "Question?", "back": "Answer!", "show_back": False}
        data_back = {"front": "Question?", "back": "Answer!", "show_back": True}
        front = render_flashcard(data_front, 400, 300)
        back = render_flashcard(data_back, 400, 300)
        assert not np.array_equal(front, back)

    def test_render_empty_text(self):
        from client.renderer.flashcard import render_flashcard
        data = {"front": "", "back": "", "show_back": False}
        img = render_flashcard(data, 200, 200)
        assert img.shape == (200, 200, 3)

    def test_render_default_show_back_false(self):
        from client.renderer.flashcard import render_flashcard
        data = {"front": "Q", "back": "A"}  # no show_back key
        img = render_flashcard(data, 200, 200)
        assert img.shape == (200, 200, 3)

    def test_spec_registered(self):
        from client.renderer.registry import get, valid_types, _REGISTRY
        _REGISTRY.clear()  # force re-discovery
        spec = get("flashcard")
        assert spec is not None
        assert spec["name"] == "flashcard"
        assert callable(spec["render"])
        assert "flashcard" in valid_types()

    def test_render_with_title(self):
        from client.renderer.flashcard import render_flashcard
        data = {"front": "Q", "back": "A"}
        img = render_flashcard(data, 300, 200, title="Card 1")
        assert img.shape == (200, 300, 3)


class TestFlipFlashcardTool:
    def test_flip_flashcard_returns_status(self):
        from backend.tools import flip_flashcard
        result = flip_flashcard("vocab_card")
        assert result["status"] == "flipping"
        assert result["overlay_name"] == "vocab_card"
