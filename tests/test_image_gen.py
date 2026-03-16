"""Tests for the image generation tool and renderer."""

import numpy as np

from backend.tools import overlay


class TestProjectOverlayImageType:
    """Validate that 'image' is accepted as a content_type."""

    def test_image_type_accepted(self):
        result = overlay(
            action="create",
            content_type="image",
            placement=[100, 100, 600, 600],
            title="Test image",
            data={"prompt": "a red circle on white background"},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "image"

    def test_image_missing_prompt_still_valid(self):
        """Tool validation doesn't check data contents — renderer handles that."""
        result = overlay(
            action="create",
            content_type="image",
            placement=[0, 0, 500, 500],
            title="No prompt",
            data={},
        )
        assert result["status"] == "displayed"


class TestRenderLoading:
    """Test the loading placeholder renderer."""

    def test_output_shape(self):
        from client.renderer.image import render_loading
        img = render_loading("test prompt", width=640, height=480)
        assert img.shape == (480, 640, 3)

    def test_has_content(self):
        from client.renderer.image import render_loading
        img = render_loading("test prompt", width=640, height=480)
        assert img.max() > 0, "Loading placeholder should have visible content"

    def test_dtype(self):
        from client.renderer.image import render_loading
        img = render_loading("test", width=400, height=300)
        assert img.dtype == np.uint8


class TestRenderImage:
    """Test the image renderer with mocked API calls."""

    def test_render_returns_correct_shape(self):
        """render_image should return a BGR image of the requested size."""
        from unittest.mock import patch, MagicMock
        from client.renderer.image import render_image
        import cv2

        # Create a fake 100x100 red PNG
        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_img[:, :, 2] = 255  # red in BGR
        _, png_bytes = cv2.imencode(".png", fake_img)

        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = None
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = png_bytes.tobytes()
        mock_part.inline_data.mime_type = "image/png"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]

        with patch("client.renderer.image.get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response
            result = render_image("a red square", width=400, height=300)

        assert result.shape == (300, 400, 3)
        assert result.dtype == np.uint8

    def test_render_fallback_on_error(self):
        """If image generation fails, return annotation-style fallback."""
        from unittest.mock import patch
        from client.renderer.image import render_image

        with patch("client.renderer.image.get_genai_client", side_effect=Exception("no key")):
            result = render_image("test prompt", width=400, height=300)

        assert result.shape == (300, 400, 3)
        assert result.dtype == np.uint8
        assert result.max() > 0

    def test_render_fallback_on_no_candidates(self):
        """If API returns no candidates, render fallback."""
        from unittest.mock import patch, MagicMock
        from client.renderer.image import render_image

        mock_response = MagicMock()
        mock_response.candidates = []

        with patch("client.renderer.image.get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response
            result = render_image("bad prompt", width=400, height=300)

        assert result.shape == (300, 400, 3)
        assert result.max() > 0

    def test_render_fallback_on_no_image_in_response(self):
        """If API returns text only (no image), render fallback."""
        from unittest.mock import patch, MagicMock
        from client.renderer.image import render_image

        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = None
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]

        with patch("client.renderer.image.get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response
            result = render_image("bad prompt", width=400, height=300)

        assert result.shape == (300, 400, 3)
        assert result.max() > 0


class TestOverlayManagerImageAsync:
    """Test that OverlayManager shows loading then swaps in result."""

    def test_handle_image_shows_loading_immediately(self):
        """handle_tool_result for image should update canvas immediately (loading)."""
        from client.overlay_manager import OverlayManager
        from unittest.mock import patch
        import threading

        om = OverlayManager(H_proj=None, proj_width=640, proj_height=480, mode="screen")
        assert om.canvas.max() == 0  # starts black

        # Use an Event to block the background thread so it can't overwrite
        # the loading placeholder before we check the canvas.
        block = threading.Event()

        def slow_render(*args, **kwargs):
            block.wait(timeout=5)
            return np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("client.overlay_manager.render_image", side_effect=slow_render):
            om.handle_tool_result("overlay", {
                "action": "create",
                "content_type": "image",
                "placement": [100, 100, 600, 600],
                "title": "test",
                "data": {"prompt": "a circle"},
            })

            # Animation runs in a background thread — give it a moment to
            # render the first frame onto the canvas.
            import time
            time.sleep(0.1)
            assert om.canvas.max() > 0, "Loading animation should be on canvas"
            block.set()  # unblock the background thread
