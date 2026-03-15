"""Tests for the refresh_view tool and image generation with camera reference."""

from backend.tools import project_overlay, refresh_view


class TestRefreshViewTool:
    def test_returns_success(self):
        result = refresh_view(reason="check work")
        assert result["status"] == "refreshing"

    def test_returns_description(self):
        result = refresh_view(reason="verify position")
        assert "description" in result

    def test_includes_reason(self):
        result = refresh_view(reason="student wrote something")
        assert result["reason"] == "student wrote something"


class TestProjectOverlayImageWithView:
    def test_include_view_accepted(self):
        result = project_overlay(
            content_type="image",
            placement=[100, 100, 600, 600],
            title="Test",
            data={"prompt": "a circle", "include_view": True},
        )
        assert result["status"] == "displayed"


class TestImageRendererWithReference:
    def test_render_image_with_reference_frame(self):
        """render_image should accept an optional reference_frame."""
        from unittest.mock import patch, MagicMock
        from client.renderer.image import render_image
        import numpy as np
        import cv2

        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", fake_img)

        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = png_bytes.tobytes()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]

        ref_frame = np.zeros((768, 768, 3), dtype=np.uint8)

        with patch("client.renderer.image._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response
            result = render_image("a circle", width=400, height=300,
                                  reference_frame=ref_frame)

        assert result.shape == (300, 400, 3)
        # Verify the API was called with image content (not just text)
        call_args = mock_client.return_value.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        # Should be a list (multi-part) when reference_frame is provided
        assert isinstance(contents, list)

    def test_render_image_without_reference_frame(self):
        """Without reference_frame, contents should be a plain string."""
        from unittest.mock import patch, MagicMock
        from client.renderer.image import render_image
        import numpy as np
        import cv2

        fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, png_bytes = cv2.imencode(".png", fake_img)

        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.data = png_bytes.tobytes()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [mock_part]

        with patch("client.renderer.image._get_genai_client") as mock_client:
            mock_client.return_value.models.generate_content.return_value = mock_response
            result = render_image("a circle", width=400, height=300)

        assert result.shape == (300, 400, 3)
        call_args = mock_client.return_value.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
        assert isinstance(contents, str)
