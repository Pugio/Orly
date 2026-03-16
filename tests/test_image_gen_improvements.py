"""Tests for image generation improvements and system prompt updates."""

import numpy as np
from unittest.mock import patch, MagicMock

from client.renderer.image import ENHANCE_PREFIX, render_image
from client.overlay_manager import OverlayManager
from backend.agent import SYSTEM_PROMPT, TOOL_DECLARATIONS


class TestEnhancePrefix:
    """Tests for the ENHANCE_PREFIX constant and prompt enhancement logic."""

    def test_enhance_prefix_constant(self):
        """ENHANCE_PREFIX is a non-empty string."""
        assert isinstance(ENHANCE_PREFIX, str)
        assert len(ENHANCE_PREFIX) > 0

    def test_render_image_enhance_false_no_prefix(self):
        """With enhance=False, prompt is unchanged (no ENHANCE_PREFIX)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response

        with patch("client.renderer.image.get_genai_client", return_value=mock_client):
            render_image("draw a cat", 200, 200, enhance=False, style="default")

            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
            # With no reference_frame, contents is just the text prompt
            assert contents == "draw a cat"

    def test_render_image_enhance_true_default_style(self):
        """When enhance=True and style=default, prompt starts with ENHANCE_PREFIX."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response

        with patch("client.renderer.image.get_genai_client", return_value=mock_client):
            render_image("draw a cat", 200, 200, enhance=True, style="default")

            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
            assert contents.startswith(ENHANCE_PREFIX)
            assert "draw a cat" in contents

    def test_render_image_enhance_true_technical_style(self):
        """With enhance=True and style=technical, ENHANCE_PREFIX is NOT prepended."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response

        with patch("client.renderer.image.get_genai_client", return_value=mock_client):
            render_image("draw a diagram", 200, 200, enhance=True, style="technical")

            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs.get("contents") or call_args[1].get("contents")
            assert not contents.startswith(ENHANCE_PREFIX)
            assert "draw a diagram" in contents


class TestOverlayManagerNewParams:
    """Tests for session_store and notify_fn parameters on OverlayManager."""

    def test_overlay_manager_session_store_param(self):
        """OverlayManager accepts session_store param."""
        mock_store = MagicMock()
        om = OverlayManager(
            H_proj=None, proj_width=640, proj_height=480,
            mode="screen", session_store=mock_store,
        )
        assert om.session_store is mock_store

    def test_overlay_manager_notify_fn_param(self):
        """OverlayManager accepts notify_fn param."""
        mock_fn = MagicMock()
        om = OverlayManager(
            H_proj=None, proj_width=640, proj_height=480,
            mode="screen", notify_fn=mock_fn,
        )
        assert om.notify_fn is mock_fn

    def test_overlay_manager_defaults_none(self):
        """session_store and notify_fn default to None."""
        om = OverlayManager(H_proj=None)
        assert om.session_store is None
        assert om.notify_fn is None


class TestSystemPromptUpdates:
    """Tests for system prompt content additions."""

    def test_system_prompt_has_overlay_naming(self):
        """SYSTEM_PROMPT contains 'OVERLAY NAMING'."""
        assert "OVERLAY NAMING" in SYSTEM_PROMPT

    def test_system_prompt_mentions_overlay_tool(self):
        """SYSTEM_PROMPT mentions 'overlay' tool."""
        assert "overlay" in SYSTEM_PROMPT

    def test_system_prompt_mentions_query_tool(self):
        """SYSTEM_PROMPT mentions 'query' tool."""
        assert "query" in SYSTEM_PROMPT


class TestToolDeclarations:
    """Tests for consolidated tool declarations."""

    def test_tool_declarations_has_consolidated_tools(self):
        """TOOL_DECLARATIONS includes overlay, query, music."""
        tool_names = {d["name"] for d in TOOL_DECLARATIONS}
        assert "overlay" in tool_names
        assert "query" in tool_names
        assert "music" in tool_names
        assert len(tool_names) == 3
