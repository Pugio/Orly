"""Integration tests for new overlay types (number_line, geometry, chemistry, steps).

Tests that the full pipeline works: project_overlay validation, overlay_manager dispatch,
and system prompt documentation.
"""

import numpy as np
import pytest

from backend.agent import SYSTEM_PROMPT, TOOL_DECLARATIONS, TOOL_REGISTRY
from backend.tools import overlay


# --- overlay create validation ---


class TestProjectOverlayNewTypes:
    def test_overlay_create_number_line_valid(self):
        result = overlay(
            action="create", content_type="number_line",
            placement=[0, 0, 500, 1000], title="NL",
            data={"min_val": 0, "max_val": 10},
        )
        assert result["status"] == "displayed"

    def test_overlay_create_geometry_valid(self):
        result = overlay(
            action="create", content_type="geometry",
            placement=[0, 0, 1000, 1000], title="Triangle",
            data={"elements": [{"type": "point", "pos": [0, 0], "label": "A"}]},
        )
        assert result["status"] == "displayed"

    def test_overlay_create_chemistry_valid(self):
        result = overlay(
            action="create", content_type="chemistry",
            placement=[0, 0, 500, 500], title="H2O",
            data={"atoms": [{"symbol": "O", "pos": [0, 0]}], "bonds": []},
        )
        assert result["status"] == "displayed"

    def test_overlay_create_steps_valid(self):
        result = overlay(
            action="create", content_type="steps",
            placement=[0, 0, 1000, 1000], title="Solution",
            data={"steps": [{"title": "Step 1", "content": "Factor"}]},
        )
        assert result["status"] == "displayed"


# --- OverlayManager dispatch ---


class TestOverlayManagerNewRenderers:
    """Test that OverlayManager.render_overlay dispatches to new renderers."""

    def _make_manager(self):
        from client.overlay_manager import OverlayManager
        return OverlayManager(H_proj=None, mode="screen", proj_width=1280, proj_height=720)

    def test_renders_number_line(self):
        om = self._make_manager()
        result = om.render_overlay(
            "number_line", [0, 0, 500, 1000], "NL",
            {"min_val": -5, "max_val": 5},
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.sum() > 0

    def test_renders_geometry(self):
        om = self._make_manager()
        result = om.render_overlay(
            "geometry", [0, 0, 1000, 1000], "Tri",
            {"elements": [{"type": "point", "pos": [0, 0], "label": "O"}],
             "x_range": [-5, 5], "y_range": [-5, 5]},
        )
        assert isinstance(result, np.ndarray)
        assert result.sum() > 0

    def test_renders_chemistry(self):
        om = self._make_manager()
        result = om.render_overlay(
            "chemistry", [0, 0, 500, 500], "H2O",
            {"atoms": [{"symbol": "O", "pos": [0, 0]}], "bonds": []},
        )
        assert isinstance(result, np.ndarray)
        assert result.sum() > 0

    def test_renders_steps(self):
        om = self._make_manager()
        result = om.render_overlay(
            "steps", [0, 0, 1000, 500], "Steps",
            {"steps": [{"title": "Step 1", "content": "x=1"}],
             "visible_count": 1},
        )
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3


# --- System prompt documentation ---


class TestSystemPromptNewTypes:
    def test_mentions_number_line(self):
        assert "number_line" in SYSTEM_PROMPT

    def test_mentions_geometry(self):
        assert "geometry" in SYSTEM_PROMPT

    def test_mentions_chemistry(self):
        assert "chemistry" in SYSTEM_PROMPT

    def test_mentions_steps(self):
        assert "steps" in SYSTEM_PROMPT
