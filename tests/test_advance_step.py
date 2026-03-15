"""Tests for advance_step tool and steps content_type in project_overlay."""

from backend.tools import advance_step, project_overlay
from backend.agent import TOOL_DECLARATIONS, TOOL_REGISTRY, function_to_declaration


class TestAdvanceStep:
    def test_advance_step_returns_status(self):
        result = advance_step(overlay_name="math-steps", step_number=2)
        assert result["status"] == "advancing"
        assert result["overlay_name"] == "math-steps"
        assert result["step_number"] == 2

    def test_advance_step_in_tool_registry(self):
        assert "advance_step" in TOOL_REGISTRY

    def test_advance_step_declaration_schema(self):
        decl = function_to_declaration(advance_step)
        assert decl["name"] == "advance_step"
        props = decl["parameters"]["properties"]
        assert props["overlay_name"]["type"] == "STRING"
        assert props["step_number"]["type"] == "INTEGER"

    def test_project_overlay_steps_valid(self):
        result = project_overlay(
            content_type="steps",
            placement=[100.0, 100.0, 800.0, 800.0],
            title="Solution Steps",
            data={"steps": [{"title": "Step 1", "content": "Factor"}]},
        )
        assert result["status"] == "displayed"
        assert result["content_type"] == "steps"

    def test_project_overlay_steps_in_valid_types(self):
        # "steps" should not be rejected as invalid
        result = project_overlay(
            content_type="steps",
            placement=[0.0, 0.0, 500.0, 500.0],
            title="Test",
            data={},
        )
        assert result["status"] != "error"
