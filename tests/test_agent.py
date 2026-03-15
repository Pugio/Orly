"""Tests for backend/agent.py — system prompt, model, tool declarations, schema generation."""

import pytest


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_model_contains_gemini(self):
        from backend.agent import MODEL
        assert "gemini" in MODEL.lower()

    def test_system_prompt_exists(self):
        from backend.agent import SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 100

    def test_tool_declarations_is_list(self):
        from backend.agent import TOOL_DECLARATIONS
        assert isinstance(TOOL_DECLARATIONS, list)
        assert len(TOOL_DECLARATIONS) >= 3

    def test_tool_registry_is_dict(self):
        from backend.agent import TOOL_REGISTRY
        assert isinstance(TOOL_REGISTRY, dict)
        assert len(TOOL_REGISTRY) >= 3

    def test_all_tools_in_both(self):
        from backend.agent import TOOL_DECLARATIONS, TOOL_REGISTRY
        decl_names = {d["name"] for d in TOOL_DECLARATIONS}
        assert decl_names == set(TOOL_REGISTRY.keys())

    def test_tools_include_project_overlay(self):
        from backend.agent import TOOL_REGISTRY
        assert "project_overlay" in TOOL_REGISTRY

    def test_tools_include_refresh_view(self):
        from backend.agent import TOOL_REGISTRY
        assert "refresh_view" in TOOL_REGISTRY

    def test_tools_include_show_scene(self):
        from backend.agent import TOOL_REGISTRY
        assert "show_scene" in TOOL_REGISTRY


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_contains_lumi(self):
        from backend.agent import SYSTEM_PROMPT
        assert "Lumi" in SYSTEM_PROMPT

    def test_contains_overlay_instructions(self):
        from backend.agent import SYSTEM_PROMPT
        assert "project_overlay" in SYSTEM_PROMPT

    def test_contains_coordinate_system(self):
        from backend.agent import SYSTEM_PROMPT
        assert "0-1000" in SYSTEM_PROMPT or "1000" in SYSTEM_PROMPT

    def test_contains_content_types(self):
        from backend.agent import SYSTEM_PROMPT
        for ct in ["graph", "annotation", "highlight", "markdown", "image"]:
            assert ct in SYSTEM_PROMPT.lower()

    def test_mentions_refresh_view(self):
        from backend.agent import SYSTEM_PROMPT
        assert "refresh_view" in SYSTEM_PROMPT

    def test_mentions_show_scene(self):
        from backend.agent import SYSTEM_PROMPT
        assert "show_scene" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# function_to_declaration
# ---------------------------------------------------------------------------


class TestFunctionToDeclaration:
    def test_simple_function(self):
        from backend.agent import function_to_declaration

        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number.
                b: Second number.
            """
            return a + b

        decl = function_to_declaration(add)
        assert decl["name"] == "add"
        assert "Add two numbers" in decl["description"]
        props = decl["parameters"]["properties"]
        assert props["a"]["type"] == "INTEGER"
        assert props["b"]["type"] == "INTEGER"
        assert set(decl["parameters"]["required"]) == {"a", "b"}

    def test_list_param(self):
        from backend.agent import function_to_declaration

        def foo(items: list[float]) -> dict:
            """Do something.

            Args:
                items: A list of floats.
            """
            return {}

        decl = function_to_declaration(foo)
        props = decl["parameters"]["properties"]
        assert props["items"]["type"] == "ARRAY"
        assert props["items"]["items"]["type"] == "NUMBER"

    def test_dict_param(self):
        from backend.agent import function_to_declaration

        def bar(data: dict) -> dict:
            """Process data.

            Args:
                data: Arbitrary data.
            """
            return {}

        decl = function_to_declaration(bar)
        assert decl["parameters"]["properties"]["data"]["type"] == "OBJECT"

    def test_optional_param_not_required(self):
        from backend.agent import function_to_declaration

        def baz(name: str, color: str = "blue") -> dict:
            """Baz.

            Args:
                name: The name.
                color: The color.
            """
            return {}

        decl = function_to_declaration(baz)
        assert decl["parameters"]["required"] == ["name"]

    def test_project_overlay_declaration(self):
        from backend.agent import TOOL_DECLARATIONS

        overlay_decl = next(d for d in TOOL_DECLARATIONS if d["name"] == "project_overlay")
        assert "placement" in overlay_decl["parameters"]["properties"]
        assert "content_type" in overlay_decl["parameters"]["properties"]
        assert overlay_decl["parameters"]["properties"]["placement"]["type"] == "ARRAY"
        assert set(overlay_decl["parameters"]["required"]) == {
            "content_type", "placement", "title", "data",
        }


# ---------------------------------------------------------------------------
# _parse_docstring_params
# ---------------------------------------------------------------------------


class TestParseDocstringParams:
    def test_multiline_description(self):
        from backend.agent import _parse_docstring_params

        doc = """Do something.

        Args:
            name: The name of the thing
                which can be very long.
            age: How old it is.

        Returns:
            A dict.
        """
        params = _parse_docstring_params(doc)
        assert "name" in params
        assert "long" in params["name"]
        assert "age" in params

    def test_empty_docstring(self):
        from backend.agent import _parse_docstring_params
        assert _parse_docstring_params("") == {}
        assert _parse_docstring_params(None) == {}
