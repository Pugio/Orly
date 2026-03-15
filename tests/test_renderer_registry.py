"""Tests for the renderer registry — auto-discovery of SPEC dicts."""

import numpy as np
import pytest


# All renderer types that should be discovered (image is registered but special-cased).
_EXPECTED_TYPES = {
    "graph", "annotation", "highlight", "markdown", "image",
    "number_line", "geometry", "chemistry", "steps", "flashcard",
}

# Minimal valid data dicts for each renderer type (for smoke-testing render calls).
_MINIMAL_DATA = {
    "graph": {"expression": "x", "x_range": [-1, 1], "y_range": [-1, 1]},
    "annotation": {"text": "hello"},
    "highlight": {"color": "#ff0000"},
    "markdown": {"text": "# Title"},
    "number_line": {"min_val": 0, "max_val": 5, "points": [], "ranges": []},
    "geometry": {"elements": [], "x_range": [-5, 5], "y_range": [-5, 5]},
    "chemistry": {"atoms": [{"symbol": "H", "pos": [0, 0]}], "bonds": []},
    "steps": {"steps": [{"title": "S1", "content": "c"}], "visible_count": 1},
    "flashcard": {"front": "Q", "back": "A", "show_back": False},
}


class TestRegistryDiscovery:
    def test_registry_discovers_all_existing_renderers(self):
        from client.renderer.registry import all_specs, _REGISTRY
        _REGISTRY.clear()  # force re-discovery
        names = {s["name"] for s in all_specs()}
        assert names == _EXPECTED_TYPES

    def test_registry_get_known_type(self):
        from client.renderer.registry import get
        spec = get("graph")
        assert spec is not None
        assert spec["name"] == "graph"
        assert "render" in spec
        assert "description" in spec
        assert "data_format" in spec

    def test_registry_get_unknown_type(self):
        from client.renderer.registry import get
        assert get("nonexistent") is None

    def test_valid_types_matches_specs(self):
        from client.renderer.registry import valid_types, all_specs
        assert valid_types() == {s["name"] for s in all_specs()}

    def test_data_format_docs_contains_all_types(self):
        from client.renderer.registry import data_format_docs, all_specs
        docs = data_format_docs()
        for spec in all_specs():
            assert spec["name"] in docs

    def test_prompt_overlay_docs_contains_all_types(self):
        from client.renderer.registry import prompt_overlay_docs, all_specs
        docs = prompt_overlay_docs()
        for spec in all_specs():
            assert spec["name"] in docs
            assert spec["description"] in docs


class TestSpecRenderFunctions:
    def test_spec_render_functions_callable(self):
        from client.renderer.registry import all_specs
        for spec in all_specs():
            assert callable(spec["render"]), f"{spec['name']} render not callable"

    @pytest.mark.parametrize("type_name", sorted(_EXPECTED_TYPES - {"image"}))
    def test_spec_render_returns_valid_image(self, type_name):
        from client.renderer.registry import get
        spec = get(type_name)
        assert spec is not None
        data = _MINIMAL_DATA[type_name]
        result = spec["render"](data, 200, 200)
        assert isinstance(result, np.ndarray)
        assert result.shape == (200, 200, 3)
        assert result.dtype == np.uint8


class TestRegistryIntegration:
    def test_valid_types_used_in_tool_validation(self):
        """Registry valid_types should match what backend/tools.py accepts."""
        from client.renderer.registry import valid_types
        # The registry should contain at least all the types the tool validates.
        expected = {"graph", "annotation", "highlight", "markdown", "image",
                    "number_line", "steps", "geometry", "chemistry", "flashcard"}
        assert valid_types() >= expected
