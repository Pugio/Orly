"""Tests for backend WebSocket server message parsing and formatting."""

import json

import pytest

from backend.main import (
    PREFIX_AUDIO_IN,
    PREFIX_AUDIO_OUT,
    PREFIX_VIDEO_IN,
    execute_tool,
    format_audio_response,
    format_interrupted,
    format_tool_result,
    format_transcript,
    parse_binary_message,
    parse_text_message,
)


# ---------------------------------------------------------------------------
# parse_binary_message
# ---------------------------------------------------------------------------


class TestParseBinaryMessage:
    """Tests for parse_binary_message."""

    def test_audio_message(self):
        raw = b"\x00\x01\x02\x03"
        data = PREFIX_AUDIO_IN + raw
        msg_type, payload = parse_binary_message(data)
        assert msg_type == "audio"
        assert payload == raw

    def test_video_message(self):
        raw = b"\xff\xd8\xff\xe0"  # JPEG magic bytes
        data = PREFIX_VIDEO_IN + raw
        msg_type, payload = parse_binary_message(data)
        assert msg_type == "video"
        assert payload == raw

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown binary prefix"):
            parse_binary_message(b"\xff\x00\x01")

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            parse_binary_message(b"\x01")


# ---------------------------------------------------------------------------
# parse_text_message
# ---------------------------------------------------------------------------


class TestParseTextMessage:
    """Tests for parse_text_message."""

    def test_text_message(self):
        data = json.dumps({"type": "text", "text": "What is 2+2?"})
        msg_type, payload = parse_text_message(data)
        assert msg_type == "text"
        assert payload == "What is 2+2?"

    def test_close_message(self):
        data = json.dumps({"type": "close"})
        msg_type, payload = parse_text_message(data)
        assert msg_type == "close"
        assert payload is None

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="type"):
            parse_text_message(json.dumps({"data": "abc"}))

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            parse_text_message(json.dumps({"type": "audio"}))


# ---------------------------------------------------------------------------
# format_audio_response
# ---------------------------------------------------------------------------


class TestFormatAudioResponse:
    """Tests for format_audio_response."""

    def test_returns_bytes(self):
        result = format_audio_response(b"\x00\x01")
        assert isinstance(result, bytes)

    def test_has_prefix(self):
        result = format_audio_response(b"\x00\x01")
        assert result[0:1] == PREFIX_AUDIO_OUT

    def test_round_trip(self):
        original = b"\xde\xad\xbe\xef" * 100
        result = format_audio_response(original)
        assert result[1:] == original


# ---------------------------------------------------------------------------
# format_transcript
# ---------------------------------------------------------------------------


class TestFormatTranscript:
    """Tests for format_transcript."""

    def test_input_transcript(self):
        result = format_transcript("in", "hello world")
        assert result["type"] == "transcript_in"
        assert result["text"] == "hello world"

    def test_output_transcript(self):
        result = format_transcript("out", "The answer is 4.")
        assert result["type"] == "transcript_out"
        assert result["text"] == "The answer is 4."


# ---------------------------------------------------------------------------
# format_tool_result
# ---------------------------------------------------------------------------


class TestFormatToolResult:
    """Tests for format_tool_result."""

    def test_structure(self):
        response = {"status": "displayed", "content_type": "graph"}
        result = format_tool_result("project_overlay", response)
        assert result["type"] == "tool_result"
        assert result["name"] == "project_overlay"
        assert result["result"] == response


# ---------------------------------------------------------------------------
# format_interrupted
# ---------------------------------------------------------------------------


class TestFormatInterrupted:
    """Tests for format_interrupted."""

    def test_structure(self):
        result = format_interrupted()
        assert result == {"type": "interrupted"}


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------


class TestExecuteTool:
    """Tests for execute_tool."""

    def test_calls_registered_function(self):
        def greet(name: str) -> dict:
            return {"greeting": f"Hello, {name}!"}

        registry = {"greet": greet}
        result = execute_tool("greet", {"name": "Alice"}, registry)
        assert result == {"greeting": "Hello, Alice!"}

    def test_strips_unexpected_args(self):
        def greet(name: str) -> dict:
            return {"greeting": f"Hello, {name}!"}

        registry = {"greet": greet}
        result = execute_tool(
            "greet", {"name": "Bob", "extra_junk": 42}, registry
        )
        assert result == {"greeting": "Hello, Bob!"}

    def test_unknown_tool_returns_error(self):
        result = execute_tool("nonexistent", {}, {})
        assert result["status"] == "error"
        assert "Unknown tool" in result["message"]

    def test_exception_returns_error(self):
        def bad_tool() -> dict:
            raise RuntimeError("boom")

        registry = {"bad_tool": bad_tool}
        result = execute_tool("bad_tool", {}, registry)
        assert result["status"] == "error"
        assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# Tool schema generation (from backend.agent)
# ---------------------------------------------------------------------------


class TestFunctionToDeclaration:
    """Tests for function_to_declaration auto-schema generation."""

    def test_simple_function(self):
        from backend.agent import function_to_declaration

        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: First number.
                b: Second number.

            Returns:
                The sum.
            """
            return a + b

        decl = function_to_declaration(add)
        assert decl["name"] == "add"
        assert "Add two numbers" in decl["description"]
        props = decl["parameters"]["properties"]
        assert props["a"]["type"] == "INTEGER"
        assert props["b"]["type"] == "INTEGER"
        assert "First number" in props["a"]["description"]
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
        props = decl["parameters"]["properties"]
        assert props["data"]["type"] == "OBJECT"

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
        """Verify the actual project_overlay tool produces a valid schema."""
        from backend.agent import TOOL_DECLARATIONS

        overlay_decl = next(
            d for d in TOOL_DECLARATIONS if d["name"] == "project_overlay"
        )
        assert "placement" in overlay_decl["parameters"]["properties"]
        assert "content_type" in overlay_decl["parameters"]["properties"]
        assert overlay_decl["parameters"]["properties"]["placement"]["type"] == "ARRAY"
        assert set(overlay_decl["parameters"]["required"]) == {
            "content_type",
            "placement",
            "title",
            "data",
        }

    def test_tool_registry_has_all_tools(self):
        from backend.agent import TOOL_DECLARATIONS, TOOL_REGISTRY

        decl_names = {d["name"] for d in TOOL_DECLARATIONS}
        assert decl_names == set(TOOL_REGISTRY.keys())
        assert "project_overlay" in decl_names
        assert "refresh_view" in decl_names
        assert "show_scene" in decl_names


# ---------------------------------------------------------------------------
# Docstring param parsing
# ---------------------------------------------------------------------------


class TestParseDocstringParams:
    """Tests for _parse_docstring_params."""

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
        assert "old" in params["age"]

    def test_empty_docstring(self):
        from backend.agent import _parse_docstring_params

        assert _parse_docstring_params("") == {}
        assert _parse_docstring_params(None) == {}


# ---------------------------------------------------------------------------
# FastAPI app existence & route check
# ---------------------------------------------------------------------------


class TestAppExists:
    """Verify the FastAPI app is importable and has the expected route."""

    def test_app_is_fastapi(self):
        from backend.main import app

        assert app is not None
        # FastAPI apps have a .routes attribute
        assert hasattr(app, "routes")

    def test_websocket_route_registered(self):
        from backend.main import app

        ws_paths = [
            r.path for r in app.routes if hasattr(r, "path") and "/ws/" in r.path
        ]
        assert "/ws/session" in ws_paths
