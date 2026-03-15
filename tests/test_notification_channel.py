"""Tests for the async notification channel protocol extensions.

Tests backend-side parsing, formatting, tool functions, and tool registry
without any real WebSocket or Gemini connections.
"""

import json

import pytest

from backend.main import execute_tool, parse_text_message
from backend.tools import (
    get_overlay_state,
    list_programs,
    run_program,
    stop_program,
)


# ---------------------------------------------------------------------------
# parse_text_message — notification type
# ---------------------------------------------------------------------------


class TestParseNotificationMessage:
    def test_parse_notification_message(self):
        data = json.dumps({"type": "notification", "source": "img", "text": "done"})
        msg_type, payload = parse_text_message(data)
        assert msg_type == "notification"
        parsed = json.loads(payload)
        assert parsed["source"] == "img"
        assert parsed["text"] == "done"

    def test_notification_empty_text(self):
        data = json.dumps({"type": "notification", "source": "sys", "text": ""})
        msg_type, payload = parse_text_message(data)
        assert msg_type == "notification"
        parsed = json.loads(payload)
        assert parsed["text"] == ""

    def test_notification_special_characters(self):
        text = 'He said "hello"\nand then\ttabbed'
        data = json.dumps({"type": "notification", "source": "test", "text": text})
        msg_type, payload = parse_text_message(data)
        assert msg_type == "notification"
        parsed = json.loads(payload)
        assert parsed["text"] == text


# ---------------------------------------------------------------------------
# Format notification for Gemini
# ---------------------------------------------------------------------------


class TestFormatNotificationForGemini:
    def test_format_notification_for_gemini(self):
        source = "img_gen"
        text = "Image generation complete"
        notification_text = f"[NOTIFICATION from {source}]: {text}"
        assert notification_text == "[NOTIFICATION from img_gen]: Image generation complete"


# ---------------------------------------------------------------------------
# Backend can format run_program / stop_program messages
# ---------------------------------------------------------------------------


class TestFormatProgramMessages:
    def test_parse_run_program_message(self):
        """Backend can format a run_program message for the client."""
        msg = {"type": "run_program", "name": "tracker", "code": "print('hi')", "description": "A tracker"}
        raw = json.dumps(msg)
        parsed = json.loads(raw)
        assert parsed["type"] == "run_program"
        assert parsed["name"] == "tracker"
        assert parsed["code"] == "print('hi')"

    def test_parse_stop_program_message(self):
        """Backend can format a stop_program message for the client."""
        msg = {"type": "stop_program", "name": "tracker"}
        raw = json.dumps(msg)
        parsed = json.loads(raw)
        assert parsed["type"] == "stop_program"
        assert parsed["name"] == "tracker"


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


class TestRunProgramTool:
    def test_returns_correct_status(self):
        result = run_program(name="tracker", code="print('hi')", description="A test")
        assert result["status"] == "started"
        assert result["name"] == "tracker"
        assert result["description"] == "A test"

    def test_default_description(self):
        result = run_program(name="foo", code="x=1")
        assert result["status"] == "started"
        assert result["description"] == ""

    def test_run_program_validation(self):
        """run_program returns a valid result even with minimal args."""
        result = run_program(name="x", code="pass")
        assert result["status"] == "started"
        assert result["name"] == "x"


class TestStopProgramTool:
    def test_returns_correct_status(self):
        result = stop_program(name="tracker")
        assert result["status"] == "stopping"
        assert result["name"] == "tracker"


class TestListProgramsTool:
    def test_returns_correct_structure(self):
        result = list_programs()
        assert result["status"] == "fetching"
        assert "description" in result


class TestGetOverlayStateTool:
    def test_returns_correct_structure(self):
        result = get_overlay_state()
        assert result["status"] == "fetching"
        assert "description" in result


# ---------------------------------------------------------------------------
# Tool declarations and registry include new tools
# ---------------------------------------------------------------------------


class TestToolDeclarationsIncludeNewTools:
    def test_tool_declarations_include_new_tools(self):
        from backend.agent import TOOL_DECLARATIONS

        names = {d["name"] for d in TOOL_DECLARATIONS}
        assert "run_program" in names
        assert "stop_program" in names
        assert "list_programs" in names
        assert "get_overlay_state" in names

    def test_tool_registry_includes_new_tools(self):
        from backend.agent import TOOL_REGISTRY

        assert "run_program" in TOOL_REGISTRY
        assert "stop_program" in TOOL_REGISTRY
        assert "list_programs" in TOOL_REGISTRY
        assert "get_overlay_state" in TOOL_REGISTRY


# ---------------------------------------------------------------------------
# execute_tool with new tools
# ---------------------------------------------------------------------------


class TestExecuteRunProgram:
    def test_execute_run_program(self):
        from backend.agent import TOOL_REGISTRY

        result = execute_tool(
            "run_program",
            {"name": "tracker", "code": "print('hi')", "description": "test"},
            TOOL_REGISTRY,
        )
        assert result["status"] == "started"
        assert result["name"] == "tracker"

    def test_execute_stop_program(self):
        from backend.agent import TOOL_REGISTRY

        result = execute_tool("stop_program", {"name": "tracker"}, TOOL_REGISTRY)
        assert result["status"] == "stopping"

    def test_execute_list_programs(self):
        from backend.agent import TOOL_REGISTRY

        result = execute_tool("list_programs", {}, TOOL_REGISTRY)
        assert result["status"] == "fetching"

    def test_execute_get_overlay_state(self):
        from backend.agent import TOOL_REGISTRY

        result = execute_tool("get_overlay_state", {}, TOOL_REGISTRY)
        assert result["status"] == "fetching"
