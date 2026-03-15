"""Tests for backend WebSocket server message parsing and formatting."""

import base64

import pytest

from backend.main import (
    format_audio_response,
    format_interrupted,
    format_tool_result,
    format_transcript,
    parse_client_message,
)


# ---------------------------------------------------------------------------
# parse_client_message
# ---------------------------------------------------------------------------


class TestParseClientMessage:
    """Tests for parse_client_message."""

    def test_audio_message(self):
        raw = b"\x00\x01\x02\x03"
        msg = {"type": "audio", "data": base64.b64encode(raw).decode()}
        msg_type, payload = parse_client_message(msg)
        assert msg_type == "audio"
        assert payload == raw

    def test_video_message(self):
        raw = b"\xff\xd8\xff\xe0"  # JPEG magic bytes
        msg = {"type": "video", "data": base64.b64encode(raw).decode()}
        msg_type, payload = parse_client_message(msg)
        assert msg_type == "video"
        assert payload == raw

    def test_text_message(self):
        msg = {"type": "text", "text": "What is 2+2?"}
        msg_type, payload = parse_client_message(msg)
        assert msg_type == "text"
        assert payload == "What is 2+2?"

    def test_close_message(self):
        msg = {"type": "close"}
        msg_type, payload = parse_client_message(msg)
        assert msg_type == "close"
        assert payload is None

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="type"):
            parse_client_message({"data": "abc"})

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            parse_client_message({"type": "invalid"})

    def test_audio_missing_data_raises(self):
        with pytest.raises(ValueError, match="data"):
            parse_client_message({"type": "audio"})

    def test_video_missing_data_raises(self):
        with pytest.raises(ValueError, match="data"):
            parse_client_message({"type": "video"})


# ---------------------------------------------------------------------------
# format_audio_response
# ---------------------------------------------------------------------------


class TestFormatAudioResponse:
    """Tests for format_audio_response."""

    def test_structure(self):
        result = format_audio_response(b"\x00\x01")
        assert result["type"] == "audio"
        assert "data" in result

    def test_round_trip(self):
        original = b"\xde\xad\xbe\xef" * 100
        result = format_audio_response(original)
        decoded = base64.b64decode(result["data"])
        assert decoded == original


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
