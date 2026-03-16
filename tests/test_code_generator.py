"""Tests for async code generation via Gemini 3 Flash."""

import threading
import time

import pytest
from unittest.mock import MagicMock, patch

from client.code_generator import CodeGenerator, extract_code
from client.session_store import SessionStore


# ---------------------------------------------------------------------------
# extract_code tests
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_plain_code(self):
        """Plain Python code returned as-is."""
        code = "x = 1\nprint(x)"
        assert extract_code(code) == code

    def test_markdown_python_fence(self):
        """Extracts code from ```python ... ``` blocks."""
        text = "Here's the code:\n```python\nx = 1\nprint(x)\n```\nDone!"
        assert extract_code(text) == "x = 1\nprint(x)"

    def test_markdown_bare_fence(self):
        """Extracts code from ``` ... ``` blocks without language tag."""
        text = "```\nx = 42\n```"
        assert extract_code(text) == "x = 42"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert extract_code("") == ""

    def test_whitespace_stripping(self):
        """Leading/trailing whitespace stripped."""
        assert extract_code("  \n  x = 1  \n  ") == "x = 1"


# ---------------------------------------------------------------------------
# CodeGenerator tests
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return SessionStore(session_dir=str(tmp_path / "session"))


@pytest.fixture
def notifications():
    """Collect notifications in a list."""
    return []


@pytest.fixture
def generator(store, notifications):
    def notify(msg):
        notifications.append(msg)

    def validate(code):
        # Simple validation: just check syntax
        try:
            compile(code, "<test>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    return CodeGenerator(
        session_store=store,
        validate_fn=validate,
        notify_fn=notify,
    )


class TestCodeGenerator:
    def test_successful_generation(self, generator, store, notifications):
        """Happy path: generate, validate, save, notify."""
        mock_response = MagicMock()
        mock_response.text = "x = 1\ntable.notify('done')"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            generator._generate_code_thread("my-prog", "do something", "")

        # Code saved
        code = store.load_program("my-prog")
        assert code is not None
        assert "x = 1" in code

        # Notification sent
        assert len(notifications) == 1
        assert "generated and saved" in notifications[0]
        assert "my-prog" in notifications[0]

    def test_generation_with_markdown_fences(self, generator, store, notifications):
        """Code wrapped in markdown fences is extracted correctly."""
        mock_response = MagicMock()
        mock_response.text = "```python\ny = 42\n```"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            generator._generate_code_thread("fenced", "compute y", "")

        code = store.load_program("fenced")
        assert code == "y = 42"

    def test_empty_response_notifies_failure(self, generator, store, notifications):
        """Empty model response sends failure notification."""
        mock_response = MagicMock()
        mock_response.text = ""

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            generator._generate_code_thread("empty", "do nothing", "")

        assert store.load_program("empty") is None
        assert len(notifications) == 1
        assert "failed" in notifications[0]

    def test_validation_failure_retries(self, generator, store, notifications):
        """Invalid code triggers retry; if retry succeeds, code is saved."""
        bad_response = MagicMock()
        bad_response.text = "def f(:"  # syntax error

        good_response = MagicMock()
        good_response.text = "x = 1"

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [bad_response, good_response]

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            generator._generate_code_thread("retry-prog", "fix it", "")

        code = store.load_program("retry-prog")
        assert code == "x = 1"
        assert len(notifications) == 1
        assert "generated and saved" in notifications[0]

    def test_validation_failure_both_attempts(self, generator, store, notifications):
        """Both attempts produce invalid code — failure notification."""
        bad1 = MagicMock()
        bad1.text = "def f(:"
        bad2 = MagicMock()
        bad2.text = "def g(:"

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [bad1, bad2]

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            generator._generate_code_thread("bad-prog", "broken", "")

        assert store.load_program("bad-prog") is None
        assert len(notifications) == 1
        assert "failed validation" in notifications[0]

    def test_api_exception_notifies_failure(self, generator, store, notifications):
        """API exception sends failure notification."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API down")

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            generator._generate_code_thread("err-prog", "crash", "")

        assert store.load_program("err-prog") is None
        assert len(notifications) == 1
        assert "failed" in notifications[0]
        assert "API down" in notifications[0]

    def test_generate_async_spawns_thread(self, generator):
        """generate_async starts a daemon thread."""
        with patch.object(generator, "_generate_code_thread") as mock_thread:
            generator.generate_async("test", "desc")
            # Give the thread a moment to start
            time.sleep(0.1)
            mock_thread.assert_called_once_with("test", "desc", "")

    def test_context_included_in_prompt(self, generator, notifications):
        """Context string is passed to the model."""
        mock_response = MagicMock()
        mock_response.text = "x = 1"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            generator._generate_code_thread("ctx-prog", "do it", "there's a cat on the table")

        call_args = mock_client.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents") or call_args[0][0]
        assert "cat on the table" in prompt
