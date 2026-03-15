"""Tests for Feature 10: Multi-subject support."""
import pytest


class TestMultiSubjectPrompt:
    def test_prompt_mentions_science(self):
        from backend.agent import SYSTEM_PROMPT
        assert "science" in SYSTEM_PROMPT.lower()

    def test_prompt_mentions_language(self):
        from backend.agent import SYSTEM_PROMPT
        assert "language" in SYSTEM_PROMPT.lower()

    def test_prompt_mentions_history(self):
        from backend.agent import SYSTEM_PROMPT
        assert "history" in SYSTEM_PROMPT.lower()

    def test_prompt_mentions_subject_awareness(self):
        from backend.agent import SYSTEM_PROMPT
        assert "SUBJECT" in SYSTEM_PROMPT
