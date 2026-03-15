"""Extended tests for client/renderer/markdown.py internals and edge cases."""

import numpy as np
import pytest

from client.renderer.markdown import (
    _parse_line_segments,
    _parse_markdown,
    _sanitize_latex,
    _wrap_segments,
    render_markdown,
)


# ---------------------------------------------------------------------------
# _parse_line_segments
# ---------------------------------------------------------------------------


class TestParseLineSegments:
    def test_plain_text(self):
        segs = _parse_line_segments("hello world")
        assert len(segs) == 1
        assert segs[0]["text"] == "hello world"
        assert segs[0]["bold"] is False
        assert segs[0]["math"] is False

    def test_bold_text(self):
        segs = _parse_line_segments("**bold**")
        assert len(segs) == 1
        assert segs[0]["text"] == "bold"
        assert segs[0]["bold"] is True

    def test_math_text(self):
        segs = _parse_line_segments("$x^2$")
        assert len(segs) == 1
        assert segs[0]["text"] == "$x^2$"
        assert segs[0]["math"] is True

    def test_mixed_segments(self):
        segs = _parse_line_segments("Start **bold** then $math$ end")
        texts = [s["text"] for s in segs]
        assert "Start " in texts
        assert "bold" in texts
        assert " then " in texts
        assert "$math$" in texts
        assert " end" in texts

    def test_multiple_bold(self):
        segs = _parse_line_segments("**a** and **b**")
        bold_segs = [s for s in segs if s["bold"]]
        assert len(bold_segs) == 2
        assert bold_segs[0]["text"] == "a"
        assert bold_segs[1]["text"] == "b"

    def test_adjacent_bold_math(self):
        segs = _parse_line_segments("**x** = $y$")
        assert any(s["bold"] and s["text"] == "x" for s in segs)
        assert any(s["math"] and s["text"] == "$y$" for s in segs)

    def test_empty_string(self):
        segs = _parse_line_segments("")
        assert segs == []

    def test_bold_with_spaces(self):
        segs = _parse_line_segments("**hello world**")
        assert len(segs) == 1
        assert segs[0]["text"] == "hello world"
        assert segs[0]["bold"] is True

    def test_nested_markers_not_supported(self):
        """Nested **$x$** is not expected to parse perfectly, just not crash."""
        segs = _parse_line_segments("**$x$**")
        assert len(segs) > 0  # some segments produced


# ---------------------------------------------------------------------------
# _parse_markdown
# ---------------------------------------------------------------------------


class TestParseMarkdown:
    def test_h1_header(self):
        blocks = _parse_markdown("# Title")
        assert len(blocks) == 1
        assert blocks[0]["style"] == "h1"
        assert blocks[0]["segments"][0]["text"] == "Title"

    def test_h2_header(self):
        blocks = _parse_markdown("## Subtitle")
        assert len(blocks) == 1
        assert blocks[0]["style"] == "h2"

    def test_body_text(self):
        blocks = _parse_markdown("Just text")
        assert blocks[0]["style"] == "body"

    def test_bullet_item(self):
        blocks = _parse_markdown("- Item one")
        assert blocks[0]["style"] == "bullet"
        # First segment should be the bullet character
        assert "\u2022" in blocks[0]["segments"][0]["text"]

    def test_spacer_from_empty_line(self):
        blocks = _parse_markdown("Line 1\n\nLine 2")
        styles = [b["style"] for b in blocks]
        assert "spacer" in styles

    def test_mixed_markdown(self):
        md = "# Title\n\nSome text\n\n- Bullet 1\n- Bullet 2\n\n## Section"
        blocks = _parse_markdown(md)
        styles = [b["style"] for b in blocks]
        assert "h1" in styles
        assert "body" in styles
        assert "bullet" in styles
        assert "h2" in styles
        assert "spacer" in styles

    def test_empty_string(self):
        blocks = _parse_markdown("")
        assert len(blocks) == 1
        assert blocks[0]["style"] == "spacer"

    def test_h1_with_bold(self):
        blocks = _parse_markdown("# **Bold Title**")
        assert blocks[0]["style"] == "h1"
        bold_segs = [s for s in blocks[0]["segments"] if s["bold"]]
        assert len(bold_segs) == 1

    def test_bullet_with_math(self):
        blocks = _parse_markdown("- solve $x^2=4$")
        assert blocks[0]["style"] == "bullet"
        math_segs = [s for s in blocks[0]["segments"] if s["math"]]
        assert len(math_segs) == 1


# ---------------------------------------------------------------------------
# _sanitize_latex
# ---------------------------------------------------------------------------


class TestSanitizeLatex:
    def test_strips_dollar_signs(self):
        result = _sanitize_latex("$x^2$")
        assert not result.startswith("$")
        assert not result.endswith("$")
        assert "x^2" in result

    def test_replaces_implies(self):
        result = _sanitize_latex(r"$a \implies b$")
        assert "\\implies" not in result
        assert "=>" in result

    def test_replaces_rightarrow(self):
        result = _sanitize_latex(r"$a \Rightarrow b$")
        assert "\\Rightarrow" not in result

    def test_replaces_pm(self):
        result = _sanitize_latex(r"$a \pm b$")
        assert "+/-" in result

    def test_strips_unknown_commands(self):
        result = _sanitize_latex(r"$\unknowncommand x$")
        assert "\\unknowncommand" not in result
        assert "x" in result

    def test_removes_braces(self):
        result = _sanitize_latex(r"${a}{b}$")
        assert "{" not in result
        assert "}" not in result

    def test_removes_frac(self):
        result = _sanitize_latex(r"$\frac{a}{b}$")
        assert "\\frac" not in result
        assert "a" in result
        assert "b" in result

    def test_plain_text_passthrough(self):
        result = _sanitize_latex("hello")
        assert result == "hello"

    def test_collapses_whitespace(self):
        result = _sanitize_latex(r"$a  \quad  b$")
        # Should not have excessive whitespace
        assert "  " not in result or result.count("  ") <= 1


# ---------------------------------------------------------------------------
# _wrap_segments
# ---------------------------------------------------------------------------


class TestWrapSegments:
    def test_short_line_no_wrap(self):
        segs = [{"text": "hello", "bold": False, "math": False}]
        result = _wrap_segments(segs, max_chars=50)
        assert len(result) == 1

    def test_long_line_wraps(self):
        segs = [{"text": "this is a much longer line that should wrap", "bold": False, "math": False}]
        result = _wrap_segments(segs, max_chars=15)
        assert len(result) > 1

    def test_exact_length_no_wrap(self):
        segs = [{"text": "12345", "bold": False, "math": False}]
        result = _wrap_segments(segs, max_chars=5)
        assert len(result) == 1

    def test_single_long_word(self):
        segs = [{"text": "a" * 50, "bold": False, "math": False}]
        result = _wrap_segments(segs, max_chars=10)
        # Single word can't be split further
        assert len(result) >= 1

    def test_preserves_all_text(self):
        text = "hello world foo bar"
        segs = [{"text": text, "bold": False, "math": False}]
        result = _wrap_segments(segs, max_chars=10)
        combined = " ".join(s["text"] for line in result for s in line)
        assert "hello" in combined
        assert "bar" in combined

    def test_empty_segments(self):
        result = _wrap_segments([], max_chars=50)
        assert result == [[]]


# ---------------------------------------------------------------------------
# render_markdown — additional
# ---------------------------------------------------------------------------


class TestRenderMarkdownExtended:
    def test_only_whitespace_produces_black(self):
        img = render_markdown("   \n\n   ", width=400, height=200)
        assert img.max() == 0

    def test_multiple_headers(self):
        md = "# Header 1\n## Header 2\n# Header 3"
        img = render_markdown(md, width=640, height=480)
        assert img.max() > 0

    def test_deeply_nested_bullets(self):
        """Nested bullets should not crash (rendered as flat bullets)."""
        md = "- Level 1\n  - Level 2\n    - Level 3"
        img = render_markdown(md, width=640, height=480)
        assert img.shape == (480, 640, 3)

    def test_special_characters(self):
        md = "Special chars: & < > @ # %"
        img = render_markdown(md, width=640, height=480)
        assert img.max() > 0

    def test_latex_only(self):
        img = render_markdown("$x = \\frac{-b}{2a}$", width=640, height=480)
        assert img.max() > 0
        assert img.shape == (480, 640, 3)

    def test_very_small_canvas(self):
        img = render_markdown("# Big Header", width=50, height=50)
        assert img.shape == (50, 50, 3)

    def test_many_lines_truncated_gracefully(self):
        """Renderer should stop drawing when it runs out of space."""
        md = "\n".join(f"- Item {i}" for i in range(100))
        img = render_markdown(md, width=640, height=480)
        assert img.shape == (480, 640, 3)
        assert img.max() > 0

    def test_fallback_on_bad_input(self):
        """Even if internal rendering fails, should produce a valid image."""
        # Extremely long single line
        md = "x" * 10000
        img = render_markdown(md, width=640, height=480)
        assert img.shape == (480, 640, 3)
