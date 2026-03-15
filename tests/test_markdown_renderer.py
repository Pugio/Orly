"""Tests for the markdown overlay renderer."""

import numpy as np
import pytest

from client.renderer.markdown import _render_markdown_public as render_markdown


class TestRenderMarkdown:
    def test_output_shape(self):
        img = render_markdown("Hello world", width=640, height=480)
        assert img.shape == (480, 640, 3), "Expected BGR image with shape (H, W, 3)"

    def test_dtype_is_uint8(self):
        img = render_markdown("Hello world", width=640, height=480)
        assert img.dtype == np.uint8

    def test_background_is_mostly_black(self):
        img = render_markdown("Hi", width=640, height=480)
        pixel_brightness = img.sum(axis=2)
        black_fraction = (pixel_brightness < 30).mean()
        assert black_fraction > 0.5, f"Expected >50% black, got {black_fraction:.1%}"

    def test_content_exists(self):
        img = render_markdown("Hello World", width=640, height=480)
        pixel_brightness = img.sum(axis=2)
        bright_fraction = (pixel_brightness > 50).mean()
        assert bright_fraction > 0.001, "Expected some non-black pixels"

    def test_empty_text_produces_all_black(self):
        img = render_markdown("", width=400, height=200)
        assert img.max() == 0, "Empty text should produce all-black"

    def test_header_produces_larger_text(self):
        img_header = render_markdown("# Big Title", width=640, height=480)
        img_plain = render_markdown("small text", width=640, height=480)
        # Header should use more bright pixels (larger font)
        header_bright = (img_header.sum(axis=2) > 50).sum()
        plain_bright = (img_plain.sum(axis=2) > 50).sum()
        assert header_bright > plain_bright, "Header should produce more bright pixels than plain text"

    def test_bullet_list(self):
        md = "- First item\n- Second item\n- Third item"
        img = render_markdown(md, width=640, height=480)
        assert img.max() > 0, "Bullet list should produce visible content"

    def test_bold_text(self):
        img = render_markdown("**bold text**", width=640, height=480)
        assert img.max() > 0, "Bold text should produce visible content"

    def test_inline_bold_markers_stripped(self):
        """**markers** should not appear literally in the rendered image."""
        from client.renderer.markdown import _parse_line_segments
        segments = _parse_line_segments("hello **world** end")
        texts = [s["text"] for s in segments]
        for t in texts:
            assert "**" not in t, f"Bold markers should be stripped, got: {t}"
        # The bold segment should be flagged
        bold_segs = [s for s in segments if s["bold"]]
        assert len(bold_segs) == 1
        assert bold_segs[0]["text"] == "world"

    def test_inline_math_parsed(self):
        from client.renderer.markdown import _parse_line_segments
        segments = _parse_line_segments("solve $x^2 + 1 = 0$ please")
        math_segs = [s for s in segments if s["math"]]
        assert len(math_segs) == 1
        assert math_segs[0]["text"] == "$x^2 + 1 = 0$"

    def test_math_expression(self):
        img = render_markdown("The solution is $x^2 + 3x + 2$", width=640, height=480)
        assert img.max() > 0, "Math expression should produce visible content"

    def test_unsupported_latex_does_not_crash(self):
        """LaTeX commands like \\implies that matplotlib can't handle should not crash."""
        md = r"$x^2 + 3x + 2 = 0 \implies (x+1)(x+2) = 0$"
        img = render_markdown(md, width=640, height=480)
        assert img.shape == (480, 640, 3)
        assert img.max() > 0

    def test_complex_latex_does_not_crash(self):
        md = r"$\frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \Rightarrow x = \frac{-3 \pm 1}{2}$"
        img = render_markdown(md, width=640, height=480)
        assert img.shape == (480, 640, 3)
        assert img.max() > 0

    def test_sanitize_latex_strips_commands(self):
        from client.renderer.markdown import _sanitize_latex
        result = _sanitize_latex(r"$x \implies y$")
        assert "\\implies" not in result
        assert "x" in result and "y" in result

    def test_multiline_markdown(self):
        md = "# Step 1\n\nFirst, factor the expression:\n\n- $x^2 + 3x + 2 = (x+1)(x+2)$\n\nSo the roots are **x = -1** and **x = -2**."
        img = render_markdown(md, width=640, height=480)
        assert img.shape == (480, 640, 3)
        assert img.max() > 0

    def test_custom_dimensions(self):
        img = render_markdown("test", width=800, height=600)
        assert img.shape == (600, 800, 3)

    def test_long_text_wraps(self):
        long_text = "This is a very long paragraph that should be wrapped nicely across multiple lines without crashing or producing an empty image even though it is quite long."
        img = render_markdown(long_text, width=400, height=300)
        assert img.shape == (300, 400, 3)
        assert img.max() > 0
