"""Render markdown text on black backgrounds for projector overlay.

Uses matplotlib for text rendering with support for:
- Headers (# / ##)
- Inline bold (**text**)
- Bullet lists (- item)
- Inline LaTeX math ($expr$)
- Plain text with word wrapping

All font sizes are tuned for low-resolution projector readability.
"""

import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Regex to split a line into segments: bold, math, or plain text.
# Matches **bold** or $math$ and captures them as groups.
_INLINE_RE = re.compile(r'(\*\*.*?\*\*|\$.*?\$)')

# Common LaTeX commands → readable plain text replacements.
_LATEX_REPLACEMENTS = [
    (r'\implies', ' => '),
    (r'\therefore', ' ∴ '),
    (r'\Rightarrow', ' => '),
    (r'\rightarrow', ' -> '),
    (r'\leftarrow', ' <- '),
    (r'\leq', ' <= '),
    (r'\geq', ' >= '),
    (r'\neq', ' != '),
    (r'\approx', ' ~ '),
    (r'\times', ' x '),
    (r'\cdot', ' . '),
    (r'\pm', ' +/- '),
    (r'\infty', 'inf'),
    (r'\sqrt', 'sqrt'),
    (r'\frac', ''),
    (r'\left', ''),
    (r'\right', ''),
    (r'\quad', '  '),
    (r'\qquad', '    '),
    (r'\,', ' '),
    (r'\;', ' '),
    (r'\!', ''),
]


def _sanitize_latex(text: str) -> str:
    """Convert LaTeX math to readable plain text.

    Matplotlib's mathtext only supports a small subset of LaTeX.
    Instead of risking crashes, strip $ delimiters and replace
    common LaTeX commands with Unicode/ASCII equivalents.
    """
    # Strip $ delimiters
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1]

    # Replace known commands
    for cmd, replacement in _LATEX_REPLACEMENTS:
        text = text.replace(cmd, replacement)

    # Strip any remaining \command sequences
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Clean up braces used for grouping
    text = text.replace('{', '').replace('}', '')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _strip_inline_markers(text: str) -> str:
    """Strip **bold** and $math$ markers, returning plain display text."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text)
    return text


def _parse_line_segments(line: str) -> list[dict]:
    """Split a line into segments with inline formatting.

    Returns list of {"text": str, "bold": bool, "math": bool}.
    """
    parts = _INLINE_RE.split(line)
    segments = []
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            segments.append({"text": part[2:-2], "bold": True, "math": False})
        elif part.startswith("$") and part.endswith("$"):
            # Keep $ delimiters for matplotlib mathtext rendering
            segments.append({"text": part, "bold": False, "math": True})
        else:
            segments.append({"text": part, "bold": False, "math": False})
    return segments


def _parse_markdown(text: str) -> list[dict]:
    """Parse markdown into a list of styled blocks.

    Each block is a dict with keys:
        segments: list of {"text", "bold", "math"} dicts
        style: "h1", "h2", "body", "bullet", "spacer"
    """
    blocks: list[dict] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            blocks.append({"segments": [], "style": "spacer"})
            continue

        if stripped.startswith("## "):
            content = stripped[3:]
            blocks.append({"segments": _parse_line_segments(content), "style": "h2"})
        elif stripped.startswith("# "):
            content = stripped[2:]
            blocks.append({"segments": _parse_line_segments(content), "style": "h1"})
        elif stripped.startswith("- "):
            content = stripped[2:]
            segs = [{"text": "  \u2022  ", "bold": False, "math": False}]
            segs.extend(_parse_line_segments(content))
            blocks.append({"segments": segs, "style": "bullet"})
        else:
            blocks.append({"segments": _parse_line_segments(stripped), "style": "body"})
    return blocks


# Font sizes tuned for 1280x720 projector at typical viewing distance.
_STYLE_FONTSIZE = {
    "h1": 36,
    "h2": 28,
    "body": 22,
    "bullet": 22,
    "bold": 22,
}

_BASE_COLOR = "#00ffff"  # cyan — visible on projector over white paper


def _flat_text(segments: list[dict]) -> str:
    """Get plain text from segments for wrapping calculation."""
    return "".join(
        s["text"].strip("$") if s["math"] else s["text"]
        for s in segments
    )


def _wrap_segments(segments: list[dict], max_chars: int) -> list[list[dict]]:
    """Word-wrap segments into lines that fit within max_chars.

    Returns a list of lines, each line being a list of segments.
    """
    flat = _flat_text(segments)
    if len(flat) <= max_chars:
        return [segments]

    # Simple approach: join all text, wrap, then re-split into segments per line.
    # This loses per-word formatting boundaries but is reliable.
    words = flat.split()
    lines_text: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip() if current else word
        if len(test) > max_chars and current:
            lines_text.append(current)
            current = word
        else:
            current = test
    if current:
        lines_text.append(current)

    # For wrapped lines, return plain segments (formatting is approximate)
    return [
        [{"text": line, "bold": False, "math": False}]
        for line in lines_text
    ]


def render_markdown(
    text: str,
    width: int,
    height: int,
) -> np.ndarray:
    """Render markdown text on a black background.

    Args:
        text: Markdown-formatted text. Supports # headers, **bold**,
              - bullet lists, and $latex$ math.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if not text or not text.strip():
        return img

    try:
        return _render_markdown_impl(text, width, height)
    except Exception as e:
        # If rendering fails (e.g. unsupported LaTeX), fall back to
        # plain text annotation so the client never crashes.
        import logging
        logging.getLogger(__name__).warning("Falling back to annotation: %s", e)
        from client.renderer.annotation import render_annotation
        # Strip all markdown/LaTeX for the fallback
        plain = re.sub(r'[#*$`]', '', text)
        plain = re.sub(r'\\[a-zA-Z]+', '', plain)
        return render_annotation(plain.strip(), width, height)


def _render_markdown_impl(text: str, width: int, height: int) -> np.ndarray:
    """Internal implementation — may raise on bad LaTeX."""
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")

    blocks = _parse_markdown(text)

    # Estimate chars per line — use larger divisor for bigger fonts
    chars_per_line = max(15, int(width / 14))

    margin_x = 0.05
    y_cursor = 0.92

    for block in blocks:
        if y_cursor < 0.02:
            break

        style = block["style"]
        if style == "spacer":
            y_cursor -= 0.04
            continue

        fontsize = _STYLE_FONTSIZE.get(style, 22)
        line_height = fontsize / (fig_h * dpi) * 1.8

        segments = block["segments"]
        wrapped_lines = _wrap_segments(segments, chars_per_line)

        for line_segs in wrapped_lines:
            if y_cursor < 0.02:
                break

            # Render each segment inline
            x_cursor = margin_x
            for seg in line_segs:
                weight = "bold" if (seg["bold"] or style in ("h1", "h2")) else "normal"
                seg_fontsize = fontsize

                # Sanitize math to plain text — matplotlib mathtext
                # only supports a tiny LaTeX subset and crashes on
                # common commands like \implies, \text{}, etc.
                display_text = (
                    _sanitize_latex(seg["text"]) if seg["math"]
                    else seg["text"]
                )

                fig.text(
                    x_cursor, y_cursor, display_text,
                    fontsize=seg_fontsize,
                    fontweight=weight,
                    color=_BASE_COLOR,
                    verticalalignment="top",
                    fontfamily="sans-serif",
                )

                # Estimate x advance
                char_width = seg_fontsize * 0.55 / (fig_w * dpi)
                x_cursor += len(display_text) * char_width

            y_cursor -= line_height

    # Render to numpy
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", edgecolor="none", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    import cv2
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    result = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if result.shape[0] != height or result.shape[1] != width:
        result = cv2.resize(result, (width, height), interpolation=cv2.INTER_AREA)

    return result
