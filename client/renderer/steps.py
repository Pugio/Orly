"""Render step-by-step overlay showing incremental steps.

Each step has a title and content rendered using the markdown renderer.
Steps are stacked vertically, and only visible_count steps are shown.
"""

from __future__ import annotations

import numpy as np

from client.renderer.markdown import _render_markdown_impl as _render_md


def _render_steps_impl(
    steps: list[dict],
    visible_count: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Render step-by-step overlay showing first visible_count steps.

    Args:
        steps: List of dicts with "title" and "content" keys.
        visible_count: How many steps to show (0 = none, clamped to len(steps)).
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if not steps or visible_count <= 0:
        return img

    # Clamp visible_count to the number of steps available.
    visible_count = min(visible_count, len(steps))

    # Each step gets an equal vertical slice.
    step_height = height // len(steps)

    for i in range(visible_count):
        step = steps[i]
        title = step.get("title", "")
        content = step.get("content", "")
        md_text = f"## {title}\n\n{content}"

        rendered = _render_md(md_text, width, step_height)
        y_start = i * step_height
        y_end = y_start + step_height
        img[y_start:y_end, :, :] = rendered

    return img


def render_steps(data: dict, width: int, height: int, title: str = "") -> np.ndarray:
    """Registry-compatible wrapper: render steps from data dict."""
    return _render_steps_impl(
        data.get("steps", []),
        data.get("visible_count", 0),
        width, height,
    )


SPEC = {
    "name": "steps",
    "description": "Multi-step explanations that reveal one step at a time.",
    "data_format": '{"steps": [{"title": "Step 1", "content": "..."}, ...], "visible_count": 1}.',
    "prompt_hint": "After projecting, use advance_step to reveal each step.",
    "render": render_steps,
}
