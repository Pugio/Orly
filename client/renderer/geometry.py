"""Render geometric constructions on black backgrounds for projector overlay."""

import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def _render_geometry_impl(
    elements: list[dict],
    x_range: list[float],
    y_range: list[float],
    width: int,
    height: int,
    show_grid: bool = False,
) -> np.ndarray:
    """Render geometric construction on black background.

    Elements can be: point, line, circle, arc.

    Args:
        elements: List of geometry element dicts, each with a "type" key.
        x_range: [x_min, x_max] for the plot domain.
        y_range: [y_min, y_max] for the plot range.
        width: Output image width in pixels.
        height: Output image height in pixels.
        show_grid: Whether to show a coordinate grid.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")

    if show_grid:
        ax.grid(True, color="#333333", linewidth=0.5)
        ax.tick_params(colors="#666666", labelsize=10)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    DEFAULT_COLOR = "#00ffff"

    for elem in elements:
        etype = elem.get("type", "")
        color = elem.get("color", DEFAULT_COLOR)

        if etype == "point":
            x, y = elem.get("pos", [0, 0])
            ax.plot(x, y, "o", color=color, markersize=10, zorder=5)
            label = elem.get("label", "")
            if label:
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(8, 8),
                    color=color,
                    fontsize=14,
                    fontweight="bold",
                )

        elif etype == "line":
            x0, y0 = elem.get("from", [0, 0])
            x1, y1 = elem.get("to", [0, 0])
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.5)

        elif etype == "circle":
            cx, cy = elem.get("center", [0, 0])
            r = elem.get("radius", 1)
            style = elem.get("style", "solid")
            ls = "--" if style == "dashed" else "-"
            circle = plt.Circle(
                (cx, cy), r, fill=False, color=color, linewidth=2, linestyle=ls
            )
            ax.add_patch(circle)

        elif etype == "arc":
            cx, cy = elem.get("center", [0, 0])
            r = elem.get("radius", 1)
            start = elem.get("start_angle", 0)
            end = elem.get("end_angle", 90)
            arc = patches.Arc(
                (cx, cy),
                2 * r,
                2 * r,
                angle=0,
                theta1=start,
                theta2=end,
                color=color,
                linewidth=2,
            )
            ax.add_patch(arc)
            label = elem.get("label", "")
            if label:
                mid_angle = math.radians((start + end) / 2)
                lx = cx + (r + 0.3) * math.cos(mid_angle)
                ly = cy + (r + 0.3) * math.sin(mid_angle)
                ax.text(lx, ly, label, color=color, fontsize=14, ha="center")

        # Unknown element types silently skipped.

    fig.tight_layout()

    # Render to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", edgecolor="none", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    # Decode PNG to numpy array
    import cv2

    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Resize to exact requested dimensions (savefig may differ slightly)
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    return img


def render_geometry(data: dict, width: int, height: int, title: str = "") -> np.ndarray:
    """Registry-compatible wrapper: render geometry from data dict."""
    return _render_geometry_impl(
        data.get("elements", []),
        data.get("x_range", [-10, 10]),
        data.get("y_range", [-10, 10]),
        width, height,
        data.get("show_grid", False),
    )


SPEC = {
    "name": "geometry",
    "description": "Geometric constructions — points, lines, circles, arcs, angles.",
    "data_format": (
        '{"elements": [{"type": "point", "pos": [3, 4], "label": "A"}, '
        '{"type": "line", "from": [0, 0], "to": [3, 4]}, '
        '{"type": "circle", "center": [0, 0], "radius": 5}], '
        '"x_range": [-6, 6], "y_range": [-6, 6], "show_grid": true}.'
    ),
    "prompt_hint": "Use for geometric constructions with points, lines, circles, arcs.",
    "render": render_geometry,
}
