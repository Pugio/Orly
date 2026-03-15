"""Render number lines on black backgrounds for projector overlay."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_number_line(
    min_val: float,
    max_val: float,
    points: list[dict],
    ranges: list[dict],
    width: int,
    height: int,
) -> np.ndarray:
    """Render a number line with labeled points and shaded ranges.

    Args:
        min_val: Left end of number line.
        max_val: Right end of number line.
        points: List of {"value": float, "label": str, "color": str} dicts.
        ranges: List of {"start": float, "end": float, "color": str, "label": str} dicts.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    # Handle degenerate case
    if min_val >= max_val:
        max_val = min_val + 1

    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Draw the number line (horizontal axis)
    ax.axhline(y=0, color="cyan", linewidth=3)
    ax.set_xlim(min_val - 0.5, max_val + 0.5)
    ax.set_ylim(-1, 1)

    # Integer tick marks
    tick_start = int(np.ceil(min_val))
    tick_end = int(np.floor(max_val))
    ticks = list(range(tick_start, tick_end + 1))
    ax.set_xticks(ticks)
    for t in ticks:
        ax.plot([t, t], [-0.15, 0.15], color="white", linewidth=2)

    # Draw ranges (semi-transparent rectangles)
    for r in ranges:
        start = r.get("start", 0)
        end = r.get("end", 0)
        color = r.get("color", "#ffff00")
        label = r.get("label", "")
        ax.axvspan(start, end, alpha=0.3, color=color)
        if label:
            mid = (start + end) / 2
            ax.text(mid, 0.5, label, color=color, ha="center", fontsize=14)

    # Draw points
    for p in points:
        val = p.get("value", 0)
        label = p.get("label", "")
        color = p.get("color", "#00ffff")
        ax.plot(val, 0, "o", color=color, markersize=15, zorder=5)
        if label:
            ax.text(val, 0.3, label, color=color, ha="center",
                    fontsize=16, fontweight="bold")

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("cyan")
    ax.tick_params(axis="x", colors="white", labelsize=14)
    ax.tick_params(axis="y", left=False, labelleft=False)

    fig.tight_layout()

    # Render to numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", edgecolor="none", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    # Decode PNG to numpy array
    import cv2
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR

    # Resize to exact requested dimensions (savefig may differ slightly)
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    return img
