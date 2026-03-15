"""Render math expression graphs on black backgrounds for projector overlay."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_graph(
    expression: str,
    x_range: list[float],
    y_range: list[float],
    width: int,
    height: int,
) -> np.ndarray:
    """Render a math expression as a graph on a black background.

    Args:
        expression: Math expression using x as variable, evaluated with numpy
                    (e.g. "x**2 - 3*x + 2", "np.sin(x)").
        x_range: [x_min, x_max] for the plot domain.
        y_range: [y_min, y_max] for the plot range.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Evaluate expression — fix implicit multiplication (e.g. "7x" → "7*x")
    import re
    safe_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = eval(safe_expr, {"__builtins__": {}, "x": x, "np": np})

    ax.plot(x, y, color="cyan", linewidth=3)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Style axes for projector visibility
    ax.tick_params(colors="cyan", labelsize=12)
    ax.spines["bottom"].set_color("cyan")
    ax.spines["left"].set_color("cyan")
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.xaxis.label.set_color("cyan")
    ax.yaxis.label.set_color("cyan")
    ax.set_xlabel("x", fontsize=14, color="cyan")
    ax.set_ylabel("y", fontsize=14, color="cyan")

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
