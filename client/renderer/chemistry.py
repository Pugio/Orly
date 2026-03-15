"""Render simple molecular structure diagrams on black backgrounds for projector overlay."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ELEMENT_COLORS = {
    "H": "#FFFFFF",
    "C": "#808080",
    "N": "#0000FF",
    "O": "#FF0000",
    "S": "#FFFF00",
    "P": "#FFA500",
    "Cl": "#00FF00",
    "F": "#00FFFF",
    "Br": "#A52A2A",
}

_DEFAULT_COLOR = "#CCCCCC"


def render_chemistry(
    atoms: list[dict],
    bonds: list[dict],
    width: int,
    height: int,
    title: str = "",
) -> np.ndarray:
    """Render a simple molecular structure diagram.

    atoms: [{"symbol": "O", "pos": [x, y], "color": "#ff0000"}, ...]
    bonds: [{"from": 0, "to": 1, "order": 1}, ...]

    Returns an (height, width, 3) uint8 BGR image on black background.
    """
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Auto-compute axis limits from atom positions
    if atoms:
        xs = [a["pos"][0] for a in atoms]
        ys = [a["pos"][1] for a in atoms]
        margin = 1.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
    else:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw bonds first (behind atoms)
    for bond in bonds:
        i_from = bond.get("from", 0)
        i_to = bond.get("to", 0)
        order = bond.get("order", 1)
        if i_from < 0 or i_to < 0 or i_from >= len(atoms) or i_to >= len(atoms):
            continue  # invalid index, skip
        x0, y0 = atoms[i_from]["pos"]
        x1, y1 = atoms[i_to]["pos"]

        # Perpendicular offset for parallel bonds
        dx = x1 - x0
        dy = y1 - y0
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            continue
        px, py = -dy / length * 0.08, dx / length * 0.08

        offsets = [0] if order == 1 else [-1, 1] if order == 2 else [-1, 0, 1]
        for off in offsets:
            ox, oy = px * off, py * off
            ax.plot(
                [x0 + ox, x1 + ox],
                [y0 + oy, y1 + oy],
                color="#888888",
                linewidth=2.5,
                zorder=1,
            )

    # Draw atoms
    for atom in atoms:
        symbol = atom.get("symbol", "?")
        x, y = atom.get("pos", [0, 0])
        color = atom.get("color", ELEMENT_COLORS.get(symbol, _DEFAULT_COLOR))
        circle = plt.Circle((x, y), 0.3, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            symbol,
            color="white" if color != "#FFFFFF" else "black",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            zorder=4,
        )

    if title:
        ax.set_title(title, color="white", fontsize=16, pad=10)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", edgecolor="none", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    import cv2

    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img
