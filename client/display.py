"""
Display detection and window placement utilities.

Detects connected displays and provides helpers to place OpenCV windows
on specific displays (e.g. projector vs laptop screen).
"""

import subprocess
import re

import cv2
import numpy as np


def get_displays() -> list[dict]:
    """Detect connected displays via displayplacer.

    Returns list of dicts with keys: id, type, resolution (w,h), origin (x,y), scaling.
    Ordered by origin x (leftmost first).
    """
    try:
        output = subprocess.check_output(["displayplacer", "list"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    displays = []
    current = {}

    for line in output.splitlines():
        line = line.strip()

        if line.startswith("Persistent screen id:"):
            if current:
                displays.append(current)
            current = {"id": line.split(": ", 1)[1]}

        elif line.startswith("Type:"):
            current["type"] = line.split(": ", 1)[1]

        elif line.startswith("Resolution:") and "resolution" not in current:
            m = re.match(r"Resolution:\s*(\d+)x(\d+)", line)
            if m:
                current["resolution"] = (int(m.group(1)), int(m.group(2)))

        elif line.startswith("Origin:"):
            m = re.search(r"\((-?\d+),(-?\d+)\)", line)
            if m:
                current["origin"] = (int(m.group(1)), int(m.group(2)))

        elif line.startswith("Scaling:"):
            current["scaling"] = "on" in line

    if current:
        displays.append(current)

    displays.sort(key=lambda d: d.get("origin", (0, 0))[0])
    return displays


def find_projector() -> dict | None:
    """Find the projector display (non-built-in, external)."""
    for d in get_displays():
        if "built in" not in d.get("type", "").lower():
            return d
    return None


def find_laptop() -> dict | None:
    """Find the laptop's built-in display."""
    for d in get_displays():
        if "built in" in d.get("type", "").lower():
            return d
    return None


def show_on_projector(window_name: str, image: np.ndarray, fullscreen: bool = True) -> None:
    """Show an OpenCV window on the projector display.

    Creates/updates a named window positioned on the projector.
    Falls back to default display if no projector found.
    """
    proj = find_projector()

    if fullscreen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if proj and "origin" in proj:
        ox, oy = proj["origin"]
        cv2.moveWindow(window_name, ox, oy)

    cv2.imshow(window_name, image)


def show_on_laptop(window_name: str, image: np.ndarray) -> None:
    """Show an OpenCV window on the laptop display.

    Positions the window on the built-in display to avoid it appearing on the projector.
    """
    laptop = find_laptop()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    if laptop and "origin" in laptop:
        ox, oy = laptop["origin"]
        cv2.moveWindow(window_name, ox + 50, oy + 50)

    cv2.imshow(window_name, image)


def get_projector_resolution() -> tuple[int, int]:
    """Get the projector's resolution, or default (1280, 720)."""
    proj = find_projector()
    if proj and "resolution" in proj:
        return proj["resolution"]
    return (1280, 720)
