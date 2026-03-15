"""Transition effects for overlay animations.

Provides crossfade, slide_in, and build_up transitions, plus a
TransitionState dataclass for tracking animated transitions over time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def crossfade(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    """Blend two overlays. alpha=0 -> all old, alpha=1 -> all new."""
    if old.shape != new.shape:
        raise ValueError(f"Shape mismatch: {old.shape} vs {new.shape}")
    return np.clip(old * (1 - alpha) + new * alpha, 0, 255).astype(np.uint8)


def slide_in(new: np.ndarray, direction: str, progress: float) -> np.ndarray:
    """Slide new overlay in from direction ('left','right','up','down').

    progress=0 -> all black, progress=1 -> full image.
    The image slides in from the given edge: e.g. direction='left' means
    the image enters from the left side.

    Args:
        new: The image to slide in.
        direction: One of 'left', 'right', 'up', 'down'.
        progress: 0.0 to 1.0 fraction of the slide completed.

    Returns:
        Result image with the slide-in applied.
    """
    valid = {"left", "right", "up", "down"}
    if direction not in valid:
        raise ValueError(f"Invalid direction '{direction}'. Must be one of {valid}")

    result = np.zeros_like(new)
    h, w = new.shape[:2]

    if progress <= 0.0:
        return result
    if progress >= 1.0:
        return new.copy()

    if direction == "left":
        # Image slides in from left: at progress p, rightmost p*w columns are filled
        cols = int(w * progress)
        # The visible portion is the rightmost 'cols' pixels of the image
        # placed at the right side of the result (leading edge on the right)
        result[:, w - cols :, :] = new[:, :cols, :]
    elif direction == "right":
        cols = int(w * progress)
        result[:, :cols, :] = new[:, w - cols :, :]
    elif direction == "up":
        rows = int(h * progress)
        result[h - rows :, :, :] = new[:rows, :, :]
    elif direction == "down":
        rows = int(h * progress)
        result[:rows, :, :] = new[h - rows :, :, :]

    return result


def build_up(full_image: np.ndarray, step: int, total_steps: int) -> np.ndarray:
    """Show only the top (step/total_steps) fraction of the image.

    Args:
        full_image: The complete image.
        step: How many steps to reveal (0 = nothing, total_steps = full).
        total_steps: Total number of steps.

    Returns:
        Image with only the top fraction visible, rest black.
    """
    result = np.zeros_like(full_image)
    if step <= 0:
        return result
    if step >= total_steps:
        return full_image.copy()

    h = full_image.shape[0]
    rows = int(h * step / total_steps)
    result[:rows, :, :] = full_image[:rows, :, :]
    return result


@dataclass
class TransitionState:
    """Tracks an animated transition between two overlay frames.

    Attributes:
        old_frame: The starting frame.
        new_frame: The ending frame.
        duration: Duration of the transition in seconds.
        start_time: Monotonic timestamp when the transition started.
    """

    old_frame: np.ndarray
    new_frame: np.ndarray
    duration: float = 0.3
    start_time: float = field(default_factory=time.monotonic)

    def progress(self) -> float:
        """Return current progress as a float 0.0 to 1.0."""
        elapsed = time.monotonic() - self.start_time
        return min(max(elapsed / self.duration, 0.0), 1.0)

    def is_done(self) -> bool:
        """Return True if the transition has completed."""
        return self.progress() >= 1.0

    def current_frame(self) -> np.ndarray:
        """Return the blended frame at the current progress."""
        return crossfade(self.old_frame, self.new_frame, self.progress())
