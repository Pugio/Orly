"""Animated overlay player — runs frame generators on the projector canvas.

Mirrors VideoPlayer's threading model: each animation gets a daemon thread
that calls frame_fn(elapsed, w, h) at the target FPS and pushes frames
to the overlay manager's canvas via _show_overlay().
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class AnimatedOverlay:
    """Plays procedural animations on the projector canvas.

    Args:
        overlay_manager: OverlayManager instance to render onto.
    """

    def __init__(self, overlay_manager):
        self._om = overlay_manager
        self._animations: dict[str, threading.Event] = {}

    def start(
        self,
        name: str,
        frame_fn,
        placement: list,
        fps: float = 15,
    ) -> None:
        """Start an animation.

        Args:
            name: Unique name for this animation (used to stop it).
            frame_fn: Callable(elapsed_sec, width, height) -> np.ndarray (BGR).
            placement: [ymin, xmin, ymax, xmax] in table 0-1000 coords.
            fps: Target frames per second.
        """
        self.stop(name)

        stop_event = threading.Event()
        self._animations[name] = stop_event

        thread = threading.Thread(
            target=self._animation_thread,
            args=(name, frame_fn, placement, fps, stop_event),
            daemon=True,
        )
        thread.start()

    def stop(self, name: str) -> bool:
        """Stop a running animation. Returns True if it was running."""
        stop_event = self._animations.pop(name, None)
        if stop_event:
            stop_event.set()
            return True
        return False

    def stop_all(self) -> None:
        """Stop all running animations."""
        for stop_event in self._animations.values():
            stop_event.set()
        self._animations.clear()

    def is_running(self, name: str) -> bool:
        """Check if an animation is currently running."""
        return name in self._animations

    def _animation_thread(
        self,
        name: str,
        frame_fn,
        placement: list,
        fps: float,
        stop_event: threading.Event,
    ) -> None:
        """Background thread: generate frames and display at target FPS."""
        try:
            w, h = self._om.transform.placement_pixel_size(placement)
            if w <= 0 or h <= 0:
                logger.error("Invalid animation dimensions: %dx%d", w, h)
                return

            frame_delay = 1.0 / fps
            start_time = time.monotonic()

            logger.info("Animation '%s' started at %s (%dx%d, %.0f FPS)",
                        name, placement, w, h, fps)

            while not stop_event.is_set():
                elapsed = time.monotonic() - start_time
                frame = frame_fn(elapsed, w, h)
                self._om._show_overlay(frame, placement)
                stop_event.wait(frame_delay)

        except Exception as e:
            logger.error("Animation '%s' error: %s", name, e)
        finally:
            # Only remove our own entry — a new start() may have already
            # replaced it with a different stop_event.
            if self._animations.get(name) is stop_event:
                del self._animations[name]
            logger.info("Animation '%s' stopped.", name)
