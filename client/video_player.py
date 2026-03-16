"""MP4 video playback on the projector canvas frame by frame."""

import logging
import threading

import cv2

logger = logging.getLogger(__name__)


class VideoPlayer:
    """Plays MP4 video files on the projector canvas at native FPS."""

    def __init__(self, overlay_manager):
        self._om = overlay_manager
        self._players: dict[str, threading.Event] = {}  # name → stop_event

    def play(self, name: str, video_path: str, placement: list, loop: bool = False) -> None:
        """Start playing a video at the given placement.

        Args:
            name: Unique name for this playback instance.
            video_path: Path to the MP4 file.
            placement: [ymin, xmin, ymax, xmax] normalised 0-1000.
            loop: Whether to loop the video.
        """
        # Stop existing playback with same name
        self.stop(name)

        stop_event = threading.Event()
        self._players[name] = stop_event

        thread = threading.Thread(
            target=self._playback_thread,
            args=(name, video_path, placement, loop, stop_event),
            daemon=True,
        )
        thread.start()

    def stop(self, name: str) -> bool:
        """Stop a playing video. Returns True if it was playing."""
        stop_event = self._players.pop(name, None)
        if stop_event:
            stop_event.set()
            return True
        return False

    def stop_all(self) -> None:
        """Stop all playing videos."""
        for stop_event in self._players.values():
            stop_event.set()
        self._players.clear()

    def _playback_thread(
        self, name: str, video_path: str, placement: list,
        loop: bool, stop_event: threading.Event
    ) -> None:
        """Background thread: read frames from MP4 and display at native FPS."""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Failed to open video: %s", video_path)
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            frame_delay = 1.0 / fps

            # Calculate pixel size from placement
            ymin, xmin, ymax, xmax = placement
            w = int((xmax - xmin) / 1000 * self._om.proj_width)
            h = int((ymax - ymin) / 1000 * self._om.proj_height)
            if w <= 0 or h <= 0:
                logger.error("Invalid video placement dimensions: %dx%d", w, h)
                return

            logger.info("Playing video '%s' at %s (%dx%d, %.1f FPS)", name, placement, w, h, fps)

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                # Resize to overlay dimensions
                resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

                # Place on canvas
                self._om._show_overlay(resized, placement, "video")

                # Wait for next frame
                stop_event.wait(frame_delay)

            logger.info("Video '%s' playback finished.", name)

        except Exception as e:
            logger.error("Video playback error for '%s': %s", name, e)
        finally:
            if cap is not None:
                cap.release()
            self._players.pop(name, None)
