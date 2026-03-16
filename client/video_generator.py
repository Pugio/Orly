"""Async video generation using Veo and playback on the projector."""

import logging
import threading
import time

logger = logging.getLogger(__name__)

VEO_MODEL = "veo-3.1-generate-preview"
POLL_INTERVAL = 10  # seconds between polling checks
PROGRESS_INTERVAL = 60  # seconds between progress notifications
MAX_WAIT = 600  # 10 minutes max


class VideoGenerator:
    """Generates videos via Veo and plays them on the projector."""

    def __init__(self, overlay_manager, video_player, session_store, notify_fn):
        """
        Args:
            overlay_manager: OverlayManager for placing loading placeholder.
            video_player: VideoPlayer for playing generated videos.
            session_store: SessionStore for saving generated videos.
            notify_fn: Callable[[str], None] — sends notification to backend.
        """
        self._om = overlay_manager
        self._player = video_player
        self._session_store = session_store
        self._notify_fn = notify_fn

    def generate_async(
        self, name: str, prompt: str, placement: list,
        duration: int = 5, aspect_ratio: str = "16:9"
    ) -> None:
        """Start async video generation in a background thread.

        Shows a loading placeholder immediately, then generates in background.
        """
        # Show loading placeholder
        from client.renderer.video import render_video_loading
        w, h = self._om._placement_pixel_size(placement)
        loading = render_video_loading(
            {"prompt": prompt, "duration": duration}, w, h, name
        )
        self._om._show_overlay(loading, placement, "video")
        logger.info("Video generation started: '%s'", name)

        thread = threading.Thread(
            target=self._generate_video_thread,
            args=(name, prompt, placement, duration, aspect_ratio),
            daemon=True,
        )
        thread.start()

    def _generate_video_thread(
        self, name: str, prompt: str, placement: list,
        duration: int, aspect_ratio: str
    ) -> None:
        """Background thread: generate video, poll, download, save, play, notify."""
        try:
            from client.genai_utils import get_genai_client
            from google.genai import types

            client = get_genai_client()

            # Start generation
            operation = client.models.generate_videos(
                model=VEO_MODEL,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    duration_seconds=str(duration),
                ),
            )

            # Poll until done
            start_time = time.time()
            last_progress = start_time

            while not operation.done:
                elapsed = time.time() - start_time
                if elapsed > MAX_WAIT:
                    self._notify_fn(
                        f"Video '{name}' generation timed out after {MAX_WAIT}s."
                    )
                    return

                # Send progress notification periodically
                if time.time() - last_progress > PROGRESS_INTERVAL:
                    self._notify_fn(
                        f"Video '{name}' still generating... ({int(elapsed)}s elapsed)"
                    )
                    last_progress = time.time()

                time.sleep(POLL_INTERVAL)
                operation = client.operations.get(operation)

            # Download the video
            response = operation.response
            if not response or not response.generated_videos:
                self._notify_fn(f"Video '{name}' generation failed: no video in response.")
                return

            generated = response.generated_videos[0]
            video_file = generated.video

            # Download to bytes
            client.files.download(file=video_file)
            video_bytes = video_file.read()

            # Save to session
            video_path = self._session_store.save_video(name, video_bytes)
            logger.info("Video '%s' saved to %s", name, video_path)

            # Start playback
            self._player.play(name, video_path, placement, loop=True)

            # Register in overlay_state
            if self._om.overlay_state:
                # Create a thumbnail from the first frame for overlay state
                import cv2
                import numpy as np
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    w, h = self._om._placement_pixel_size(placement)
                    thumbnail = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    self._om.overlay_state.add(
                        name, "video", placement, name,
                        {"prompt": prompt, "duration": duration},
                        thumbnail, recomposite=False,
                    )

            # Notify
            self._notify_fn(f"Video '{name}' is ready and playing.")

        except Exception as e:
            logger.error("Video generation failed for '%s': %s", name, e)
            self._notify_fn(f"Video '{name}' generation failed: {e}")
