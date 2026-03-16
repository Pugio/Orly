"""Receives tool results from the backend, renders and displays overlays.

Manages a canvas (black background) that overlays are placed onto.
Delegates ALL coordinate/orientation transforms to CoordinateTransform —
the single source of truth for spatial operations.
"""

import logging
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from client.animated_overlay import AnimatedOverlay
from client.coordinate_transform import CoordinateTransform
from client.renderer.image import render_image, render_loading_frame
from client.renderer.registry import get as get_renderer_spec


class OverlayManager:
    """Manages overlay rendering and display on projector or screen.

    Args:
        H_proj: Table coords -> projector pixels homography (None for screen mode).
        proj_width: Projector/canvas width in pixels.
        proj_height: Projector/canvas height in pixels.
        mode: "projector" or "screen".
    """

    def __init__(
        self,
        H_proj: np.ndarray | None,
        proj_width: int = 1280,
        proj_height: int = 720,
        mode: str = "screen",
        image_rotate: int = 0,
        white_bg: bool = False,
        session_store=None,
        notify_fn=None,
    ):
        self.H_proj = H_proj
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.mode = mode
        self.image_rotate = image_rotate  # kept for backward compat / tests
        self.white_bg = white_bg
        self.session_store = session_store
        self.notify_fn = notify_fn
        self.overlay_state = None  # set externally to register async image completions

        # Central coordinate/orientation transform — the SINGLE authority
        # for all spatial operations (see client/coordinate_transform.py).
        self.transform = CoordinateTransform(
            rotate=image_rotate,
            H_proj=H_proj,
            display_width=proj_width,
            display_height=proj_height,
            mode=mode,
        )

        self.canvas = self._make_bg()
        self._has_content = False
        self._last_rendered_overlay: np.ndarray | None = None  # for overlay_state registration
        self.animated = AnimatedOverlay(self)
        # Named scene gallery: title → BGR numpy array.
        self.scenes: dict[str, np.ndarray] = {}
        # Ordered list of scene names for "reference_previous".
        self._scene_order: list[str] = []
        # Set by refresh_view to temporarily hide overlays for a clean capture.
        self._refresh_requested = False
        self._saved_canvas = None  # stashed canvas during refresh
        # Last clean frame (JPEG bytes) captured by video_loop.
        self.last_clean_frame: bytes | None = None
        # Generation ID — incremented on clear() so background image threads
        # can detect that an interruption happened and skip showing stale images.
        self._generation_id: int = 0

    def handle_tool_result(self, name: str, result: dict):
        """Process a tool result from the backend and render the overlay.

        Dispatches on result["action"]: create, show_scene, remove, clear.
        """
        if name != "overlay":
            return
        if result.get("status") == "error":
            return
        action = result.get("action", "")
        if action == "create":
            self._handle_create(result)
        elif action == "show_scene":
            self._handle_show_scene(result)
        elif action == "remove":
            self._handle_remove(result)
        elif action == "clear":
            self._handle_clear()
        # advance_step and flip_flashcard are handled by client/main.py

    def _handle_create(self, result: dict):
        """Handle overlay create action."""
        content_type = result.get("content_type", "annotation")
        placement = list(result.get("placement", [0, 0, 1000, 1000]))
        title = result.get("title", "")
        data = result.get("data", {})

        placement = self._unrotate_placement(placement)
        placement = self.adjust_text_placement(content_type, placement)

        if content_type == "image":
            # Image generation is slow — show animated loading indicator
            # (water-fill rising over ~60s), then swap in the real image.
            prompt = data.get("prompt", title)

            def loading_frame(elapsed, w, h):
                return render_loading_frame(elapsed, w, h, prompt)

            self.animated.start(title, loading_frame, placement)
            logger.info("Generating image at %s (animated loading)", placement)
            thread = threading.Thread(
                target=self._generate_image_async,
                args=(placement, title, data),
                daemon=True,
            )
            thread.start()
            return

        overlay = self.render_overlay(content_type, placement, title, data)
        self._last_rendered_overlay = self._show_overlay(overlay, placement)
        logger.info("Rendered %s at %s", content_type, placement)

    def _handle_remove(self, result: dict):
        """Handle overlay remove action."""
        overlay_name = result.get("overlay_name", "")
        if self.overlay_state and overlay_name:
            self.overlay_state.remove(overlay_name)
            logger.info("Removed overlay '%s'", overlay_name)

    def _handle_clear(self):
        """Handle overlay clear action."""
        if self.overlay_state:
            self.overlay_state.clear()
        else:
            self.clear()
        logger.info("Cleared all overlays")

    def _show_overlay(self, overlay: np.ndarray, placement: list) -> np.ndarray:
        """Orient overlay content and place it on the canvas.

        This is the standard entry point for displaying overlays. It
        applies orient_overlay() to rotate the content for the human's
        viewing angle, then places it on the canvas via CoordinateTransform.

        ALL new overlay content must go through this method (or
        _show_preoriented for content that's already been oriented).

        Returns the oriented overlay (for callers that need to store it).
        """
        oriented = self.transform.orient_overlay(overlay)
        self.canvas = self.place_on_canvas(oriented, placement)
        self._has_content = True
        return oriented

    def _show_preoriented(self, overlay: np.ndarray, placement: list):
        """Place an already-oriented overlay on the canvas.

        Use ONLY for content that was previously oriented and stored
        (e.g. replaying a scene from self.scenes, restoring from session).
        For new content, use _show_overlay() which orients automatically.
        """
        self.canvas = self.place_on_canvas(overlay, placement)
        self._has_content = True

    # Legacy aliases for backward compatibility with tests
    def _orient_overlay(self, overlay: np.ndarray) -> np.ndarray:
        return self.transform.orient_overlay(overlay)

    _unrotate_image = _orient_overlay

    def _placement_pixel_size(self, placement: list) -> tuple[int, int]:
        """Return (width, height) in pixels for a placement box."""
        return self.transform.placement_pixel_size(placement)

    def _generate_image_async(self, placement: list, title: str, data: dict):
        """Background thread: generate image and swap onto canvas."""
        gen_id = self._generation_id
        try:
            w, h = self._placement_pixel_size(placement)
            prompt = data.get("prompt", title)
            style = data.get("style", "default")
            include_view = data.get("include_view", False)
            reference_previous = data.get("reference_previous", False)
            reference_scene = data.get("reference_scene", "")

            ref_frame = None
            if reference_scene and reference_scene in self.scenes:
                ref_frame = self.scenes[reference_scene]
                logger.info("Using scene '%s' as reference", reference_scene)
            elif reference_previous and self._scene_order:
                last_name = self._scene_order[-1]
                ref_frame = self.scenes[last_name]
                logger.info("Using previous scene '%s' as reference", last_name)
            elif include_view and self.last_clean_frame:
                frame_bytes = np.frombuffer(self.last_clean_frame, dtype=np.uint8)
                ref_frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

            enhance = include_view and ref_frame is not None
            overlay = render_image(prompt, w, h, reference_frame=ref_frame, style=style, enhance=enhance)

            # Orient first, then store the oriented version.
            # Use _show_preoriented since we orient manually for storage.
            oriented = self.transform.orient_overlay(overlay)

            # Save oriented image to scene gallery and session store.
            self.scenes[title] = oriented.copy()
            if title not in self._scene_order:
                self._scene_order.append(title)
            logger.info("Image '%s' ready (scenes: %s)", title, list(self.scenes.keys()))

            if self.session_store:
                self.session_store.save_image(title, oriented)

            # Stop loading animation before showing the real image so
            # the animation thread can't overwrite it on the next tick.
            # (The finally block also calls stop() as a safety net for
            # the failure path — stop() is idempotent.)
            self.animated.stop(title)

            # If an interruption cleared the canvas while we were generating,
            # skip showing the stale image.
            if self._generation_id != gen_id:
                logger.info("Image '%s' ready but canvas was cleared — skipping display", title)
                if self.notify_fn:
                    self.notify_fn(f"Image '{title}' generated but not displayed (interrupted).")
                return

            self._show_preoriented(oriented, placement)

            # Register in overlay_state (no recomposite — we just placed it).
            if self.overlay_state:
                self.overlay_state.add(
                    title, "image", placement, title, data, oriented,
                    recomposite=False)

            if self.notify_fn:
                self.notify_fn(f"Image '{title}' is ready and displayed.")
        except Exception as e:
            logger.error("Image generation failed: %s", e)
        finally:
            self.animated.stop(title)

    def _handle_show_scene(self, result: dict):
        """Show a previously generated scene on the projector."""
        scene_name = result.get("scene_name", "")
        placement = list(result.get("placement") or [0, 0, 1000, 1000])
        placement = self._unrotate_placement(placement)

        if scene_name not in self.scenes:
            logger.warning("Scene '%s' not found. Available: %s", scene_name, list(self.scenes.keys()))
            return

        overlay = self.scenes[scene_name]
        self._show_preoriented(overlay, placement)
        logger.info("Showing scene '%s'", scene_name)

    def render_overlay(
        self,
        content_type: str,
        placement: list,
        title: str,
        data: dict,
    ) -> np.ndarray:
        """Render an overlay image using client/renderer/ modules.

        Args:
            content_type: Any registered renderer name (see client/renderer/registry.py).
            placement: [ymin, xmin, ymax, xmax] in table 0-1000 coords.
            title: Title text (used as prefix for annotations).
            data: Type-specific data dict.

        Returns:
            Rendered overlay as BGR numpy array (black background).
        """
        # Compute overlay dimensions from placement
        y_min, x_min, y_max, x_max = placement
        w_frac = (x_max - x_min) / 1000.0
        h_frac = (y_max - y_min) / 1000.0
        overlay_w = max(1, int(w_frac * self.proj_width))
        overlay_h = max(1, int(h_frac * self.proj_height))

        spec = get_renderer_spec(content_type)
        if spec:
            return spec["render"](data, overlay_w, overlay_h, title=title)

        # Unknown type: return black image
        return np.zeros((overlay_h, overlay_w, 3), dtype=np.uint8)

    def place_on_canvas(
        self,
        overlay: np.ndarray,
        placement: list,
    ) -> np.ndarray:
        """Place overlay on the projector/screen canvas at the given table coordinates.

        Delegates to CoordinateTransform.place_on_canvas() for all spatial mapping.

        Returns updated canvas.
        """
        return self.transform.place_on_canvas(
            self.canvas, overlay, placement, white_bg=self.white_bg
        )

    def _unrotate_placement(self, placement: list) -> list:
        """Un-rotate Gemini coordinates from rotated image back to table space.

        Delegates to CoordinateTransform.gemini_to_table().
        """
        return self.transform.gemini_to_table(placement)

    @staticmethod
    def adjust_text_placement(content_type: str, placement: list) -> list:
        """Enforce minimum placement size for text-heavy content types.

        Expands markdown/annotation placements to at least 500x400 units
        so text remains readable on a low-res projector. Returns the
        (possibly expanded) placement. Non-text types are returned unchanged.
        """
        if content_type not in ("markdown", "annotation"):
            return placement
        y_min, x_min, y_max, x_max = placement
        min_w, min_h = 500, 400
        if (x_max - x_min) < min_w:
            x_max = min(1000, x_min + min_w)
            if (x_max - x_min) < min_w:
                x_min = max(0, x_max - min_w)
        if (y_max - y_min) < min_h:
            y_max = min(1000, y_min + min_h)
            if (y_max - y_min) < min_h:
                y_min = max(0, y_max - min_h)
        return [y_min, x_min, y_max, x_max]

    def _make_bg(self) -> np.ndarray:
        """Create a blank background canvas."""
        if self.white_bg:
            return np.full(
                (self.proj_height, self.proj_width, 3), 255, dtype=np.uint8
            )
        return np.zeros((self.proj_height, self.proj_width, 3), dtype=np.uint8)

    def request_refresh(self):
        """Request a clean frame capture. Hides overlays temporarily."""
        if not self._refresh_requested:
            self._saved_canvas = self.canvas.copy()
            self.canvas = self._make_bg()
            self._refresh_requested = True
            logger.info("Refresh requested — overlays hidden for capture.")

    def complete_refresh(self):
        """Restore overlays after clean frame was captured."""
        if self._refresh_requested:
            if self._saved_canvas is not None:
                self.canvas = self._saved_canvas
                self._saved_canvas = None
            self._refresh_requested = False
            logger.info("Refresh complete — overlays restored.")

    def clear(self):
        """Clear all overlays (reset canvas to background).

        Also cancels any in-flight refresh cycle so complete_refresh()
        doesn't restore a stale pre-interrupt canvas, and stops all
        running animations (loading indicators, etc.).
        """
        self.animated.stop_all()
        self.canvas = self._make_bg()
        self._has_content = False
        self._refresh_requested = False
        self._saved_canvas = None
        self._generation_id += 1
