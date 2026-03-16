"""Receives tool results from the backend, renders and displays overlays.

Manages a canvas (black background) that overlays are placed onto.
In projector mode, uses H_proj to map table coordinates to projector pixels.
In screen mode, maps Gemini's 0-1000 coordinate system directly to pixel space.
"""

import logging
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from client.renderer.image import render_image, render_loading
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
        self.image_rotate = image_rotate  # how the image was rotated before Gemini saw it
        self.white_bg = white_bg
        self.session_store = session_store
        self.notify_fn = notify_fn
        self.overlay_state = None  # set externally to register async image completions
        self.canvas = self._make_bg()
        self._has_content = False
        self._last_rendered_overlay: np.ndarray | None = None  # for overlay_state registration
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
        """Process a tool result from the backend and render the overlay."""
        if name == "show_scene":
            self._handle_show_scene(result)
            return
        if name != "project_overlay":
            return

        content_type = result.get("content_type", "annotation")
        placement = list(result.get("placement", [0, 0, 1000, 1000]))
        title = result.get("title", "")
        data = result.get("data", {})

        placement = self._unrotate_placement(placement)
        placement = self.adjust_text_placement(content_type, placement)

        if content_type == "image":
            # Image generation is slow — show loading placeholder immediately,
            # then generate in a background thread and swap in the result.
            loading = render_loading(
                data.get("prompt", title),
                *self._placement_pixel_size(placement),
            )
            loading = self._unrotate_image(loading)
            self._show_overlay(loading, placement, content_type)
            logger.info("Generating image at %s", placement)
            thread = threading.Thread(
                target=self._generate_image_async,
                args=(placement, title, data),
                daemon=True,
            )
            thread.start()
            return

        overlay = self.render_overlay(content_type, placement, title, data)
        # Un-rotate locally-rendered overlays (text, graphs, etc.) so they
        # appear in the human's orientation, not the camera's.
        overlay = self._unrotate_image(overlay)
        self._show_overlay(overlay, placement, content_type)
        self._last_rendered_overlay = overlay  # cached for overlay_state registration
        logger.info("Rendered %s at %s", content_type, placement)

    def _unrotate_image(self, overlay: np.ndarray) -> np.ndarray:
        """Un-rotate a locally-rendered overlay to match the human's viewing angle.

        Applied to text, graphs, markdown, etc. — content rendered in standard
        orientation that must be rotated to match the canvas (camera space).
        NOT applied to AI-generated images (those are already standard orientation).
        """
        if self.image_rotate == 90:
            return cv2.rotate(overlay, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.image_rotate == 180:
            return cv2.rotate(overlay, cv2.ROTATE_180)
        elif self.image_rotate == 270:
            return cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)
        return overlay

    def _show_overlay(self, overlay: np.ndarray, placement: list, content_type: str):
        """Apply projector flip and place overlay on canvas."""
        if self.mode == "projector" and content_type != "highlight":
            overlay = cv2.rotate(overlay, cv2.ROTATE_180)
        self.canvas = self.place_on_canvas(overlay, placement)
        self._has_content = True

    def _placement_pixel_size(self, placement: list) -> tuple[int, int]:
        """Return (width, height) in pixels for a placement box."""
        y_min, x_min, y_max, x_max = placement
        w = max(1, int((x_max - x_min) / 1000.0 * self.proj_width))
        h = max(1, int((y_max - y_min) / 1000.0 * self.proj_height))
        return w, h

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

            # NOTE: Generated images are NOT un-rotated. The image gen API
            # produces images in standard orientation from a text prompt —
            # they don't inherit the camera rotation. Only the placement
            # coordinates need un-rotation (handled by _unrotate_placement).

            # Save to scene gallery by title.
            self.scenes[title] = overlay.copy()
            if title not in self._scene_order:
                self._scene_order.append(title)
            logger.info("Image '%s' ready (scenes: %s)", title, list(self.scenes.keys()))

            if self.session_store:
                self.session_store.save_image(title, overlay)

            # If an interruption cleared the canvas while we were generating,
            # skip showing the stale image.
            if self._generation_id != gen_id:
                logger.info("Image '%s' ready but canvas was cleared — skipping display", title)
                if self.notify_fn:
                    self.notify_fn(f"Image '{title}' generated but not displayed (interrupted).")
                return

            self._show_overlay(overlay, placement, "image")

            # Register in overlay_state (no recomposite — we just placed it).
            if self.overlay_state:
                self.overlay_state.add(
                    title, "image", placement, title, data, overlay,
                    recomposite=False)

            if self.notify_fn:
                self.notify_fn(f"Image '{title}' is ready and displayed.")
        except Exception as e:
            logger.error("Image generation failed: %s", e)

    def _handle_show_scene(self, result: dict):
        """Show a previously generated scene on the projector."""
        scene_name = result.get("scene_name", "")
        placement = list(result.get("placement", [0, 0, 1000, 1000]))

        if scene_name not in self.scenes:
            logger.warning("Scene '%s' not found. Available: %s", scene_name, list(self.scenes.keys()))
            return

        overlay = self.scenes[scene_name]
        self._show_overlay(overlay, placement, "image")
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
            placement: [ymin, xmin, ymax, xmax] in Gemini 0-1000 coords.
            title: Title text (used as prefix for annotations).
            data: Type-specific data dict.

        Returns:
            Rendered overlay as BGR numpy array (black background).
        """
        # Compute overlay dimensions from placement
        # Gemini returns [ymin, xmin, ymax, xmax]
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

        In screen mode: maps Gemini 0-1000 coords directly to pixel coordinates.
        In projector mode: uses H_proj to map table coords to projector pixels.

        Returns updated canvas.
        """
        canvas = self.canvas.copy()
        # Gemini returns [ymin, xmin, ymax, xmax]
        y_min, x_min, y_max, x_max = placement

        use_direct = True  # default to direct screen mapping

        if self.mode == "projector" and self.H_proj is not None:
            # Placement is [y,x] but H_proj was calibrated with (x,y) input.
            # Swap columns to match calibration convention.
            src_corners = np.array([
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_max],
                [x_max, y_min],
            ], dtype=np.float64).reshape(1, 4, 2)
            dst_corners = cv2.perspectiveTransform(src_corners, self.H_proj)
            dst_corners = dst_corners.reshape(4, 2)

            # H_proj output: column 0 = projector x, column 1 = projector y.
            # Use perspective warp instead of bounding-box resize to
            # preserve perspective correction for off-axis projectors.
            oh, ow = overlay.shape[:2]
            # Must match src_corners order: TL, BL, BR, TR
            overlay_corners = np.array(
                [[0, 0], [0, oh], [ow, oh], [ow, 0]], dtype=np.float32
            )
            # dst_corners columns: 0=px, 1=py → getPerspectiveTransform wants (x,y)
            dst_xy = dst_corners.astype(np.float32)
            M = cv2.getPerspectiveTransform(overlay_corners, dst_xy)
            warped = cv2.warpPerspective(
                overlay, M, (self.proj_width, self.proj_height)
            )
            # Composite: non-black warped pixels onto canvas
            threshold = 30 if self.white_bg else 0
            mask = warped.sum(axis=2) > threshold
            canvas[mask] = warped[mask]
            use_direct = False

        if use_direct:
            px_min = max(0, int(x_min / 1000.0 * self.proj_width))
            py_min = max(0, int(y_min / 1000.0 * self.proj_height))
            px_max = min(self.proj_width, int(x_max / 1000.0 * self.proj_width))
            py_max = min(self.proj_height, int(y_max / 1000.0 * self.proj_height))

            if px_max > px_min and py_max > py_min:
                resized = cv2.resize(overlay, (px_max - px_min, py_max - py_min))
                self._composite(canvas, resized, py_min, py_max, px_min, px_max)

        return canvas

    def _composite(self, canvas: np.ndarray, overlay: np.ndarray,
                   y1: int, y2: int, x1: int, x2: int) -> None:
        """Place overlay onto canvas region, handling white background mode.

        With black bg: direct overwrite (black is transparent to projector).
        With white bg: only overwrite pixels where overlay has content
        (non-black), so white background shows through elsewhere.
        """
        if self.white_bg:
            mask = overlay.sum(axis=2) > 30  # non-black content
            region = canvas[y1:y2, x1:x2]
            region[mask] = overlay[mask]
        else:
            canvas[y1:y2, x1:x2] = overlay

    def _unrotate_placement(self, placement: list) -> list:
        """Un-rotate Gemini coordinates from rotated image back to marker space.

        Gemini returns [ymin, xmin, ymax, xmax] in 0-1000 of the image it saw.
        If the image was rotated CW by N degrees before Gemini saw it,
        we reverse that rotation on the coordinates.
        """
        ymin, xmin, ymax, xmax = placement

        if self.image_rotate == 90:
            # CW 90 forward: (y, x) → (x, 1000-y)
            # Inverse: (yr, xr) → (y=1000-xr, x=yr)
            return [1000 - xmax, ymin, 1000 - xmin, ymax]
        elif self.image_rotate == 180:
            return [1000 - ymax, 1000 - xmax, 1000 - ymin, 1000 - xmin]
        elif self.image_rotate == 270:
            # CCW 90 forward: (y, x) → (1000-x, y)
            # Inverse: (yr, xr) → (y=xr, x=1000-yr)
            return [xmin, 1000 - ymax, xmax, 1000 - ymin]

        return placement

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
        doesn't restore a stale pre-interrupt canvas.
        """
        self.canvas = self._make_bg()
        self._has_content = False
        self._refresh_requested = False
        self._saved_canvas = None
        self._generation_id += 1
