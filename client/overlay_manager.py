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

from client.renderer.graph import render_graph
from client.renderer.annotation import render_annotation
from client.renderer.highlight import render_highlight
from client.renderer.markdown import render_markdown
from client.renderer.image import render_image, render_loading


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
    ):
        self.H_proj = H_proj
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.mode = mode
        self.image_rotate = image_rotate  # how the image was rotated before Gemini saw it
        self.white_bg = white_bg
        self.canvas = self._make_bg()
        # Named scene gallery: title → BGR numpy array.
        self.scenes: dict[str, np.ndarray] = {}
        # Ordered list of scene names for "reference_previous".
        self._scene_order: list[str] = []
        # Set by refresh_view to temporarily hide overlays for a clean capture.
        self._refresh_requested = False
        self._saved_canvas = None  # stashed canvas during refresh
        # Last clean frame (JPEG bytes) captured by video_loop.
        self.last_clean_frame: bytes | None = None

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

        # Enforce minimum placement size for text-heavy content types so
        # they remain readable on a low-res projector.
        if content_type in ("markdown", "annotation"):
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
            placement = [y_min, x_min, y_max, x_max]

        if content_type == "image":
            # Image generation is slow — show loading placeholder immediately,
            # then generate in a background thread and swap in the result.
            self._show_overlay(
                render_loading(data.get("prompt", title),
                               *self._placement_pixel_size(placement)),
                placement, content_type,
            )
            logger.info("Generating image at %s", placement)
            thread = threading.Thread(
                target=self._generate_image_async,
                args=(placement, title, data),
                daemon=True,
            )
            thread.start()
            return

        overlay = self.render_overlay(content_type, placement, title, data)
        self._show_overlay(overlay, placement, content_type)
        logger.info("Rendered %s at %s", content_type, placement)

    def _show_overlay(self, overlay: np.ndarray, placement: list, content_type: str):
        """Apply projector flip and place overlay on canvas."""
        if self.mode == "projector" and content_type != "highlight":
            overlay = cv2.rotate(overlay, cv2.ROTATE_180)
        self.canvas = self.place_on_canvas(overlay, placement)

    def _placement_pixel_size(self, placement: list) -> tuple[int, int]:
        """Return (width, height) in pixels for a placement box."""
        y_min, x_min, y_max, x_max = placement
        w = max(1, int((x_max - x_min) / 1000.0 * self.proj_width))
        h = max(1, int((y_max - y_min) / 1000.0 * self.proj_height))
        return w, h

    def _generate_image_async(self, placement: list, title: str, data: dict):
        """Background thread: generate image and swap onto canvas."""
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

            overlay = render_image(prompt, w, h, reference_frame=ref_frame, style=style)

            # Save to scene gallery by title.
            self.scenes[title] = overlay.copy()
            if title not in self._scene_order:
                self._scene_order.append(title)
            logger.info("Image '%s' ready (scenes: %s)", title, list(self.scenes.keys()))

            self._show_overlay(overlay, placement, "image")
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
            content_type: "graph", "annotation", or "highlight".
            placement: [x_min, y_min, x_max, y_max] in Gemini 0-1000 coords.
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

        if content_type == "graph":
            expression = data.get("expression", "x")
            x_range = data.get("x_range", [-10, 10])
            y_range = data.get("y_range", [-10, 10])
            return render_graph(expression, x_range, y_range, overlay_w, overlay_h)

        elif content_type == "annotation":
            text = data.get("text", title)
            return render_annotation(text, overlay_w, overlay_h)

        elif content_type == "markdown":
            text = data.get("text", title)
            return render_markdown(text, overlay_w, overlay_h)

        elif content_type == "image":
            prompt = data.get("prompt", title)
            return render_image(prompt, overlay_w, overlay_h)

        elif content_type == "highlight":
            color_hex = data.get("color", "#00ffff")
            # render_highlight returns BGRA; convert to BGR for canvas compositing
            bgra = render_highlight(overlay_w, overlay_h, color_hex=color_hex)
            return bgra[:, :, :3]

        else:
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
            src_corners = np.array([
                [y_min, x_min],
                [y_max, x_min],
                [y_max, x_max],
                [y_min, x_max],
            ], dtype=np.float64).reshape(1, 4, 2)
            dst_corners = cv2.perspectiveTransform(src_corners, self.H_proj)
            dst_corners = dst_corners.reshape(4, 2)

            # H_proj was calibrated with [y,x] input and the output was
            # originally consumed as column 0=x, column 1=y.  The
            # calibration baked in that convention, so we keep it:
            # column 0 = projector x, column 1 = projector y.
            px_min = max(0, int(dst_corners[:, 0].min()))
            py_min = max(0, int(dst_corners[:, 1].min()))
            px_max = min(self.proj_width, int(dst_corners[:, 0].max()))
            py_max = min(self.proj_height, int(dst_corners[:, 1].max()))
            if px_max > px_min and py_max > py_min:
                w, h = px_max - px_min, py_max - py_min
                resized = cv2.resize(overlay, (w, h))
                self._composite(canvas, resized, py_min, py_max, px_min, px_max)
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
            # CW 90: image (y, x) came from original (x, 1000-y)
            # So to go back: orig_y = x, orig_x = 1000 - y
            return [xmin, 1000 - ymax, xmax, 1000 - ymin]
        elif self.image_rotate == 180:
            return [1000 - ymax, 1000 - xmax, 1000 - ymin, 1000 - xmin]
        elif self.image_rotate == 270:
            # CCW 90: image (y, x) came from original (1000-x, y)
            # So to go back: orig_y = 1000 - x, orig_x = y
            return [1000 - xmax, ymin, 1000 - xmin, ymax]

        return placement

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
        """Clear all overlays (reset canvas to background)."""
        self.canvas = self._make_bg()
