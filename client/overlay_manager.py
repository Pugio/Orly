"""Receives tool results from the backend, renders and displays overlays.

Manages a canvas (black background) that overlays are placed onto.
In projector mode, uses H_proj to map table coordinates to projector pixels.
In screen mode, maps Gemini's 0-1000 coordinate system directly to pixel space.
"""

import cv2
import numpy as np

from client.renderer.graph import render_graph
from client.renderer.annotation import render_annotation
from client.renderer.highlight import render_highlight


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
    ):
        self.H_proj = H_proj
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.mode = mode
        self.image_rotate = image_rotate  # how the image was rotated before Gemini saw it
        self.canvas = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)

    def handle_tool_result(self, name: str, result: dict):
        """Process a tool result from the backend and render the overlay.

        Only handles 'project_overlay' tool results. Other tools are ignored.
        """
        if name != "project_overlay":
            return

        content_type = result.get("content_type", "annotation")
        placement = list(result.get("placement", [0, 0, 1000, 1000]))
        title = result.get("title", "")
        data = result.get("data", {})

        # TODO: Un-rotation disabled until coordinate system is sorted out
        # if self.image_rotate != 0:
        #     original = list(placement)
        #     placement = self._unrotate_placement(placement)
        #     print(f"[OverlayManager] Un-rotated {original} -> {placement}")

        overlay = self.render_overlay(content_type, placement, title, data)

        # Flip overlay content for projector orientation.
        # The projector projects from behind the mat, so content appears
        # flipped from the viewer's perspective. Rotate 180° to compensate.
        if self.mode == "projector" and content_type != "highlight":
            overlay = cv2.rotate(overlay, cv2.ROTATE_180)

        self.canvas = self.place_on_canvas(overlay, placement)

        print(f"[OverlayManager] Rendered {content_type} at {placement}")

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
            # Map the four corners of the placement rectangle through H_proj.
            # Map placement rectangle corners through H_proj.
            src_corners = np.array([
                [y_min, x_min],
                [y_max, x_min],
                [y_max, x_max],
                [y_min, x_max],
            ], dtype=np.float64).reshape(1, 4, 2)
            dst_corners = cv2.perspectiveTransform(src_corners, self.H_proj)
            dst_corners = dst_corners.reshape(4, 2)

            # Get bounding box in projector pixels
            px_min = max(0, int(dst_corners[:, 0].min()))
            py_min = max(0, int(dst_corners[:, 1].min()))
            px_max = min(self.proj_width, int(dst_corners[:, 0].max()))
            py_max = min(self.proj_height, int(dst_corners[:, 1].max()))

            if px_max > px_min and py_max > py_min:
                resized = cv2.resize(overlay, (px_max - px_min, py_max - py_min))
                canvas[py_min:py_max, px_min:px_max] = resized
                use_direct = False
            else:
                print(f"[OverlayManager] WARNING: H_proj mapped outside bounds, "
                      f"falling back to direct mapping")

        if use_direct:
            # Screen mode: direct mapping from 0-1000 to pixel coords
            px_min = max(0, int(x_min / 1000.0 * self.proj_width))
            py_min = max(0, int(y_min / 1000.0 * self.proj_height))
            px_max = min(self.proj_width, int(x_max / 1000.0 * self.proj_width))
            py_max = min(self.proj_height, int(y_max / 1000.0 * self.proj_height))

            if px_max > px_min and py_max > py_min:
                resized = cv2.resize(overlay, (px_max - px_min, py_max - py_min))
                canvas[py_min:py_max, px_min:px_max] = resized

        return canvas

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

    def clear(self):
        """Clear all overlays (reset canvas to black)."""
        self.canvas = np.zeros(
            (self.proj_height, self.proj_width, 3), dtype=np.uint8
        )
