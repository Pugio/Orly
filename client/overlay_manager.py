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
    ):
        self.H_proj = H_proj
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.mode = mode
        self.canvas = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)

    def handle_tool_result(self, name: str, result: dict):
        """Process a tool result from the backend and render the overlay.

        Only handles 'project_overlay' tool results. Other tools are ignored.
        """
        if name != "project_overlay":
            return

        content_type = result.get("content_type", "annotation")
        placement = result.get("placement", [0, 0, 1000, 1000])
        title = result.get("title", "")
        data = result.get("data", {})

        overlay = self.render_overlay(content_type, placement, title, data)
        self.canvas = self.place_on_canvas(overlay, placement)

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
        # Use a reasonable pixel size based on the fraction of the canvas
        x_min, y_min, x_max, y_max = placement
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
        x_min, y_min, x_max, y_max = placement

        if self.mode == "projector" and self.H_proj is not None:
            # Map the four corners of the placement rectangle through H_proj
            src_corners = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
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

        else:
            # Screen mode: direct mapping from 0-1000 to pixel coords
            px_min = max(0, int(x_min / 1000.0 * self.proj_width))
            py_min = max(0, int(y_min / 1000.0 * self.proj_height))
            px_max = min(self.proj_width, int(x_max / 1000.0 * self.proj_width))
            py_max = min(self.proj_height, int(y_max / 1000.0 * self.proj_height))

            if px_max > px_min and py_max > py_min:
                resized = cv2.resize(overlay, (px_max - px_min, py_max - py_min))
                canvas[py_min:py_max, px_min:px_max] = resized

        return canvas

    def clear(self):
        """Clear all overlays (reset canvas to black)."""
        self.canvas = np.zeros(
            (self.proj_height, self.proj_width, 3), dtype=np.uint8
        )
