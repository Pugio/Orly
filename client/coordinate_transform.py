"""Central authority for all coordinate and orientation transforms.

This module is the SINGLE place that handles the relationship between
three coordinate spaces:

1. **Gemini space** — 0-1000, the rotated image Gemini actually sees.
   Gemini returns [ymin, xmin, ymax, xmax] placements in this space.

2. **Table space** — 0-1000, aligned with ArUco markers.
   M0=(0,0), M1=(1000,0), M2=(1000,1000), M3=(0,1000).
   This is the canonical coordinate system for the physical table.

3. **Display space** — projector or screen pixels.
   In projector mode, H_proj maps table coords to projector pixels.
   In screen mode, table coords map linearly to pixel coords.

Physical setup (with --rotate 270):
                    Projector
           M1                    M2
  camera
           M0                    M3
                 Human

Table space has M0 at top-left of the rectified image. The --rotate flag
rotates the image before Gemini sees it so that "bottom" = human's side.

Key principle: H_proj (from calibration) encodes the geometric
position mapping between table and projector — where things land on
the table. It does NOT handle content orientation. Since the projector
sits opposite the human, overlay content is flipped 180° in
orient_overlay() so text/images are readable from the human's side.
"""

from __future__ import annotations

import cv2
import numpy as np


class CoordinateTransform:
    """Manages all coordinate and image orientation transforms.

    Args:
        rotate: CW rotation applied to camera image before Gemini sees it
                (0, 90, 180, 270).
        H_proj: 3x3 homography mapping table (x,y) to projector pixels.
                None for screen mode.
        display_width: Output display width in pixels.
        display_height: Output display height in pixels.
        mode: "projector" or "screen".
    """

    def __init__(
        self,
        rotate: int = 0,
        H_proj: np.ndarray | None = None,
        display_width: int = 1280,
        display_height: int = 720,
        mode: str = "screen",
    ):
        self.rotate = rotate
        self.H_proj = H_proj
        self.display_width = display_width
        self.display_height = display_height
        self.mode = mode

    # ------------------------------------------------------------------
    # 1. Coordinate transforms
    # ------------------------------------------------------------------

    def gemini_to_table(self, placement: list) -> list:
        """Convert Gemini's [ymin, xmin, ymax, xmax] to table space.

        Reverses the CW rotation that was applied to the camera image
        before Gemini saw it.
        """
        ymin, xmin, ymax, xmax = placement

        if self.rotate == 90:
            # CW 90 forward: (y, x) → (x, 1000-y)
            # Inverse: (yr, xr) → (y=1000-xr, x=yr)
            return [1000 - xmax, ymin, 1000 - xmin, ymax]
        elif self.rotate == 180:
            return [1000 - ymax, 1000 - xmax, 1000 - ymin, 1000 - xmin]
        elif self.rotate == 270:
            # CCW 90 forward: (y, x) → (1000-x, y)
            # Inverse: (yr, xr) → (y=xr, x=1000-yr)
            return [xmin, 1000 - ymax, xmax, 1000 - ymin]

        return placement

    def placement_to_pixels(self, placement: list) -> tuple[int, int, int, int]:
        """Convert table-space placement to display pixel bounds.

        Returns (px_min, py_min, px_max, py_max) in display pixels.
        Only used for screen mode (direct mapping).
        """
        y_min, x_min, y_max, x_max = placement
        px_min = max(0, int(x_min / 1000.0 * self.display_width))
        py_min = max(0, int(y_min / 1000.0 * self.display_height))
        px_max = min(self.display_width, int(x_max / 1000.0 * self.display_width))
        py_max = min(self.display_height, int(y_max / 1000.0 * self.display_height))
        return px_min, py_min, px_max, py_max

    def placement_pixel_size(self, placement: list) -> tuple[int, int]:
        """Return (width, height) in pixels for a placement box."""
        y_min, x_min, y_max, x_max = placement
        w = max(1, int((x_max - x_min) / 1000.0 * self.display_width))
        h = max(1, int((y_max - y_min) / 1000.0 * self.display_height))
        return w, h

    # ------------------------------------------------------------------
    # 2. Image orientation
    # ------------------------------------------------------------------

    def orient_overlay(self, overlay: np.ndarray) -> np.ndarray:
        """Rotate overlay content so it's readable from the human's perspective.

        Two independent rotations are applied:

        1. Camera rotation (self.rotate): aligns overlay content with the
           table coordinate system used by Gemini. This is the same rotation
           applied to camera frames before Gemini sees them.

        2. Projector flip (180°): in projector mode, the projector sits
           opposite the human, so content must be flipped 180° to be
           readable from the human's side. H_proj handles the geometric
           position mapping (where things land on the table), but NOT
           the content orientation — that's this flip's job.
        """
        result = overlay
        if self.rotate == 90:
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotate == 180:
            result = cv2.rotate(result, cv2.ROTATE_180)
        elif self.rotate == 270:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

        # In projector mode, human sits opposite the projector — flip content.
        if self.mode == "projector":
            result = cv2.rotate(result, cv2.ROTATE_180)

        return result

    # ------------------------------------------------------------------
    # 3. Canvas placement
    # ------------------------------------------------------------------

    def place_on_canvas(
        self,
        canvas: np.ndarray,
        overlay: np.ndarray,
        placement: list,
        white_bg: bool = False,
    ) -> np.ndarray:
        """Place an oriented overlay onto the display canvas.

        In projector mode: uses H_proj to perspective-warp the overlay
        from table coordinates to projector pixels.

        In screen mode: maps table 0-1000 coords to pixel coordinates
        with a simple linear scale.

        Args:
            canvas: The display canvas (copied, not modified in place).
            overlay: The overlay image (already oriented via orient_overlay).
            placement: [ymin, xmin, ymax, xmax] in table space (0-1000).
            white_bg: If True, only composite non-black pixels.

        Returns:
            Updated canvas.
        """
        canvas = canvas.copy()

        if self.mode == "projector" and self.H_proj is not None:
            canvas = self._place_projector(canvas, overlay, placement, white_bg)
        else:
            canvas = self._place_screen(canvas, overlay, placement, white_bg)

        return canvas

    def _place_projector(
        self,
        canvas: np.ndarray,
        overlay: np.ndarray,
        placement: list,
        white_bg: bool,
    ) -> np.ndarray:
        """Place overlay using H_proj perspective warp."""
        y_min, x_min, y_max, x_max = placement

        # Placement is [y,x] but H_proj was calibrated with (x,y) input.
        src_corners = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min],
        ], dtype=np.float64).reshape(1, 4, 2)
        dst_corners = cv2.perspectiveTransform(src_corners, self.H_proj)
        dst_corners = dst_corners.reshape(4, 2)

        oh, ow = overlay.shape[:2]
        # Must match src_corners order: TL, BL, BR, TR
        overlay_corners = np.array(
            [[0, 0], [0, oh], [ow, oh], [ow, 0]], dtype=np.float32
        )
        dst_xy = dst_corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(overlay_corners, dst_xy)
        warped = cv2.warpPerspective(
            overlay, M, (self.display_width, self.display_height)
        )

        # Composite: non-black warped pixels onto canvas
        threshold = 30 if white_bg else 0
        mask = warped.sum(axis=2) > threshold
        canvas[mask] = warped[mask]
        return canvas

    def _place_screen(
        self,
        canvas: np.ndarray,
        overlay: np.ndarray,
        placement: list,
        white_bg: bool,
    ) -> np.ndarray:
        """Place overlay using direct pixel mapping."""
        px_min, py_min, px_max, py_max = self.placement_to_pixels(placement)

        if px_max > px_min and py_max > py_min:
            resized = cv2.resize(overlay, (px_max - px_min, py_max - py_min))
            if white_bg:
                mask = resized.sum(axis=2) > 30
                region = canvas[py_min:py_max, px_min:px_max]
                region[mask] = resized[mask]
            else:
                canvas[py_min:py_max, px_min:px_max] = resized

        return canvas
