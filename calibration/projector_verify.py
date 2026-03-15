"""
Projector calibration verification.

Loads H_proj from file and projects test patterns at known table coordinates
so the user can visually check alignment.

Usage:
    python projector_verify.py --homography projector_homography.npz
"""

import argparse
import sys

import cv2
import numpy as np

from calibration.projector_calibrate import (
    create_dot_image,
    table_to_projector,
)


def create_verification_pattern(
    proj_width: int,
    proj_height: int,
    H_proj: np.ndarray,
) -> np.ndarray:
    """Create a verification image with dots at known table coordinates.

    Projects dots at a regular grid of table coordinates (every 200 units
    in the 0-1000 space) so the user can check alignment against the
    calibration mat.
    """
    img = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)

    # Grid of table coordinates to verify
    table_coords = []
    for ty in range(0, 1001, 200):
        for tx in range(0, 1001, 200):
            table_coords.append((float(tx), float(ty)))

    for tx, ty in table_coords:
        px, py = table_to_projector((tx, ty), H_proj)

        # Skip if outside projector bounds
        if px < 0 or px >= proj_width or py < 0 or py >= proj_height:
            continue

        # Draw dot — color-coded by position
        # Corners: red, center: green, edges: blue
        if (tx in (0, 1000)) and (ty in (0, 1000)):
            color = (0, 0, 255)  # Red corners
        elif tx == 500 and ty == 500:
            color = (0, 255, 0)  # Green center
        else:
            color = (255, 200, 0)  # Cyan-ish for others

        cv2.circle(img, (px, py), 8, color, -1)

        # Label with table coordinates
        label = f"({int(tx)},{int(ty)})"
        cv2.putText(
            img, label, (px + 12, py + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
        )

    return img


def create_crosshair_pattern(
    proj_width: int,
    proj_height: int,
    H_proj: np.ndarray,
) -> np.ndarray:
    """Create a crosshair pattern at the table center (500, 500)."""
    img = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)

    cx, cy = table_to_projector((500.0, 500.0), H_proj)

    # Horizontal and vertical lines through center
    line_color = (0, 255, 0)
    cv2.line(img, (cx - 100, cy), (cx + 100, cy), line_color, 1)
    cv2.line(img, (cx, cy - 100), (cx, cy + 100), line_color, 1)
    cv2.circle(img, (cx, cy), 5, line_color, -1)

    # Corner markers
    corners = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
    for tx, ty in corners:
        px, py = table_to_projector((float(tx), float(ty)), H_proj)
        if 0 <= px < proj_width and 0 <= py < proj_height:
            cv2.drawMarker(
                img, (px, py), (0, 0, 255),
                cv2.MARKER_CROSS, 20, 2,
            )

    return img


def main():
    parser = argparse.ArgumentParser(description="Verify projector calibration")
    parser.add_argument(
        "--homography", type=str, default="projector_homography.npz",
        help="Path to projector homography .npz file",
    )
    parser.add_argument("--proj-width", type=int, default=1280)
    parser.add_argument("--proj-height", type=int, default=720)
    args = parser.parse_args()

    # Load homography
    data = np.load(args.homography)
    H_proj = data["H_proj"]
    print(f"Loaded H_proj from {args.homography}")
    print(f"H_proj:\n{H_proj}")

    # Use stored dimensions if available, otherwise use args
    proj_width = int(data.get("proj_width", args.proj_width))
    proj_height = int(data.get("proj_height", args.proj_height))
    print(f"Projector resolution: {proj_width}x{proj_height}")

    # Create verification patterns
    grid_img = create_verification_pattern(proj_width, proj_height, H_proj)
    crosshair_img = create_crosshair_pattern(proj_width, proj_height, H_proj)

    # Show on projector
    win_name = "Projector Verification"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    patterns = [
        ("Grid pattern (dots at every 200 table units)", grid_img),
        ("Crosshair at center + corner markers", crosshair_img),
    ]
    current = 0

    print("\nControls:")
    print("  SPACE / n — next pattern")
    print("  p         — previous pattern")
    print("  q         — quit")
    print(f"\nShowing: {patterns[current][0]}")

    cv2.imshow(win_name, patterns[current][1])

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord(" "), ord("n")):
            current = (current + 1) % len(patterns)
            print(f"Showing: {patterns[current][0]}")
            cv2.imshow(win_name, patterns[current][1])
        elif key == ord("p"):
            current = (current - 1) % len(patterns)
            print(f"Showing: {patterns[current][0]}")
            cv2.imshow(win_name, patterns[current][1])

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
