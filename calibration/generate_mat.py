#!/usr/bin/env python3
"""
Generate a printable calibration mat with 4 ArUco markers at the corners.
The markers define the table coordinate system for TableLight.

Supports standard paper sizes (A4, US Letter, A3, Tabloid) and custom
dimensions. Output is a PNG at the specified DPI.

Usage:
    python -m calibration.generate_mat                     # A4 (default)
    python -m calibration.generate_mat --paper letter       # US Letter
    python -m calibration.generate_mat --paper a3           # A3
    python -m calibration.generate_mat --paper tabloid      # Tabloid (11x17)
    python -m calibration.generate_mat --width 400 --height 600  # Custom (mm)
    python -m calibration.generate_mat --dpi 300            # Higher resolution

Output:
    calibration/calibration_mat.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# --- Paper sizes in mm (width x height, portrait orientation) ---
PAPER_SIZES: dict[str, tuple[float, float]] = {
    "a4": (210, 297),
    "a3": (297, 420),
    "letter": (215.9, 279.4),
    "tabloid": (279.4, 431.8),
    "legal": (215.9, 355.6),
}

# ArUco dictionary — 4x4_50 is small and fast to detect
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Marker IDs for the four corners (top-left, top-right, bottom-right, bottom-left)
MARKER_IDS = [0, 1, 2, 3]


def generate_mat(
    page_w_mm: float = 210,
    page_h_mm: float = 297,
    dpi: int = 150,
    marker_size_mm: float = 25,
    margin_mm: float = 5,
    output_path: str | Path = "calibration/calibration_mat.png",
) -> Path:
    """Generate a calibration mat PNG.

    Args:
        page_w_mm: Page width in mm.
        page_h_mm: Page height in mm.
        dpi: Print resolution (150 is fine for markers, 300 for crisp lines).
        marker_size_mm: ArUco marker side length in mm. 20-30mm recommended.
        margin_mm: Distance from page edge to marker edge in mm. 5mm minimum
            for most printers; increase to 7-8 if your printer clips corners.
        output_path: Where to save the PNG.

    Returns:
        Path to the saved PNG.
    """
    mm_to_px = dpi / 25.4
    page_w_px = int(page_w_mm * mm_to_px)
    page_h_px = int(page_h_mm * mm_to_px)
    marker_px = int(marker_size_mm * mm_to_px)
    margin_px = int(margin_mm * mm_to_px)

    # Validate that markers + margins fit on the page
    min_page_dim = 2 * margin_mm + 2 * marker_size_mm + 10  # 10mm min active area
    if page_w_mm < min_page_dim or page_h_mm < min_page_dim:
        print(
            f"Error: page too small ({page_w_mm}x{page_h_mm}mm). "
            f"Need at least {min_page_dim}x{min_page_dim}mm with "
            f"{marker_size_mm}mm markers and {margin_mm}mm margins.",
            file=sys.stderr,
        )
        sys.exit(1)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # White page
    page = np.ones((page_h_px, page_w_px), dtype=np.uint8) * 255

    # Corner positions (top-left pixel of each marker)
    positions = [
        (margin_px, margin_px),  # ID 0: top-left
        (page_w_px - margin_px - marker_px, margin_px),  # ID 1: top-right
        (page_w_px - margin_px - marker_px, page_h_px - margin_px - marker_px),  # ID 2: bottom-right
        (margin_px, page_h_px - margin_px - marker_px),  # ID 3: bottom-left
    ]

    for marker_id, (x, y) in zip(MARKER_IDS, positions):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        page[y : y + marker_px, x : x + marker_px] = marker_img

    # Draw a thin border connecting the inner corners of the markers
    # to visualize the "active area"
    page_color = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    inner_corners = [
        (margin_px + marker_px, margin_px + marker_px),
        (page_w_px - margin_px, margin_px + marker_px),
        (page_w_px - margin_px, page_h_px - margin_px),
        (margin_px + marker_px, page_h_px - margin_px),
    ]
    for i in range(4):
        pt1 = inner_corners[i]
        pt2 = inner_corners[(i + 1) % 4]
        cv2.line(page_color, pt1, pt2, (200, 200, 200), 1)

    # Label each marker
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for marker_id, (x, y) in zip(MARKER_IDS, positions):
        label_pos = (x, y + marker_px + 15)
        cv2.putText(
            page_color, f"ID {marker_id}", label_pos, font, font_scale, (100, 100, 100), 1, cv2.LINE_AA
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), page_color)

    # Compute active area
    active_w_mm = page_w_mm - 2 * margin_mm - marker_size_mm
    active_h_mm = page_h_mm - 2 * margin_mm - marker_size_mm

    print(f"Saved {output} ({page_w_px}x{page_h_px}px at {dpi} DPI)")
    print(f"  Page:    {page_w_mm} x {page_h_mm} mm")
    print(f"  Markers: {marker_size_mm}mm, Margin: {margin_mm}mm")
    print(f"  Active:  {active_w_mm:.1f} x {active_h_mm:.1f} mm "
          f"({active_w_mm/25.4:.1f}\" x {active_h_mm/25.4:.1f}\")")
    print(f"  IDs:     TL={MARKER_IDS[0]} TR={MARKER_IDS[1]} BR={MARKER_IDS[2]} BL={MARKER_IDS[3]}")

    return output


def main() -> None:
    paper_names = ", ".join(PAPER_SIZES.keys())

    parser = argparse.ArgumentParser(
        description="Generate a printable ArUco calibration mat for TableLight.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Built-in paper sizes: {paper_names}\n\n"
               "Use --width and --height for custom dimensions (in mm).\n"
               "Tip: if your printer clips corners, increase --margin to 7 or 8.",
    )
    parser.add_argument(
        "--paper", "-p",
        choices=list(PAPER_SIZES.keys()),
        default=None,
        help="Paper size preset (default: a4).",
    )
    parser.add_argument("--width", "-W", type=float, default=None, help="Custom page width in mm.")
    parser.add_argument("--height", "-H", type=float, default=None, help="Custom page height in mm.")
    parser.add_argument("--dpi", type=int, default=150, help="Print resolution (default: 150).")
    parser.add_argument("--marker-size", type=float, default=25, help="Marker side length in mm (default: 25).")
    parser.add_argument("--margin", type=float, default=5, help="Page edge margin in mm (default: 5).")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path.")

    args = parser.parse_args()

    # Determine page dimensions
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            parser.error("--width and --height must both be specified for custom sizes.")
        page_w, page_h = args.width, args.height
        size_label = "custom"
    else:
        paper = args.paper or "a4"
        page_w, page_h = PAPER_SIZES[paper]
        size_label = paper

    # Default output path
    if args.output is None:
        output = f"calibration/calibration_mat_{size_label}.png"
    else:
        output = args.output

    generate_mat(
        page_w_mm=page_w,
        page_h_mm=page_h,
        dpi=args.dpi,
        marker_size_mm=args.marker_size,
        margin_mm=args.margin,
        output_path=output,
    )


if __name__ == "__main__":
    main()
