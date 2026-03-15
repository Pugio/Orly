"""
Generate a printable A4 calibration mat with 4 ArUco markers at the corners.
The markers define the table coordinate system.

Usage:
    pip install opencv-contrib-python numpy Pillow
    python generate_calibration_mat.py

Output:
    calibration_mat.png  — print this on A4 paper
"""

import cv2
import numpy as np

# --- Config ---
# A4 at 150 DPI (good enough for printing, keeps file small)
DPI = 150
PAGE_W_MM, PAGE_H_MM = 210, 297
PAGE_W_PX = int(PAGE_W_MM / 25.4 * DPI)
PAGE_H_PX = int(PAGE_H_MM / 25.4 * DPI)

MARKER_SIZE_MM = 30
MARKER_SIZE_PX = int(MARKER_SIZE_MM / 25.4 * DPI)

MARGIN_MM = 15
MARGIN_PX = int(MARGIN_MM / 25.4 * DPI)

# ArUco dictionary — 4x4_50 is small and fast to detect
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Marker IDs for the four corners (top-left, top-right, bottom-right, bottom-left)
MARKER_IDS = [0, 1, 2, 3]

def main():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # White page
    page = np.ones((PAGE_H_PX, PAGE_W_PX), dtype=np.uint8) * 255

    # Corner positions (top-left corner of each marker)
    positions = [
        (MARGIN_PX, MARGIN_PX),                                          # ID 0: top-left
        (PAGE_W_PX - MARGIN_PX - MARKER_SIZE_PX, MARGIN_PX),             # ID 1: top-right
        (PAGE_W_PX - MARGIN_PX - MARKER_SIZE_PX,
         PAGE_H_PX - MARGIN_PX - MARKER_SIZE_PX),                        # ID 2: bottom-right
        (MARGIN_PX, PAGE_H_PX - MARGIN_PX - MARKER_SIZE_PX),             # ID 3: bottom-left
    ]

    for marker_id, (x, y) in zip(MARKER_IDS, positions):
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE_PX)
        page[y:y + MARKER_SIZE_PX, x:x + MARKER_SIZE_PX] = marker_img

    # Draw a thin border connecting the inner corners of the markers
    # to show the "active area"
    page_color = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    inner_corners = [
        (MARGIN_PX + MARKER_SIZE_PX, MARGIN_PX + MARKER_SIZE_PX),
        (PAGE_W_PX - MARGIN_PX, MARGIN_PX + MARKER_SIZE_PX),
        (PAGE_W_PX - MARGIN_PX, PAGE_H_PX - MARGIN_PX),
        (MARGIN_PX + MARKER_SIZE_PX, PAGE_H_PX - MARGIN_PX),
    ]
    for i in range(4):
        pt1 = inner_corners[i]
        pt2 = inner_corners[(i + 1) % 4]
        cv2.line(page_color, pt1, pt2, (200, 200, 200), 1)

    # Label each marker
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for marker_id, (x, y) in zip(MARKER_IDS, positions):
        label_pos = (x, y + MARKER_SIZE_PX + 15)
        cv2.putText(page_color, f"ID {marker_id}", label_pos,
                    font, font_scale, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.imwrite("calibration_mat.png", page_color)
    print(f"Saved calibration_mat.png ({PAGE_W_PX}x{PAGE_H_PX}px, print at A4)")
    print(f"Marker size: {MARKER_SIZE_MM}mm, Margin: {MARGIN_MM}mm")
    print(f"Marker IDs: TL={MARKER_IDS[0]} TR={MARKER_IDS[1]} BR={MARKER_IDS[2]} BL={MARKER_IDS[3]}")


if __name__ == "__main__":
    main()
