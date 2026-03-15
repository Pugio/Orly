"""
PoC 2 — Projector-to-table calibration.

Computes H_proj: table coordinates -> projector pixels.

Interactive calibration procedure:
1. Open a fullscreen black window on the projector display.
2. Project bright dots one at a time (white circle on black, grid pattern).
3. For each dot: capture camera frame, find centroid, apply H_cam to get table coords.
4. Collect correspondences: table_points <-> projector_points.
5. Compute H_proj = cv2.findHomography(table_points, projector_points).
6. Save to projector_homography.npz.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client.display import show_on_projector, get_projector_resolution


# ---------------------------------------------------------------------------
# Pure functions (tested)
# ---------------------------------------------------------------------------


def generate_calibration_grid(
    proj_width: int,
    proj_height: int,
    cols: int = 4,
    rows: int = 3,
    margin: float = 0.1,
) -> list[tuple[int, int]]:
    """Generate a grid of calibration points in projector pixel space.

    margin is the fraction of the display to leave as border (0.1 = 10% on each side).
    Returns list of (x, y) tuples in projector pixels, ordered row-major
    (left-to-right, top-to-bottom).
    """
    x_min = proj_width * margin
    x_max = proj_width * (1 - margin)
    y_min = proj_height * margin
    y_max = proj_height * (1 - margin)

    points = []
    for row in range(rows):
        if rows == 1:
            y = int(round((y_min + y_max) / 2))
        else:
            y = int(round(y_min + row * (y_max - y_min) / (rows - 1)))
        for col in range(cols):
            if cols == 1:
                x = int(round((x_min + x_max) / 2))
            else:
                x = int(round(x_min + col * (x_max - x_min) / (cols - 1)))
            points.append((x, y))

    return points


def create_dot_image(
    proj_width: int,
    proj_height: int,
    dot_center: tuple[int, int],
    dot_radius: int = 15,
) -> np.ndarray:
    """Create a black image with a single white dot at the given position.

    Returns BGR image (H, W, 3) of dtype uint8.
    """
    img = np.zeros((proj_height, proj_width, 3), dtype=np.uint8)
    cv2.circle(img, dot_center, dot_radius, (255, 255, 255), -1)
    return img


def find_bright_centroid(
    frame: np.ndarray,
    background: np.ndarray,
    threshold: int = 50,
) -> tuple[float, float] | None:
    """Find the centroid of the brightest region by subtracting background.

    Returns (x, y) in pixel coordinates, or None if no bright region found.
    """
    # Convert to grayscale
    fg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.int16)

    # Subtract background, clip to 0
    diff = np.clip(fg - bg, 0, 255).astype(np.uint8)

    # Threshold
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Find contours / moments
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return (cx, cy)


def camera_to_table(
    point: tuple[float, float],
    H_cam: np.ndarray,
) -> tuple[float, float]:
    """Transform a camera pixel coordinate to table coordinate using H_cam.

    Returns (x, y) in table space.
    """
    pt = np.array([[point]], dtype=np.float64)  # shape (1, 1, 2)
    transformed = cv2.perspectiveTransform(pt, H_cam)
    x, y = transformed[0, 0]
    return (float(x), float(y))


def compute_projector_homography(
    table_points: list[tuple[float, float]],
    projector_points: list[tuple[int, int]],
) -> np.ndarray:
    """Compute H_proj: table coords -> projector pixels.

    Returns 3x3 homography matrix.
    """
    src = np.array(table_points, dtype=np.float64)
    dst = np.array(projector_points, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst)
    return H


def table_to_projector(
    point: tuple[float, float],
    H_proj: np.ndarray,
) -> tuple[int, int]:
    """Transform table coordinates to projector pixel coordinates.

    Returns (x, y) in projector pixels as integers.
    """
    pt = np.array([[point]], dtype=np.float64)  # shape (1, 1, 2)
    transformed = cv2.perspectiveTransform(pt, H_proj)
    x, y = transformed[0, 0]
    return (int(round(x)), int(round(y)))


# ---------------------------------------------------------------------------
# Interactive calibration (not tested)
# ---------------------------------------------------------------------------


def detect_camera_homography(cap: cv2.VideoCapture) -> np.ndarray:
    """Detect ArUco markers and compute H_cam live.

    Reuses the same ArUco config as PoC 1 / generate_calibration_mat.py.
    Waits until all 4 markers are detected, then computes the homography.
    """
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    MARKER_IDS = [0, 1, 2, 3]
    CORNER_INDICES = {0: 2, 1: 3, 2: 0, 3: 1}
    # Output size matches PoC 1 — A4 aspect ratio
    OUTPUT_W, OUTPUT_H = 600, 848
    DST_POINTS = np.array([
        [0, 0], [OUTPUT_W, 0], [OUTPUT_W, OUTPUT_H], [0, OUTPUT_H],
    ], dtype=np.float32)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    print("Detecting ArUco markers to compute camera homography...")
    print("Make sure the calibration mat is visible to the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        detected = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                detected[int(mid)] = corners[i][0]

        if all(mid in detected for mid in MARKER_IDS):
            src_points = []
            for idx, mid in enumerate(MARKER_IDS):
                src_points.append(detected[mid][CORNER_INDICES[mid]])

            src = np.array(src_points, dtype=np.float32)
            H, _ = cv2.findHomography(src, DST_POINTS)
            if H is not None:
                print(f"  All 4 markers detected. H_cam computed.")
                return H

        found = sorted(detected.keys())
        print(f"  Found markers: {found} — need all of {MARKER_IDS}", end="\r")
        time.sleep(0.2)


def open_camera(url: str | None = None, webcam: int | None = None) -> cv2.VideoCapture:
    """Open a camera source."""
    if url:
        stream_url = f"{url.rstrip('/')}/video"
        print(f"Connecting to IP Webcam at {stream_url} ...")
        cap = cv2.VideoCapture(stream_url)
    elif webcam is not None:
        print(f"Opening local webcam {webcam} ...")
        cap = cv2.VideoCapture(webcam)
    else:
        print("Error: provide --url or --webcam")
        sys.exit(1)

    if not cap.isOpened():
        print("Failed to open video stream.")
        sys.exit(1)

    return cap


def load_camera_homography(path: str) -> np.ndarray:
    """Load H_cam from a .npz file."""
    data = np.load(path)
    return data["H_cam"]


def capture_background(cap: cv2.VideoCapture, warmup_frames: int = 10) -> np.ndarray:
    """Capture a background frame (with projector showing all-black)."""
    # Discard warmup frames to let auto-exposure settle
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture background frame.")
        sys.exit(1)
    return frame


def main():
    """Interactive projector calibration.

    Projects dots one at a time, captures via camera, computes H_proj.
    """
    parser = argparse.ArgumentParser(description="PoC 2: Projector calibration")
    parser.add_argument("--url", type=str, help="IP Webcam URL")
    parser.add_argument("--webcam", type=int, default=None, help="Local webcam index")
    parser.add_argument(
        "--h-cam", type=str, default=None,
        help="Path to camera homography .npz file. If omitted, detects ArUco markers live.",
    )
    parser.add_argument("--proj-width", type=int, default=1280)
    parser.add_argument("--proj-height", type=int, default=720)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument(
        "--output", type=str, default="projector_homography.npz",
        help="Output file for H_proj",
    )
    parser.add_argument(
        "--projector-display", type=str, default="TLM37E29",
        help="Name of the projector display (for window placement)",
    )
    args = parser.parse_args()

    if not args.url and args.webcam is None:
        parser.print_help()
        sys.exit(1)

    # Open camera
    cap = open_camera(url=args.url, webcam=args.webcam)

    # Load or compute camera homography
    if args.h_cam:
        H_cam = load_camera_homography(args.h_cam)
        print(f"Loaded camera homography from {args.h_cam}")
    else:
        H_cam = detect_camera_homography(cap)
        print("Computed camera homography from ArUco markers")

    # Generate calibration grid
    grid = generate_calibration_grid(
        args.proj_width, args.proj_height,
        cols=args.cols, rows=args.rows, margin=args.margin,
    )
    print(f"Calibration grid: {args.cols}x{args.rows} = {len(grid)} points")

    # Open fullscreen window on projector (extended display, not mirrored)
    win_name = "Projector Calibration"
    black = np.zeros((args.proj_height, args.proj_width, 3), dtype=np.uint8)
    show_on_projector(win_name, black, fullscreen=True)
    cv2.waitKey(500)

    print("Capturing background frame...")
    background = capture_background(cap)

    # Collect correspondences
    table_points = []
    projector_points = []

    for i, (px, py) in enumerate(grid):
        print(f"\nDot {i + 1}/{len(grid)}: projector ({px}, {py})")

        # Project the dot
        dot_img = create_dot_image(args.proj_width, args.proj_height, (px, py))
        show_on_projector(win_name, dot_img)
        cv2.waitKey(500)  # Wait for projector to display + camera to settle

        # Capture frame
        # Discard a few frames to account for camera lag
        for _ in range(5):
            cap.read()
            time.sleep(0.05)
        ret, frame = cap.read()

        if not ret:
            print(f"  WARNING: Failed to capture frame, skipping dot {i + 1}")
            continue

        # Find centroid
        centroid = find_bright_centroid(frame, background)
        if centroid is None:
            print(f"  WARNING: No bright region found, skipping dot {i + 1}")
            continue

        cam_x, cam_y = centroid
        print(f"  Camera centroid: ({cam_x:.1f}, {cam_y:.1f})")

        # Transform to table coordinates
        table_x, table_y = camera_to_table((cam_x, cam_y), H_cam)
        print(f"  Table coords: ({table_x:.1f}, {table_y:.1f})")

        table_points.append((table_x, table_y))
        projector_points.append((px, py))

        # Show black again before next dot
        show_on_projector(win_name, black)
        cv2.waitKey(200)

    # Need at least 4 correspondences
    if len(table_points) < 4:
        print(f"\nERROR: Only {len(table_points)} correspondences found, need at least 4.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(1)

    print(f"\nCollected {len(table_points)} correspondences.")

    # Compute projector homography
    H_proj = compute_projector_homography(table_points, projector_points)
    print(f"H_proj:\n{H_proj}")

    # Compute reprojection error
    errors = []
    for tp, pp in zip(table_points, projector_points):
        result = table_to_projector(tp, H_proj)
        err = np.sqrt((result[0] - pp[0]) ** 2 + (result[1] - pp[1]) ** 2)
        errors.append(err)
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"Reprojection error: mean={mean_err:.1f}px, max={max_err:.1f}px")

    # Save
    np.savez(
        args.output,
        H_proj=H_proj,
        table_points=np.array(table_points),
        projector_points=np.array(projector_points),
        proj_width=args.proj_width,
        proj_height=args.proj_height,
    )
    print(f"Saved projector homography to {args.output}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
