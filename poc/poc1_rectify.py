"""
PoC 1 — Camera-to-table homography via ArUco markers.

Reads a video stream from the "IP Webcam" Android app, detects four
ArUco markers on the printed calibration mat, computes a homography,
and displays a rectified (top-down) view of the table surface.

Setup:
    1. Install "IP Webcam" on your Android phone (free on Play Store).
    2. Print calibration_mat.png on A4 paper (run generate_calibration_mat.py first).
    3. Place the printed mat on your table.
    4. Mount your phone on a stand pointing at the table.
    5. Open IP Webcam → Start Server. Note the IP address shown.
    6. Run this script:

        pip install opencv-contrib-python numpy
        python poc1_rectify.py --url http://<PHONE_IP>:8080

Controls:
    q — quit
    s — save current rectified image to rectified_output.png
    d — toggle debug overlay (shows detected markers + homography grid)

Usage:
    python poc1_rectify.py --url http://192.168.1.100:8080
    python poc1_rectify.py --webcam 0   # or use a local webcam for testing
"""

import argparse
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client.display import show_on_laptop


# --- Configuration ---

# ArUco setup — must match generate_calibration_mat.py
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Expected marker IDs at each corner of the mat
# Order: top-left, top-right, bottom-right, bottom-left
MARKER_IDS = [0, 1, 2, 3]

# Output rectified image size (pixels)
# This is the "normalized table view" — aspect ratio roughly A4
OUTPUT_W = 600
OUTPUT_H = 848  # 600 * 297/210 ≈ 848 for A4 aspect ratio

# Which corner of each marker to use as the reference point.
# Each detected marker gives us 4 corners. We pick the corner
# closest to the center of the mat to define the "inner" boundary.
#
# ArUco corner order is: top-left, top-right, bottom-right, bottom-left
# of the marker itself. So:
#   Marker 0 (top-left of mat)     → use its bottom-right corner (index 2)
#   Marker 1 (top-right of mat)    → use its bottom-left corner  (index 3)
#   Marker 2 (bottom-right of mat) → use its top-left corner     (index 0)
#   Marker 3 (bottom-left of mat)  → use its top-right corner    (index 1)
CORNER_INDICES = {0: 2, 1: 3, 2: 0, 3: 1}

# Where those corners should map to in the output image
DST_POINTS = np.array([
    [0, 0],                 # marker 0 inner corner → top-left
    [OUTPUT_W, 0],          # marker 1 inner corner → top-right
    [OUTPUT_W, OUTPUT_H],   # marker 2 inner corner → bottom-right
    [0, OUTPUT_H],          # marker 3 inner corner → bottom-left
], dtype=np.float32)


def open_video_stream(url=None, webcam=None):
    """Open either an IP Webcam stream or a local webcam."""
    if url:
        # IP Webcam serves MJPEG at /video
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
        print("Failed to open video stream. Check the URL or webcam index.")
        sys.exit(1)

    print("Stream opened successfully.")
    return cap


def detect_markers(frame, detector):
    """Detect ArUco markers and return a dict of {id: corners}."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    result = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            result[int(marker_id)] = corners[i][0]  # shape (4, 2)

    return result


def compute_homography(detected_markers):
    """
    Given detected markers, compute the homography from camera space
    to the rectified output space.
    Returns the 3x3 homography matrix, or None if not enough markers.
    """
    src_points = []
    dst_points = []

    for idx, marker_id in enumerate(MARKER_IDS):
        if marker_id not in detected_markers:
            return None

        corners = detected_markers[marker_id]
        # Pick the inner corner of this marker
        corner_idx = CORNER_INDICES[marker_id]
        src_points.append(corners[corner_idx])
        dst_points.append(DST_POINTS[idx])

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    H, status = cv2.findHomography(src, dst)
    return H


def draw_debug_overlay(frame, detected_markers, H):
    """Draw detected markers and a projected grid on the camera frame."""
    debug = frame.copy()

    # Draw detected markers
    for marker_id, corners in detected_markers.items():
        pts = corners.astype(int)
        cv2.polylines(debug, [pts], True, (0, 255, 0), 2)
        center = pts.mean(axis=0).astype(int)
        cv2.putText(debug, f"ID {marker_id}", tuple(center),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Highlight the inner corner we're using
        if marker_id in CORNER_INDICES:
            inner = pts[CORNER_INDICES[marker_id]]
            cv2.circle(debug, tuple(inner), 6, (0, 0, 255), -1)

    # If we have a homography, project a grid back onto the camera view
    if H is not None:
        H_inv = np.linalg.inv(H)
        # Draw grid lines in rectified space, project to camera space
        for i in range(0, OUTPUT_W + 1, OUTPUT_W // 6):
            # Vertical lines
            pts_rect = np.array([[i, 0], [i, OUTPUT_H]], dtype=np.float32)
            pts_cam = cv2.perspectiveTransform(pts_rect.reshape(1, -1, 2), H_inv)
            pts_cam = pts_cam.reshape(-1, 2).astype(int)
            cv2.line(debug, tuple(pts_cam[0]), tuple(pts_cam[1]), (255, 100, 0), 1)

        for j in range(0, OUTPUT_H + 1, OUTPUT_H // 8):
            # Horizontal lines
            pts_rect = np.array([[0, j], [OUTPUT_W, j]], dtype=np.float32)
            pts_cam = cv2.perspectiveTransform(pts_rect.reshape(1, -1, 2), H_inv)
            pts_cam = pts_cam.reshape(-1, 2).astype(int)
            cv2.line(debug, tuple(pts_cam[0]), tuple(pts_cam[1]), (255, 100, 0), 1)

    return debug


def main():
    parser = argparse.ArgumentParser(description="PoC 1: Camera-to-table homography")
    parser.add_argument("--url", type=str, help="IP Webcam URL, e.g. http://192.168.1.100:8080")
    parser.add_argument("--webcam", type=int, default=None, help="Local webcam index (e.g. 0)")
    args = parser.parse_args()

    if not args.url and args.webcam is None:
        parser.print_help()
        sys.exit(1)

    cap = open_video_stream(url=args.url, webcam=args.webcam)

    # Set up ArUco detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    show_debug = True
    H_current = None
    frame_count = 0
    fps_time = time.time()

    print("\nControls: q=quit, s=save rectified image, h=save homography, d=toggle debug overlay")
    print("(Type command + Enter in terminal, or press key in OpenCV window)\n")

    # Background thread to read stdin without blocking
    key_queue = queue.Queue()

    def stdin_reader():
        while True:
            try:
                line = sys.stdin.readline().strip().lower()
                if line:
                    key_queue.put(ord(line[0]))
            except EOFError:
                break

    stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
    stdin_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost video stream, retrying...")
            time.sleep(0.5)
            continue

        # Detect markers
        detected = detect_markers(frame, detector)
        found_ids = sorted(detected.keys())

        # Compute homography if all 4 markers visible
        H = compute_homography(detected)
        if H is not None:
            H_current = H  # Cache last good homography

        # Status text
        if H_current is not None:
            status = "LOCKED" if H is not None else "USING CACHED"
            color = (0, 255, 0) if H is not None else (0, 200, 255)
        else:
            status = f"SEARCHING (found: {found_ids})"
            color = (0, 0, 255)

        # Build display: camera view (with optional debug) + rectified view
        if show_debug:
            cam_display = draw_debug_overlay(frame, detected, H_current)
        else:
            cam_display = frame.copy()

        # Status bar
        cv2.putText(cam_display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # FPS counter
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_time
            fps = 30 / elapsed if elapsed > 0 else 0
            fps_time = time.time()
            cv2.putText(cam_display, f"{fps:.0f} FPS", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Show camera view (resized to fit screen)
        cam_h, cam_w = cam_display.shape[:2]
        if cam_w > 800:
            scale = 800 / cam_w
            cam_display = cv2.resize(cam_display, None, fx=scale, fy=scale)
        show_on_laptop("Camera View", cam_display)

        # Show rectified view if we have a homography
        if H_current is not None:
            rectified = cv2.warpPerspective(frame, H_current, (OUTPUT_W, OUTPUT_H))
            show_on_laptop("Rectified Table View", rectified)

        # Handle keys — from OpenCV window OR terminal stdin
        key = cv2.waitKey(1) & 0xFF

        # Check for terminal input from background thread
        if key == 255:
            try:
                key = key_queue.get_nowait()
            except queue.Empty:
                pass

        if key == ord('q'):
            break
        elif key == ord('s') and H_current is not None:
            rectified = cv2.warpPerspective(frame, H_current, (OUTPUT_W, OUTPUT_H))
            cv2.imwrite("rectified_output.png", rectified)
            print("Saved rectified_output.png")
        elif key == ord('h') and H_current is not None:
            np.savez("camera_homography.npz", H_cam=H_current)
            print("Saved camera_homography.npz")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug overlay: {'ON' if show_debug else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
