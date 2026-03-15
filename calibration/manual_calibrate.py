"""Manual projector calibration — click where you see each dot.

Projects dots one at a time. For each dot, shows the camera view
and you click where the dot lands on the mat. 4+ clicks = homography.

Usage:
    uv run python calibration/manual_calibrate.py --url http://192.168.0.114:8080
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from client.display import show_on_projector
from calibration.projector_calibrate import (
    generate_calibration_grid,
    create_dot_image,
    compute_projector_homography,
    table_to_projector,
)


# ArUco config (same as poc1 / camera.py)
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]
CORNER_INDICES = {0: 2, 1: 3, 2: 0, 3: 1}


def detect_and_rectify(cap, detector):
    """Capture a frame, detect markers, compute H_cam, return frame + markers + H_cam."""
    DST = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)

    for _ in range(5):  # flush buffer
        cap.read()

    ret, frame = cap.read()
    if not ret:
        return None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    detected = {}
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            detected[int(mid)] = corners[i][0]

    if not all(mid in detected for mid in MARKER_IDS):
        return frame, detected, None

    src = np.array([detected[mid][CORNER_INDICES[mid]] for mid in MARKER_IDS], dtype=np.float32)
    H_cam, _ = cv2.findHomography(src, DST)

    return frame, detected, H_cam


def main():
    parser = argparse.ArgumentParser(description="Manual projector calibration")
    parser.add_argument("--url", type=str, help="IP Webcam URL")
    parser.add_argument("--webcam", type=int, default=None)
    parser.add_argument("--proj-width", type=int, default=1280)
    parser.add_argument("--proj-height", type=int, default=720)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="projector_homography.npz")
    args = parser.parse_args()

    # Open camera
    if args.url:
        stream_url = f"{args.url.rstrip('/')}/video"
        cap = cv2.VideoCapture(stream_url)
    elif args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
    else:
        print("Provide --url or --webcam")
        sys.exit(1)

    if not cap.isOpened():
        print("Failed to open camera")
        sys.exit(1)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    # Generate grid
    grid = generate_calibration_grid(
        args.proj_width, args.proj_height,
        cols=args.cols, rows=args.rows, margin=args.margin,
    )
    print(f"Calibration: {args.cols}x{args.rows} = {len(grid)} dots")
    print()
    print("For each dot:")
    print("  - A dot appears on the projector")
    print("  - The rectified camera view shows on your laptop")
    print("  - CLICK where the dot lands on the mat in the camera view")
    print("  - Press SPACE to skip a dot (if it's off the mat)")
    print("  - Press Q to finish early")
    print()

    # Show black on projector
    win_proj = "Projector"
    black = np.zeros((args.proj_height, args.proj_width, 3), dtype=np.uint8)
    show_on_projector(win_proj, black, fullscreen=True)
    cv2.waitKey(500)

    click_point = [None]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point[0] = (x, y)

    cam_win = "Click where the dot is"
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(cam_win, on_click)

    table_points = []
    projector_points = []

    for i, (px, py) in enumerate(grid):
        print(f"\nDot {i+1}/{len(grid)}: projector ({px}, {py})")

        # Project dot — waitKey(500) forces OpenCV to actually render it
        dot_img = create_dot_image(args.proj_width, args.proj_height, (px, py), dot_radius=40)
        show_on_projector(win_proj, dot_img)
        cv2.waitKey(500)
        time.sleep(1.0)  # extra time for projector + camera

        # Take snapshot — retry until all 4 markers are visible.
        import urllib.request
        snapshot_url = f"{args.url.rstrip('/')}/photo.jpg"
        frame = None
        detected = {}
        H_cam = None
        DST = np.array([[0,0],[1000,0],[1000,1000],[0,1000]], dtype=np.float32)

        while True:
            try:
                resp = urllib.request.urlopen(snapshot_url)
                img_array = np.frombuffer(resp.read(), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"  Snapshot failed: {e}")
                frame = None

            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                detected = {}
                if ids is not None:
                    for ii, mid in enumerate(ids.flatten()):
                        detected[int(mid)] = corners[ii][0]

                if all(mid in detected for mid in MARKER_IDS):
                    src = np.array([detected[mid][CORNER_INDICES[mid]] for mid in MARKER_IDS], dtype=np.float32)
                    H_cam, _ = cv2.findHomography(src, DST)
                    break

            print("  Waiting for all 4 markers to be visible...")
            cv2.waitKey(1000)

        assert H_cam is not None
        # Show raw camera frame with markers labeled.
        # User clicks where the dot is. We transform the click through H_cam
        # to get table coordinates.
        display = frame.copy()
        # Draw detected markers for reference
        for mid in MARKER_IDS:
            if mid in detected:
                inner = detected[mid][CORNER_INDICES[mid]]
                cv2.circle(display, (int(inner[0]), int(inner[1])), 8, (0, 255, 0), -1)
                cv2.putText(display, f"M{mid}", (int(inner[0])+10, int(inner[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Resize for display
        h, w = display.shape[:2]
        scale = min(900 / w, 700 / h)
        display_resized = cv2.resize(display, (int(w * scale), int(h * scale)))

        cv2.putText(display_resized, f"Dot {i+1}: Click where the dot is (SPACE=skip, Q=done)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        click_point[0] = None
        cv2.imshow(cam_win, display_resized)

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord(' '):
                print("  Skipped")
                break
            elif key == ord('q'):
                print("  Done early")
                break
            if click_point[0] is not None:
                # Scale click back to original frame coordinates
                cx = click_point[0][0] / scale
                cy = click_point[0][1] / scale
                # Transform through H_cam to get table coordinates
                pt = np.array([[[cx, cy]]], dtype=np.float64)
                table_pt = cv2.perspectiveTransform(pt, H_cam)[0, 0]
                tx, ty = float(table_pt[0]), float(table_pt[1])
                print(f"  Clicked: camera ({cx:.0f}, {cy:.0f}) -> table ({tx:.0f}, {ty:.0f})")
                table_points.append((tx, ty))
                projector_points.append((px, py))

                # Show confirmation
                cv2.circle(display_resized, click_point[0], 8, (0, 0, 255), -1)
                cv2.imshow(cam_win, display_resized)
                cv2.waitKey(500)
                break

        if key == ord('q'):
            break

        # Clear projector
        show_on_projector(win_proj, black)
        cv2.waitKey(100)

    cv2.destroyWindow(cam_win)

    if len(table_points) < 4:
        print(f"\nOnly {len(table_points)} points — need at least 4")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(1)

    # Compute homography
    H_proj = compute_projector_homography(table_points, projector_points)
    print(f"\nH_proj computed from {len(table_points)} points:")
    print(H_proj)

    # Reprojection error
    errors = []
    for tp, pp in zip(table_points, projector_points):
        result = table_to_projector(tp, H_proj)
        err = np.sqrt((result[0] - pp[0]) ** 2 + (result[1] - pp[1]) ** 2)
        errors.append(err)
    print(f"Reprojection error: mean={np.mean(errors):.1f}px, max={np.max(errors):.1f}px")

    # Verify coverage
    print("\nTable corners -> Projector:")
    for pt in [(0, 0), (1000, 0), (0, 1000), (1000, 1000), (500, 500)]:
        r = table_to_projector(pt, H_proj)
        ok = 0 <= r[0] <= args.proj_width and 0 <= r[1] <= args.proj_height
        print(f"  ({pt[0]:4d},{pt[1]:4d}) -> ({r[0]:4d},{r[1]:4d}) {'OK' if ok else 'OUT'}")

    # Save
    np.savez(
        args.output,
        H_proj=H_proj,
        table_points=np.array(table_points),
        projector_points=np.array(projector_points),
        proj_width=args.proj_width,
        proj_height=args.proj_height,
    )
    print(f"\nSaved to {args.output}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
