"""Camera capture + ArUco homography for the edge client.

Captures frames from IP Webcam or local webcam, detects ArUco markers,
computes a homography, and returns rectified (top-down) JPEG frames
ready for the backend.

Pure functions (detect_markers, compute_homography, rectify_frame, encode_jpeg)
are extracted for testability. The CameraCapture class wraps them for use in
the main client loop.
"""

import cv2
import numpy as np

# ArUco setup — must match calibration mat
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = [0, 1, 2, 3]

# Which corner of each marker to use as the reference point (inner corner).
# ArUco corner order: top-left, top-right, bottom-right, bottom-left of the marker.
#   Marker 0 (top-left of mat)     -> bottom-right corner (index 2)
#   Marker 1 (top-right of mat)    -> bottom-left corner  (index 3)
#   Marker 2 (bottom-right of mat) -> top-left corner     (index 0)
#   Marker 3 (bottom-left of mat)  -> top-right corner    (index 1)
CORNER_INDICES = {0: 2, 1: 3, 2: 0, 3: 1}


# ---------------------------------------------------------------------------
# Pure functions (tested)
# ---------------------------------------------------------------------------


def detect_markers(frame: np.ndarray, detector: cv2.aruco.ArucoDetector) -> dict:
    """Detect ArUco markers and return a dict of {id: corners}.

    Each corners entry has shape (4, 2) — the four corner points of the marker.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    result = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            result[int(marker_id)] = corners[i][0]  # shape (4, 2)

    return result


def compute_homography(
    detected_markers: dict,
    dst_points: np.ndarray,
) -> np.ndarray | None:
    """Compute homography from detected markers to destination points.

    Uses the inner corner of each expected marker (MARKER_IDS) as source points.
    Returns 3x3 homography matrix, or None if fewer than 4 markers are detected.

    Args:
        detected_markers: {marker_id: corners_array} from detect_markers.
        dst_points: (4, 2) array of destination points for markers 0-3.
    """
    src_points = []
    dst_list = []

    for idx, marker_id in enumerate(MARKER_IDS):
        if marker_id not in detected_markers:
            return None

        corners = detected_markers[marker_id]
        corner_idx = CORNER_INDICES[marker_id]
        src_points.append(corners[corner_idx])
        dst_list.append(dst_points[idx])

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_list, dtype=np.float32)

    H, status = cv2.findHomography(src, dst)
    return H


def rectify_frame(
    frame: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Apply homography to get rectified top-down view.

    Args:
        frame: Input BGR image.
        H: 3x3 homography matrix.
        output_size: (width, height) of the output image.

    Returns:
        Rectified BGR image with shape (height, width, 3).
    """
    w, h = output_size
    return cv2.warpPerspective(frame, H, (w, h))


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode frame as JPEG bytes.

    Args:
        frame: BGR image (numpy array).
        quality: JPEG quality (1-100).

    Returns:
        JPEG-encoded bytes.
    """
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buf = cv2.imencode(".jpg", frame, params)
    if not success:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# CameraCapture class (wraps pure functions for the main client loop)
# ---------------------------------------------------------------------------


class CameraCapture:
    """Captures frames from IP Webcam, detects ArUco markers, computes homography.

    Usage:
        cam = CameraCapture(url="http://192.168.1.100:8080")
        cam.start()
        jpeg_bytes, H = cam.get_rectified_frame()
        cam.stop()
    """

    def __init__(
        self,
        url: str | None = None,
        webcam: int | None = None,
        output_size: tuple[int, int] = (768, 768),
        rotate: int = 0,
    ):
        self.url = url
        self.webcam = webcam
        self.output_size = output_size
        self.rotate = rotate  # CW rotation in degrees (0, 90, 180, 270)

        self.cap = None
        self.detector = None
        self.H_cached = None  # last-good homography

        # Destination points for the rectified output
        w, h = output_size
        self.dst_points = np.array([
            [0, 0], [w, 0], [w, h], [0, h],
        ], dtype=np.float32)

    def start(self):
        """Open video stream and initialize ArUco detector."""
        if self.url:
            stream_url = f"{self.url.rstrip('/')}/video"
            self.cap = cv2.VideoCapture(stream_url)
        elif self.webcam is not None:
            self.cap = cv2.VideoCapture(self.webcam)
        else:
            raise ValueError("Provide url or webcam index")

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video stream")

        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def get_rectified_frame(self) -> tuple[bytes | None, np.ndarray | None]:
        """Capture a frame, rectify it, return (jpeg_bytes, H_cam).

        Returns (None, None) if no frame available or no homography
        (current or cached).
        """
        if self.cap is None:
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Detect markers and compute homography
        detected = detect_markers(frame, self.detector)
        H = compute_homography(detected, self.dst_points)

        if H is not None:
            self.H_cached = H

        if self.H_cached is None:
            return None, None

        # Rectify and encode
        rectified = rectify_frame(frame, self.H_cached, self.output_size)

        # Rotate so text is upright for Gemini
        if self.rotate == 90:
            rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == 180:
            rectified = cv2.rotate(rectified, cv2.ROTATE_180)
        elif self.rotate == 270:
            rectified = cv2.rotate(rectified, cv2.ROTATE_90_COUNTERCLOCKWISE)

        jpeg_bytes = encode_jpeg(rectified)

        return jpeg_bytes, self.H_cached

    def stop(self):
        """Release camera."""
        if self.cap:
            self.cap.release()
            self.cap = None
