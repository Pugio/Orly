"""Tests for edge client modules: ws_client, camera, overlay_manager."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# WebSocket client tests
# ---------------------------------------------------------------------------


class TestTableLightClientSendAudio:
    """send_audio creates correct JSON with base64 data."""

    @pytest.fixture
    def client(self):
        from client.ws_client import TableLightClient
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        return c

    async def test_send_audio_message_type(self, client):
        pcm = b"\x00\x01\x02\x03"
        await client.send_audio(pcm)
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert msg["type"] == "audio"

    async def test_send_audio_base64_data(self, client):
        pcm = b"\x00\x01\x02\x03"
        await client.send_audio(pcm)
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        decoded = base64.b64decode(msg["data"])
        assert decoded == pcm


class TestTableLightClientSendVideo:
    """send_video creates correct JSON with base64 data."""

    @pytest.fixture
    def client(self):
        from client.ws_client import TableLightClient
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        return c

    async def test_send_video_message_type(self, client):
        jpeg = b"\xff\xd8\xff\xe0fake_jpeg"
        await client.send_video(jpeg)
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert msg["type"] == "video"

    async def test_send_video_base64_data(self, client):
        jpeg = b"\xff\xd8\xff\xe0fake_jpeg"
        await client.send_video(jpeg)
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        decoded = base64.b64decode(msg["data"])
        assert decoded == jpeg


class TestTableLightClientSendText:
    """send_text sends correct JSON."""

    @pytest.fixture
    def client(self):
        from client.ws_client import TableLightClient
        c = TableLightClient("ws://localhost:8000/ws")
        c.ws = AsyncMock()
        return c

    async def test_send_text_message(self, client):
        await client.send_text("hello")
        raw = client.ws.send.call_args[0][0]
        msg = json.loads(raw)
        assert msg["type"] == "text"
        assert msg["text"] == "hello"


class TestTableLightClientReceiveLoop:
    """receive_loop dispatches messages to the correct callbacks."""

    @pytest.fixture
    def client(self):
        from client.ws_client import TableLightClient
        c = TableLightClient("ws://localhost:8000/ws")
        return c

    async def test_dispatch_audio(self, client):
        received = []
        audio_data = b"\x00\x01\x02"
        msg = json.dumps({"type": "audio", "data": base64.b64encode(audio_data).decode()})

        async def on_audio(data):
            received.append(data)

        client.on_audio(on_audio)

        # Mock ws as an async iterator that yields one message then stops
        client.ws = MockAsyncIterator([msg])
        await client.receive_loop()

        assert len(received) == 1
        assert received[0] == audio_data

    async def test_dispatch_tool_result(self, client):
        received = []
        msg = json.dumps({
            "type": "tool_result",
            "name": "project_overlay",
            "result": {"content_type": "graph", "title": "y=x^2"},
        })

        async def on_tool(name, result):
            received.append((name, result))

        client.on_tool_result(on_tool)
        client.ws = MockAsyncIterator([msg])
        await client.receive_loop()

        assert len(received) == 1
        assert received[0][0] == "project_overlay"
        assert received[0][1]["content_type"] == "graph"

    async def test_dispatch_transcript_in(self, client):
        received = []
        msg = json.dumps({"type": "transcript_in", "text": "What is 2+2?"})

        async def on_transcript(direction, text):
            received.append((direction, text))

        client.on_transcript(on_transcript)
        client.ws = MockAsyncIterator([msg])
        await client.receive_loop()

        assert received == [("in", "What is 2+2?")]

    async def test_dispatch_transcript_out(self, client):
        received = []
        msg = json.dumps({"type": "transcript_out", "text": "The answer is 4."})

        async def on_transcript(direction, text):
            received.append((direction, text))

        client.on_transcript(on_transcript)
        client.ws = MockAsyncIterator([msg])
        await client.receive_loop()

        assert received == [("out", "The answer is 4.")]

    async def test_dispatch_interrupted(self, client):
        called = []
        msg = json.dumps({"type": "interrupted"})

        async def on_interrupted():
            called.append(True)

        client.on_interrupted(on_interrupted)
        client.ws = MockAsyncIterator([msg])
        await client.receive_loop()

        assert called == [True]


class MockAsyncIterator:
    """Mock a websocket connection that yields messages then stops."""

    def __init__(self, messages: list[str]):
        self._messages = messages
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._index]
        self._index += 1
        return msg


# ---------------------------------------------------------------------------
# Camera pure function tests
# ---------------------------------------------------------------------------


def _generate_marker_image(marker_ids, size=600):
    """Generate a synthetic image with ArUco markers at known positions.

    Places markers at four corners of the image, similar to the calibration mat.
    Returns (image, expected_positions_dict).
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 100  # pixels per marker

    img = np.ones((size, size, 3), dtype=np.uint8) * 200  # light gray background

    # Positions: top-left, top-right, bottom-right, bottom-left
    margin = 50
    positions = [
        (margin, margin),
        (size - margin - marker_size, margin),
        (size - margin - marker_size, size - margin - marker_size),
        (margin, size - margin - marker_size),
    ]

    for mid, (x, y) in zip(marker_ids, positions):
        marker_img = cv2.aruco.generateImageMarker(dictionary, mid, marker_size)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        img[y:y + marker_size, x:x + marker_size] = marker_bgr

    return img


class TestDetectMarkers:
    def test_detects_all_four_markers(self):
        from client.camera import detect_markers

        img = _generate_marker_image([0, 1, 2, 3])
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        result = detect_markers(img, detector)
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_each_marker_has_four_corners(self):
        from client.camera import detect_markers

        img = _generate_marker_image([0, 1, 2, 3])
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        result = detect_markers(img, detector)
        for mid, corners in result.items():
            assert corners.shape == (4, 2), f"Marker {mid} corners should be (4,2)"

    def test_no_markers_returns_empty(self):
        from client.camera import detect_markers

        img = np.ones((400, 400, 3), dtype=np.uint8) * 128  # plain gray
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        result = detect_markers(img, detector)
        assert result == {}


class TestComputeHomography:
    def test_four_points_returns_valid_homography(self):
        from client.camera import compute_homography

        # Simulate detected markers with known inner corners
        detected = {
            0: np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=np.float32),
            1: np.array([[540, 10], [590, 10], [590, 60], [540, 60]], dtype=np.float32),
            2: np.array([[540, 540], [590, 540], [590, 590], [540, 590]], dtype=np.float32),
            3: np.array([[10, 540], [60, 540], [60, 590], [10, 590]], dtype=np.float32),
        }
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)

        H = compute_homography(detected, dst)
        assert H is not None
        assert H.shape == (3, 3)

    def test_fewer_than_four_markers_returns_none(self):
        from client.camera import compute_homography

        detected = {
            0: np.array([[10, 10], [60, 10], [60, 60], [10, 60]], dtype=np.float32),
            1: np.array([[540, 10], [590, 10], [590, 60], [540, 60]], dtype=np.float32),
        }
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)

        H = compute_homography(detected, dst)
        assert H is None

    def test_homography_maps_points_correctly(self):
        from client.camera import compute_homography

        # Identity-like mapping: inner corners at known positions
        detected = {
            0: np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float32),
            1: np.array([[700, 0], [750, 0], [750, 50], [700, 50]], dtype=np.float32),
            2: np.array([[700, 700], [750, 700], [750, 750], [700, 750]], dtype=np.float32),
            3: np.array([[0, 700], [50, 700], [50, 750], [0, 750]], dtype=np.float32),
        }
        dst = np.array([[0, 0], [768, 0], [768, 768], [0, 768]], dtype=np.float32)

        H = compute_homography(detected, dst)
        assert H is not None

        # The inner corner of marker 0 (index 2 = bottom-right) is (50,50)
        # That should map near (0,0) in dst
        pt = np.array([[[50, 50]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)
        assert abs(result[0, 0, 0] - 0) < 5
        assert abs(result[0, 0, 1] - 0) < 5


class TestRectifyFrame:
    def test_output_size(self):
        from client.camera import rectify_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        output_size = (768, 768)

        result = rectify_frame(frame, H, output_size)
        assert result.shape == (768, 768, 3)

    def test_preserves_dtype(self):
        from client.camera import rectify_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        H = np.eye(3, dtype=np.float64)
        output_size = (768, 768)

        result = rectify_frame(frame, H, output_size)
        assert result.dtype == np.uint8


class TestEncodeJpeg:
    def test_returns_valid_jpeg(self):
        from client.camera import encode_jpeg

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[30:70, 30:70] = (0, 255, 0)

        data = encode_jpeg(frame)
        assert data[:2] == b"\xff\xd8", "JPEG should start with FFD8"

    def test_returns_bytes(self):
        from client.camera import encode_jpeg

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        data = encode_jpeg(frame)
        assert isinstance(data, bytes)

    def test_quality_affects_size(self):
        from client.camera import encode_jpeg

        frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        low_q = encode_jpeg(frame, quality=10)
        high_q = encode_jpeg(frame, quality=95)
        assert len(low_q) < len(high_q), "Lower quality should produce smaller JPEG"


# ---------------------------------------------------------------------------
# Overlay manager tests
# ---------------------------------------------------------------------------


class TestOverlayManagerRender:
    def test_render_graph_returns_image(self):
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        img = mgr.render_overlay(
            content_type="graph",
            placement=[100, 100, 500, 500],
            title="y = x^2",
            data={"expression": "x**2", "x_range": [-5, 5], "y_range": [0, 25]},
        )
        assert img is not None
        assert img.shape[0] > 0 and img.shape[1] > 0
        assert img.shape[2] == 3
        # Should have some non-black content
        assert img.max() > 0

    def test_render_annotation_returns_image(self):
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        img = mgr.render_overlay(
            content_type="annotation",
            placement=[100, 100, 400, 300],
            title="Hint",
            data={"text": "Remember to carry the one!"},
        )
        assert img is not None
        assert img.shape[2] == 3
        assert img.max() > 0

    def test_render_highlight_returns_image(self):
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        img = mgr.render_overlay(
            content_type="highlight",
            placement=[200, 200, 600, 400],
            title="Focus here",
            data={"color": "#00ff00"},
        )
        assert img is not None
        assert img.shape[0] > 0


class TestOverlayManagerPlaceOnCanvas:
    def test_screen_mode_places_at_correct_position(self):
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        overlay = np.ones((100, 200, 3), dtype=np.uint8) * 255  # white block

        # placement in Gemini 0-1000 coords: [x_min, y_min, x_max, y_max]
        canvas = mgr.place_on_canvas(overlay, [0, 0, 500, 500])
        assert canvas.shape == (720, 1280, 3)
        # Top-left corner should have content (not all black)
        assert canvas[0:50, 0:50].max() > 0

    def test_screen_mode_canvas_dimensions(self):
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=800, proj_height=600, mode="screen")
        overlay = np.ones((50, 50, 3), dtype=np.uint8) * 128
        canvas = mgr.place_on_canvas(overlay, [0, 0, 100, 100])
        assert canvas.shape == (600, 800, 3)

    def test_projector_mode_with_identity_homography(self):
        from client.overlay_manager import OverlayManager

        H_proj = np.eye(3, dtype=np.float64)
        mgr = OverlayManager(H_proj=H_proj, proj_width=1280, proj_height=720, mode="projector")
        overlay = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # Place at top-left region
        canvas = mgr.place_on_canvas(overlay, [0, 0, 200, 100])
        assert canvas.shape == (720, 1280, 3)


class TestOverlayManagerClear:
    def test_clear_produces_black_canvas(self):
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        # Place something first
        overlay = np.ones((100, 200, 3), dtype=np.uint8) * 255
        mgr.place_on_canvas(overlay, [0, 0, 500, 500])
        # Now clear
        mgr.clear()
        assert mgr.canvas.max() == 0, "Canvas should be all black after clear"


class TestOverlayManagerHandleToolResult:
    def test_handle_project_overlay_calls_render(self):
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        result = {
            "content_type": "annotation",
            "placement": [100, 100, 500, 300],
            "title": "Test",
            "data": {"text": "Hello"},
        }
        # Should not raise
        mgr.handle_tool_result("project_overlay", result)
        # Canvas should have content
        assert mgr.canvas.max() > 0

    def test_handle_non_overlay_tool_is_ignored(self):
        from client.overlay_manager import OverlayManager

        mgr = OverlayManager(H_proj=None, proj_width=1280, proj_height=720, mode="screen")
        # Should not raise for unknown tool names
        mgr.handle_tool_result("some_other_tool", {"foo": "bar"})
        assert mgr.canvas.max() == 0
