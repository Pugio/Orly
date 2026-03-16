"""Tests for pipeline latency instrumentation integration."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from client.latency_tracker import LatencyTracker


# ===== Camera capture timing =====


class TestCameraLatencyInstrumentation:
    def test_get_rectified_frame_records_latency(self):
        """CameraCapture.get_rectified_frame should record camera + encode latency."""
        from client.camera import CameraCapture

        cam = CameraCapture.__new__(CameraCapture)
        cam.cap = None
        cam.H_cached = np.eye(3)
        cam.output_size = (64, 64)
        cam.rotate = 0
        cam.stats = {"frames_captured": 0, "frames_with_markers": 0,
                     "frames_using_cache": 0, "consecutive_failures": 0}
        cam._consecutive_failures = 0
        cam.detector = None
        cam.dst_points = None
        cam._last_detection_time = 0
        cam.latency_tracker = LatencyTracker()

        test_frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        cam._capture_frame = lambda: test_frame

        import client.camera as cam_mod
        original_detect = cam_mod.detect_markers
        original_rectify = cam_mod.rectify_frame
        cam_mod.detect_markers = lambda frame, det: {}
        cam_mod.rectify_frame = lambda frame, H, size: frame

        try:
            cam.get_rectified_frame()
            s = cam.latency_tracker.summary()
            assert "camera_capture" in s
            assert "encode_jpeg" in s
            assert s["camera_capture"] >= 0
            assert s["encode_jpeg"] >= 0
        finally:
            cam_mod.detect_markers = original_detect
            cam_mod.rectify_frame = original_rectify


# ===== WS client receive timing =====


class TestWSClientLatencyInstrumentation:
    @pytest.mark.asyncio
    async def test_receive_loop_records_dispatch_latency(self):
        """OrlyClient.receive_loop should record ws_dispatch latency."""
        from client.ws_client import OrlyClient

        client = OrlyClient("ws://fake", latency_tracker=LatencyTracker())

        # Mock ws to yield one transcript message then close
        transcript_msg = json.dumps({"type": "transcript_out", "text": "hello"})

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = lambda self: self
        _sent = False

        async def _anext(self):
            nonlocal _sent
            if not _sent:
                _sent = True
                return transcript_msg
            raise StopAsyncIteration

        mock_ws.__anext__ = _anext
        client.ws = mock_ws

        transcript_received = []

        async def on_transcript(direction, text):
            transcript_received.append((direction, text))

        client.on_transcript(on_transcript)
        await client.receive_loop()

        assert len(transcript_received) == 1
        s = client.latency_tracker.summary()
        assert "ws_dispatch" in s


# ===== Backend tool execution timing =====


class TestBackendToolTiming:
    def test_execute_tool_records_latency(self):
        """execute_tool should record tool execution latency when tracker provided."""
        from backend.main import execute_tool

        def slow_tool():
            time.sleep(0.005)
            return {"status": "ok"}

        registry = {"slow_tool": slow_tool}
        tracker = LatencyTracker()
        result = execute_tool("slow_tool", {}, registry, latency_tracker=tracker)
        assert result["status"] == "ok"
        s = tracker.summary()
        assert "tool_exec" in s
        assert s["tool_exec"] >= 3.0  # at least 3ms

    def test_execute_tool_works_without_tracker(self):
        """execute_tool should still work with no tracker (backward compat)."""
        from backend.main import execute_tool

        def simple_tool():
            return {"status": "ok"}

        result = execute_tool("simple_tool", {}, {"simple_tool": simple_tool})
        assert result["status"] == "ok"


# ===== Video loop timing =====


class TestVideoLoopTiming:
    @pytest.mark.asyncio
    async def test_video_loop_records_frame_latency(self):
        """video_loop should record per-frame latency when tracker is provided."""
        from client.main import video_loop

        tracker = LatencyTracker()

        # Mock camera
        mock_camera = MagicMock()
        test_frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        mock_camera.get_rectified_frame.return_value = (b"fake_jpeg", test_frame, np.eye(3))

        # Mock client
        mock_client = AsyncMock()
        mock_client.send_video = AsyncMock()

        # Run one iteration then cancel
        iteration = 0

        original_sleep = asyncio.sleep

        async def limited_sleep(duration):
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                raise asyncio.CancelledError
            await original_sleep(0)

        with patch("client.main.asyncio.sleep", limited_sleep):
            with pytest.raises(asyncio.CancelledError):
                await video_loop(
                    mock_camera, mock_client, fps=30.0,
                    latency_tracker=tracker,
                )

        s = tracker.summary()
        assert "video_frame" in s
