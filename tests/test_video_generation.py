"""Tests for video generation, playback, and renderer."""

import numpy as np
from unittest.mock import MagicMock, patch

from backend.tools_roadmap import generate_video, play_video, stop_video
from client.renderer.video import render_video_loading
from client.video_player import VideoPlayer


# ---------------------------------------------------------------------------
# Backend tool tests
# ---------------------------------------------------------------------------


class TestGenerateVideoTool:
    def test_valid_params(self):
        result = generate_video("clip", "a cat", [0, 0, 1000, 1000], duration=5)
        assert result["status"] == "generating"
        assert result["name"] == "clip"

    def test_invalid_duration(self):
        result = generate_video("clip", "a cat", [0, 0, 1000, 1000], duration=3)
        assert result["status"] == "error"
        assert "duration" in result["message"]

    def test_valid_durations(self):
        for d in (4, 5, 6, 8):
            result = generate_video("clip", "test", [0, 0, 1000, 1000], duration=d)
            assert result["status"] == "generating"


class TestPlayVideoTool:
    def test_returns_playing(self):
        result = play_video("clip", [0, 0, 500, 500], loop=True)
        assert result["status"] == "playing"
        assert result["loop"] is True


class TestStopVideoTool:
    def test_returns_stopping(self):
        result = stop_video("clip")
        assert result["status"] == "stopping"


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------


class TestVideoRenderer:
    def test_loading_placeholder_shape(self):
        img = render_video_loading({"prompt": "test"}, 400, 300)
        assert img.shape == (300, 400, 3)
        assert img.dtype == np.uint8

    def test_loading_placeholder_not_all_black(self):
        img = render_video_loading({"prompt": "test"}, 400, 300)
        assert np.any(img > 0)


# ---------------------------------------------------------------------------
# VideoPlayer tests
# ---------------------------------------------------------------------------


class TestVideoPlayer:
    def test_stop_nonexistent(self):
        om = MagicMock()
        player = VideoPlayer(om)
        assert player.stop("nope") is False

    def test_stop_all_empty(self):
        om = MagicMock()
        player = VideoPlayer(om)
        player.stop_all()  # should not raise


# ---------------------------------------------------------------------------
# VideoGenerator tests
# ---------------------------------------------------------------------------


class TestVideoGenerator:
    def test_generate_shows_loading(self):
        """generate_async shows loading placeholder immediately."""
        om = MagicMock()
        om._placement_pixel_size.return_value = (400, 300)
        player = MagicMock()
        store = MagicMock()
        notifications = []

        from client.video_generator import VideoGenerator
        gen = VideoGenerator(om, player, store, notifications.append)

        # Mock the thread so it doesn't actually run
        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            gen.generate_async("test", "a cat", [0, 0, 1000, 1000], duration=5)

        # Loading placeholder shown
        om._show_overlay.assert_called_once()

    def test_generate_thread_success(self):
        """Full generation thread: generate, poll, save, play, notify."""
        om = MagicMock()
        om._placement_pixel_size.return_value = (400, 300)
        om.overlay_state = None

        player = MagicMock()
        store = MagicMock()
        store.save_video.return_value = "/tmp/test.mp4"
        notifications = []

        from client.video_generator import VideoGenerator
        gen = VideoGenerator(om, player, store, notifications.append)

        # Mock the genai client
        mock_video_file = MagicMock()
        mock_video_file.read.return_value = b"fake-mp4-data"

        mock_generated = MagicMock()
        mock_generated.video = mock_video_file

        mock_response = MagicMock()
        mock_response.generated_videos = [mock_generated]

        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.response = mock_response

        mock_client = MagicMock()
        mock_client.models.generate_videos.return_value = mock_operation

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            gen._generate_video_thread("test", "a cat", [0, 0, 1000, 1000], 5, "16:9")

        # Video saved
        store.save_video.assert_called_once_with("test", b"fake-mp4-data")

        # Playback started
        player.play.assert_called_once()

        # Notification sent
        assert len(notifications) == 1
        assert "ready and playing" in notifications[0]

    def test_generate_thread_failure(self):
        """API error sends failure notification."""
        om = MagicMock()
        om._placement_pixel_size.return_value = (400, 300)
        player = MagicMock()
        store = MagicMock()
        notifications = []

        from client.video_generator import VideoGenerator
        gen = VideoGenerator(om, player, store, notifications.append)

        mock_client = MagicMock()
        mock_client.models.generate_videos.side_effect = RuntimeError("API down")

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            gen._generate_video_thread("fail", "test", [0, 0, 1000, 1000], 5, "16:9")

        assert len(notifications) == 1
        assert "failed" in notifications[0]

    def test_generate_thread_empty_response(self):
        """Empty response sends failure notification."""
        om = MagicMock()
        om._placement_pixel_size.return_value = (400, 300)
        player = MagicMock()
        store = MagicMock()
        notifications = []

        from client.video_generator import VideoGenerator
        gen = VideoGenerator(om, player, store, notifications.append)

        mock_operation = MagicMock()
        mock_operation.done = True
        mock_operation.response = MagicMock()
        mock_operation.response.generated_videos = []

        mock_client = MagicMock()
        mock_client.models.generate_videos.return_value = mock_operation

        with patch("client.genai_utils.get_genai_client", return_value=mock_client):
            gen._generate_video_thread("empty", "test", [0, 0, 1000, 1000], 5, "16:9")

        assert len(notifications) == 1
        assert "failed" in notifications[0]
