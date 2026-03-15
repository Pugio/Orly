"""Tests for client/audio.py and client/main.py orchestrator logic."""

import asyncio

import pytest


# ---------------------------------------------------------------------------
# audio.py — constants and class interface
# ---------------------------------------------------------------------------


class TestAudioConstants:
    def test_sample_rate(self):
        from client.audio import SAMPLE_RATE
        assert SAMPLE_RATE == 16000

    def test_channels(self):
        from client.audio import CHANNELS
        assert CHANNELS == 1

    def test_chunk_size(self):
        from client.audio import CHUNK_SIZE
        assert CHUNK_SIZE == 320

    def test_playback_rate(self):
        """Gemini outputs 24kHz audio; AudioPlayer should default to 24000."""
        from client.audio import AudioPlayer
        player = AudioPlayer()
        assert player.rate == 24000


class TestAudioCaptureInterface:
    def test_has_start_method(self):
        from client.audio import AudioCapture
        cap = AudioCapture()
        assert callable(cap.start)

    def test_has_stop_method(self):
        from client.audio import AudioCapture
        cap = AudioCapture()
        assert callable(cap.stop)

    def test_has_get_chunk_method(self):
        from client.audio import AudioCapture
        cap = AudioCapture()
        assert callable(cap.get_chunk)

    def test_get_chunk_returns_none_when_empty(self):
        from client.audio import AudioCapture
        cap = AudioCapture()
        assert cap.get_chunk() is None

    def test_custom_parameters(self):
        from client.audio import AudioCapture
        cap = AudioCapture(rate=44100, channels=2, chunk_size=4096)
        assert cap.rate == 44100
        assert cap.channels == 2
        assert cap.chunk_size == 4096


class TestAudioPlayerInterface:
    def test_has_start_method(self):
        from client.audio import AudioPlayer
        player = AudioPlayer()
        assert callable(player.start)

    def test_has_stop_method(self):
        from client.audio import AudioPlayer
        player = AudioPlayer()
        assert callable(player.stop)

    def test_has_play_method(self):
        from client.audio import AudioPlayer
        player = AudioPlayer()
        assert callable(player.play)

    def test_custom_rate(self):
        from client.audio import AudioPlayer
        player = AudioPlayer(rate=48000, channels=2)
        assert player.rate == 48000
        assert player.channels == 2


# ---------------------------------------------------------------------------
# main.py — video_loop logic
# ---------------------------------------------------------------------------


class TestVideoLoop:
    async def test_video_loop_sends_frames(self):
        """video_loop should capture frames and send them via client."""
        from client.main import video_loop

        # Mock camera that returns known JPEG bytes
        class MockCamera:
            def __init__(self):
                self.call_count = 0

            def get_rectified_frame(self):
                self.call_count += 1
                return b"fake-jpeg-data", None

        # Mock client that records sent messages
        class MockClient:
            def __init__(self):
                self.sent_frames = []

            async def send_video(self, jpeg_bytes):
                self.sent_frames.append(jpeg_bytes)

        camera = MockCamera()
        client = MockClient()

        # Run video_loop at 10 FPS for ~0.25s — expect 2-3 frames
        task = asyncio.create_task(video_loop(camera, client, fps=10.0))
        await asyncio.sleep(0.35)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(client.sent_frames) >= 2
        assert len(client.sent_frames) <= 5
        assert client.sent_frames[0] == b"fake-jpeg-data"

    async def test_video_loop_skips_none_frames(self):
        """video_loop should not send when camera returns None."""
        from client.main import video_loop

        class MockCamera:
            def get_rectified_frame(self):
                return None, None

        class MockClient:
            def __init__(self):
                self.sent_frames = []

            async def send_video(self, jpeg_bytes):
                self.sent_frames.append(jpeg_bytes)

        camera = MockCamera()
        client = MockClient()

        task = asyncio.create_task(video_loop(camera, client, fps=10.0))
        await asyncio.sleep(0.25)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(client.sent_frames) == 0


class TestAudioSendLoop:
    async def test_audio_send_loop_sends_chunks(self):
        """audio_send_loop should forward audio chunks from capture to client."""
        from client.main import audio_send_loop

        class MockAudioCapture:
            def __init__(self):
                self._chunks = [b"chunk1", b"chunk2", b"chunk3"]
                self._idx = 0

            def get_chunk(self):
                if self._idx < len(self._chunks):
                    chunk = self._chunks[self._idx]
                    self._idx += 1
                    return chunk
                return None

        class MockClient:
            def __init__(self):
                self.sent_audio = []

            async def send_audio(self, pcm_bytes):
                self.sent_audio.append(pcm_bytes)

        capture = MockAudioCapture()
        client = MockClient()

        task = asyncio.create_task(audio_send_loop(capture, client))
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert client.sent_audio == [b"chunk1", b"chunk2", b"chunk3"]
