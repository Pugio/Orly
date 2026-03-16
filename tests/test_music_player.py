"""Tests for music player and related backend tools."""

import io
import struct
import wave

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from backend.tools import play_music, stop_music, pause_music, resume_music, replay_music
from client.music_player import MusicPlayer, DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS
from client.session_store import SessionStore


# ---------------------------------------------------------------------------
# Backend tool tests
# ---------------------------------------------------------------------------


class TestPlayMusicTool:
    def test_returns_starting(self):
        result = play_music("track", "gentle piano")
        assert result["status"] == "starting"
        assert result["name"] == "track"
        assert result["prompt"] == "gentle piano"

    def test_default_params(self):
        result = play_music("track", "piano")
        assert result["bpm"] == 120
        assert result["temperature"] == 1.0
        assert result["guidance"] == 3.0


class TestStopMusicTool:
    def test_returns_stopping(self):
        result = stop_music("track")
        assert result["status"] == "stopping"


class TestPauseMusicTool:
    def test_returns_pausing(self):
        assert pause_music()["status"] == "pausing"


class TestResumeMusicTool:
    def test_returns_resuming(self):
        assert resume_music()["status"] == "resuming"


class TestReplayMusicTool:
    def test_returns_replaying(self):
        result = replay_music("track")
        assert result["status"] == "replaying"
        assert result["name"] == "track"


# ---------------------------------------------------------------------------
# MusicPlayer unit tests (no actual audio or Lyria)
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return SessionStore(session_dir=str(tmp_path / "session"))


@pytest.fixture
def notifications():
    return []


@pytest.fixture
def player(store, notifications):
    return MusicPlayer(session_store=store, notify_fn=notifications.append)


class TestMusicPlayerInit:
    def test_defaults(self, player):
        assert player.is_playing is False
        assert player.current_name == ""


class TestMusicPlayerVolume:
    def test_set_volume_clamped(self, player):
        player.set_volume(1.5)
        assert player._volume == 1.0
        player.set_volume(-0.5)
        assert player._volume == 0.0
        player.set_volume(0.7)
        assert player._volume == 0.7


class TestApplyVolume:
    def test_full_volume(self, player):
        """At volume 1.0, data is unchanged."""
        player._volume = 1.0
        data = struct.pack("<4h", 100, -200, 300, -400)
        result = player._apply_volume(data)
        assert result == data

    def test_zero_volume(self, player):
        """At volume 0.0, all samples are zero."""
        player._volume = 0.0
        data = struct.pack("<4h", 100, -200, 300, -400)
        result = player._apply_volume(data)
        assert result == b"\x00" * len(data)

    def test_half_volume(self, player):
        """At volume 0.5, samples are halved."""
        player._volume = 0.5
        data = struct.pack("<2h", 1000, -1000)
        result = player._apply_volume(data)
        samples = np.frombuffer(result, dtype=np.int16)
        assert samples[0] == 500
        assert samples[1] == -500


class TestSaveAsWav:
    def test_saves_valid_wav(self, player, store):
        """Saves a valid WAV file to session store."""
        # Create some PCM data
        pcm = struct.pack("<4h", 0, 100, 200, 300) * 100
        player._save_as_wav("test-track", bytearray(pcm))

        # Load and verify it's valid WAV
        wav_data = store.load_music("test-track")
        assert wav_data is not None

        buf = io.BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == DEFAULT_CHANNELS
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == DEFAULT_SAMPLE_RATE


class TestMusicPlayerPauseResume:
    def test_pause_sets_not_playing(self, player):
        player._playing = True
        player.pause()
        assert player._playing is False

    def test_resume_sets_playing(self, player):
        player._playing = False
        player.resume()
        assert player._playing is True


class TestMusicPlayerReplay:
    def test_replay_nonexistent_notifies(self, player, notifications):
        """Replaying nonexistent track sends notification."""
        player.replay("nonexistent")
        # Give thread a moment
        import time
        time.sleep(0.1)
        assert any("not found" in n for n in notifications)

    def test_replay_existing_starts_thread(self, player, store, notifications):
        """Replaying existing track starts playback."""
        # Create a minimal WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(DEFAULT_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(DEFAULT_SAMPLE_RATE)
            wf.writeframes(b"\x00" * 100)
        store.save_music("saved-track", buf.getvalue())

        # Mock PyAudio so we don't need actual audio hardware
        with patch("client.music_player.MusicPlayer._open_audio_stream"):
            with patch("client.music_player.MusicPlayer._close_audio_stream"):
                player.replay("saved-track")
                assert player._current_name == "saved-track"


class TestMusicPlayerStop:
    def test_stop_when_not_playing(self, player, notifications):
        """Stopping when not playing is a no-op."""
        player.stop()
        # Should not crash, no notification about stopping
