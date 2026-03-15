"""Mic capture + speaker playback for the TableLight edge client.

Uses PyAudio for audio I/O. PyAudio may not be available in all environments
(e.g. CI without audio hardware), so the import is wrapped in try/except.
The AudioCapture and AudioPlayer classes can be instantiated without PyAudio,
but start() will fail if it's not available.
"""

import queue

try:
    import pyaudio

    _PA_AVAILABLE = True
except (ImportError, OSError):
    pyaudio = None  # type: ignore[assignment]
    _PA_AVAILABLE = False

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1600  # 100ms of audio at 16kHz
_FORMAT = pyaudio.paInt16 if _PA_AVAILABLE else 8  # paInt16 == 8


class AudioCapture:
    """Captures audio from the microphone in a background thread."""

    def __init__(
        self,
        rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        chunk_size: int = CHUNK_SIZE,
    ):
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._pa = None
        self._stream = None

    def start(self):
        """Start capturing audio from the default input device."""
        if not _PA_AVAILABLE:
            raise RuntimeError("PyAudio is not available — cannot capture audio")
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=_FORMAT,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def get_chunk(self) -> bytes | None:
        """Get next audio chunk, non-blocking. Returns None if no data."""
        try:
            return self._audio_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop capturing and release resources."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None


class AudioPlayer:
    """Plays audio received from the backend."""

    def __init__(self, rate: int = 24000, channels: int = 1):
        # Gemini outputs 24kHz audio
        self.rate = rate
        self.channels = channels
        self._pa = None
        self._stream = None

    def start(self):
        """Open the output audio stream."""
        if not _PA_AVAILABLE:
            raise RuntimeError("PyAudio is not available — cannot play audio")
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=_FORMAT,
            channels=self.channels,
            rate=self.rate,
            output=True,
        )

    def play(self, pcm_bytes: bytes):
        """Play audio bytes (blocking write to output stream)."""
        if self._stream:
            self._stream.write(pcm_bytes)

    def stop(self):
        """Stop playback and release resources."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
