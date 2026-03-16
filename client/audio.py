"""Mic capture + speaker playback for the TableLight edge client.

Uses PyAudio for audio I/O. PyAudio may not be available in all environments
(e.g. CI without audio hardware), so the import is wrapped in try/except.
The AudioCapture and AudioPlayer classes can be instantiated without PyAudio,
but start() will fail if it's not available.
"""

import queue
import threading

try:
    import pyaudio

    _PA_AVAILABLE = True
except (ImportError, OSError):
    pyaudio = None  # type: ignore[assignment]
    _PA_AVAILABLE = False

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 320  # 20ms of audio at 16kHz (lower = less latency, faster VAD)
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
    """Plays audio received from the backend.

    Uses a background thread for playback so callers never block.
    ``clear()`` flushes queued audio and signals the playback thread
    to skip writes so interruptions are near-instant.

    Thread safety: ``clear()`` is called from the async event loop thread
    while ``_playback_loop`` runs in a background thread.  A lock protects
    the stream so ``clear()`` never calls ``stop_stream()`` while
    ``write()`` is in progress.
    """

    def __init__(self, rate: int = 24000, channels: int = 1):
        # Gemini outputs 24kHz audio
        self.rate = rate
        self.channels = channels
        self._pa = None
        self._stream = None
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._thread = None
        self._interrupted = False  # signal playback thread to skip writes
        self._stream_lock = threading.Lock()

    @property
    def is_playing(self) -> bool:
        """True when there is audio queued or being written to the speaker."""
        return not self._queue.empty()

    def clear(self) -> None:
        """Immediately stop all audio playback (on interruption).

        1. Sets _interrupted so the playback thread skips writes.
        2. Drains the Python queue (pending chunks).
        3. Stops and restarts the PyAudio stream to flush the OS buffer,
           but only when the playback thread isn't mid-write.
        """
        self._interrupted = True
        # Drain the queue.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        # Flush the OS audio buffer by restarting the stream.
        # The lock ensures we don't race with _playback_loop's write().
        with self._stream_lock:
            if self._stream:
                try:
                    self._stream.stop_stream()
                    self._stream.start_stream()
                except Exception:
                    pass
        self._interrupted = False

    def start(self):
        """Open the output audio stream and start playback thread."""
        if not _PA_AVAILABLE:
            raise RuntimeError("PyAudio is not available — cannot play audio")
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=_FORMAT,
            channels=self.channels,
            rate=self.rate,
            output=True,
        )
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def _playback_loop(self):
        """Background thread: drain queue and write to audio stream."""
        while True:
            try:
                pcm_bytes = self._queue.get(timeout=0.5)
                if pcm_bytes is None:  # poison pill
                    return
                if self._interrupted:
                    continue  # skip — clear() is flushing
                with self._stream_lock:
                    if self._stream and not self._interrupted:
                        try:
                            self._stream.write(pcm_bytes)
                        except OSError:
                            # PortAudio error (e.g. stream was reset by clear()).
                            # Not fatal — next write will work after restart.
                            pass
            except queue.Empty:
                continue

    def play(self, pcm_bytes: bytes):
        """Queue audio bytes for playback (non-blocking)."""
        self._queue.put(pcm_bytes)

    def stop(self):
        """Stop playback and release resources."""
        if self._thread:
            self._queue.put(None)  # poison pill
            self._thread.join(timeout=2)
            self._thread = None
        with self._stream_lock:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
