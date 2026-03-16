"""Background music generation and playback via Lyria RealTime."""

import asyncio
import io
import logging
import os
import threading
import wave

import numpy as np

logger = logging.getLogger(__name__)

LYRIA_MODEL = "models/lyria-realtime-exp"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2
DEFAULT_VOLUME = 0.3


class MusicPlayer:
    """Manages a Lyria RealTime music session and audio playback.

    Music plays on a separate audio stream from Gemini's voice so the
    two mix naturally. Volume is adjustable (default 30%).
    """

    def __init__(self, session_store, notify_fn):
        self._session_store = session_store
        self._notify_fn = notify_fn
        self._session = None
        self._playing = False
        self._current_name = ""
        self._volume = DEFAULT_VOLUME
        self._pcm_buffer = bytearray()
        self._stop_event = threading.Event()
        self._pa_stream = None
        self._pa = None
        self._loop = None  # asyncio event loop for the music session
        self._thread = None

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def current_name(self) -> str:
        return self._current_name

    def play(self, name: str, prompt: str, bpm: int = 120,
             temperature: float = 1.0, guidance: float = 3.0) -> None:
        """Start music generation and playback in a background thread.

        Args:
            name: Name for this music track (used for saving).
            prompt: Music description (e.g. "gentle piano lullaby").
            bpm: Beats per minute (60-200).
            temperature: Randomness (0.0-3.0, default 1.0).
            guidance: Prompt adherence (0.0-6.0, default 3.0).
        """
        # Stop any current playback (check thread, not just _playing flag,
        # because _playing is set asynchronously after connection)
        if self._thread and self._thread.is_alive():
            self.stop()

        self._current_name = name
        self._pcm_buffer = bytearray()
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._music_thread,
            args=(name, prompt, bpm, temperature, guidance),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop music playback. Saves accumulated PCM to session as WAV."""
        if not self._playing and not self._thread:
            return

        self._stop_event.set()

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        # Save buffer if we have data
        saved = False
        if self._pcm_buffer and self._current_name:
            self._save_as_wav(self._current_name, self._pcm_buffer)
            saved = True

        self._playing = False
        name = self._current_name
        self._current_name = ""
        if saved:
            self._notify_fn(f"Music '{name}' stopped and saved.")
        else:
            self._notify_fn(f"Music '{name}' stopped.")

    def pause(self) -> None:
        """Pause playback (mute output, keep session alive)."""
        self._playing = False

    def resume(self) -> None:
        """Resume playback after pause."""
        self._playing = True

    def set_volume(self, volume: float) -> None:
        """Set music volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))

    def replay(self, name: str) -> None:
        """Play a previously saved music track from session storage."""
        if self._thread and self._thread.is_alive():
            self.stop()

        wav_data = self._session_store.load_music(name)
        if not wav_data:
            self._notify_fn(f"Music '{name}' not found in session.")
            return

        self._current_name = name
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._replay_thread,
            args=(name, wav_data),
            daemon=True,
        )
        self._thread.start()

    def _music_thread(self, name: str, prompt: str, bpm: int,
                      temperature: float, guidance: float) -> None:
        """Background thread: connect to Lyria, receive PCM, play audio."""
        logger.info("Music thread started for '%s' (prompt='%s')", name, prompt)
        loop = asyncio.new_event_loop()
        try:
            self._loop = loop
            loop.run_until_complete(
                self._music_session(name, prompt, bpm, temperature, guidance)
            )
        except Exception as e:
            logger.error("Music session error: %s", e)
            self._notify_fn(f"Music '{name}' playback failed: {e}")
        finally:
            self._playing = False
            self._loop = None
            loop.close()
            logger.info("Music thread ended for '%s'", name)

    async def _music_session(self, name: str, prompt: str, bpm: int,
                             temperature: float, guidance: float) -> None:
        """Async: connect to Lyria RealTime and stream audio."""
        from google import genai
        from google.genai import types

        # Lyria requires v1alpha API version — resolve API key the same
        # way genai_utils does (env vars, then llm keys).
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        logger.info("Connecting to Lyria (%s)...", LYRIA_MODEL)
        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"},
        )

        async with client.aio.live.music.connect(model=LYRIA_MODEL) as session:
            self._session = session
            logger.info("Lyria connected — configuring prompt and params")

            # Configure music
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text=prompt, weight=1.0)]
            )
            await session.set_music_generation_config(
                types.LiveMusicGenerationConfig(
                    bpm=bpm,
                    temperature=temperature,
                    guidance=guidance,
                )
            )

            # Start playback
            await session.play()
            self._playing = True
            logger.info("Lyria playing — streaming PCM for '%s'", name)
            self._notify_fn(f"Music '{name}' is now playing.")

            # Open PyAudio stream
            self._open_audio_stream()

            # Receive and play PCM chunks
            chunks_received = 0
            try:
                async for message in session.receive():
                    if self._stop_event.is_set():
                        break

                    sc = getattr(message, "server_content", None)
                    if sc is None:
                        continue
                    audio_chunks = getattr(sc, "audio_chunks", None)
                    if not audio_chunks:
                        continue

                    for chunk in audio_chunks:
                        pcm_data = chunk.data
                        if not pcm_data:
                            continue
                        chunks_received += 1
                        if chunks_received == 1:
                            logger.info(
                                "First audio chunk received (%d bytes, mime=%s)",
                                len(pcm_data),
                                getattr(chunk, "mime_type", "?"),
                            )

                        # Buffer for saving
                        self._pcm_buffer.extend(pcm_data)

                        # Apply volume and play
                        if self._playing and self._pa_stream:
                            scaled = self._apply_volume(pcm_data)
                            try:
                                self._pa_stream.write(scaled)
                            except Exception:
                                pass

            except asyncio.CancelledError:
                pass
            finally:
                await session.stop()
                self._session = None
                self._close_audio_stream()

    def _replay_thread(self, name: str, wav_data: bytes) -> None:
        """Background thread: play saved WAV data through audio stream."""
        try:
            self._open_audio_stream()
            self._playing = True
            self._notify_fn(f"Music '{name}' is now playing (replay).")

            # Parse WAV and play
            buf = io.BytesIO(wav_data)
            try:
                with wave.open(buf, "rb") as wf:
                    chunk_size = 4096
                    data = wf.readframes(chunk_size)
                    while data and not self._stop_event.is_set():
                        if self._playing and self._pa_stream:
                            scaled = self._apply_volume(data)
                            try:
                                self._pa_stream.write(scaled)
                            except Exception:
                                pass
                        data = wf.readframes(chunk_size)
            except wave.Error:
                # Raw PCM data (not WAV) — play directly
                chunk_size = 4096
                offset = 0
                while offset < len(wav_data) and not self._stop_event.is_set():
                    chunk = wav_data[offset:offset + chunk_size]
                    if self._playing and self._pa_stream:
                        scaled = self._apply_volume(chunk)
                        try:
                            self._pa_stream.write(scaled)
                        except Exception:
                            pass
                    offset += chunk_size

        except Exception as e:
            logger.error("Music replay error: %s", e)
            self._notify_fn(f"Music '{name}' replay failed: {e}")
        finally:
            self._playing = False
            self._close_audio_stream()

    def _open_audio_stream(self) -> None:
        """Open a PyAudio output stream for music at 48kHz stereo."""
        try:
            import pyaudio
            if self._pa is None:
                self._pa = pyaudio.PyAudio()
            self._pa_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=DEFAULT_CHANNELS,
                rate=DEFAULT_SAMPLE_RATE,
                output=True,
            )
        except Exception as e:
            logger.warning("Could not open music audio stream: %s", e)
            self._pa_stream = None

    def _close_audio_stream(self) -> None:
        """Close the PyAudio music stream."""
        if self._pa_stream:
            try:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
            except Exception:
                pass
            self._pa_stream = None

    def _apply_volume(self, pcm_data: bytes) -> bytes:
        """Scale PCM int16 samples by volume factor."""
        if self._volume >= 1.0:
            return pcm_data
        if self._volume <= 0.0:
            return b"\x00" * len(pcm_data)

        samples = np.frombuffer(pcm_data, dtype=np.int16)
        scaled = (samples.astype(np.float32) * self._volume).clip(-32768, 32767).astype(np.int16)
        return scaled.tobytes()

    def _save_as_wav(self, name: str, pcm_data: bytearray) -> None:
        """Save raw PCM buffer as WAV file to session store."""
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(DEFAULT_CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(DEFAULT_SAMPLE_RATE)
                wf.writeframes(bytes(pcm_data))
            self._session_store.save_music(name, buf.getvalue())
            logger.info("Saved music '%s' (%d bytes)", name, len(pcm_data))
        except Exception as e:
            logger.error("Failed to save music '%s': %s", name, e)
