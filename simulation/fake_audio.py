"""Generate synthetic audio for simulation (no mic required).

All functions produce raw PCM: 16kHz, 16-bit signed mono (little-endian),
matching the format the real AudioCapture sends to the backend.
"""

from __future__ import annotations

import math
import struct

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes per sample
CHANNELS = 1
CHUNK_SAMPLES = 800  # 50ms at 16kHz — same as client/audio.py


def generate_silence(duration_s: float) -> bytes:
    """Return *duration_s* seconds of PCM silence.

    >>> len(generate_silence(1.0))
    32000
    """
    num_samples = int(SAMPLE_RATE * duration_s)
    return b"\x00" * (num_samples * SAMPLE_WIDTH)


def generate_sine(frequency_hz: float, duration_s: float, amplitude: float = 0.5) -> bytes:
    """Return a sine wave tone as PCM bytes.

    Args:
        frequency_hz: Tone frequency in Hz (e.g. 440 for A4).
        duration_s: Duration in seconds.
        amplitude: Peak amplitude 0.0-1.0 (0.5 = -6dB).

    >>> pcm = generate_sine(440.0, 0.1)
    >>> len(pcm) == int(16000 * 0.1) * 2
    True
    """
    num_samples = int(SAMPLE_RATE * duration_s)
    max_val = 32767 * amplitude
    samples = []
    for i in range(num_samples):
        t = i / SAMPLE_RATE
        value = int(max_val * math.sin(2 * math.pi * frequency_hz * t))
        samples.append(value)
    return struct.pack(f"<{num_samples}h", *samples)


def chunk_audio(pcm_bytes: bytes, chunk_samples: int = CHUNK_SAMPLES) -> list[bytes]:
    """Split PCM bytes into fixed-size chunks like the real mic produces.

    The last chunk is zero-padded to *chunk_samples* if needed.

    >>> chunks = chunk_audio(generate_silence(0.1))
    >>> all(len(c) == 800 * 2 for c in chunks)
    True
    """
    chunk_bytes = chunk_samples * SAMPLE_WIDTH
    chunks = []
    for offset in range(0, len(pcm_bytes), chunk_bytes):
        chunk = pcm_bytes[offset : offset + chunk_bytes]
        if len(chunk) < chunk_bytes:
            chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))
        chunks.append(chunk)
    return chunks


def tts_to_pcm(text: str) -> bytes:
    """Convert text to PCM using gTTS with pydub, falling back to silence.

    Requires ``gtts`` and ``pydub`` to be installed. If unavailable,
    returns 2 seconds of silence so the caller always gets usable audio.
    The result is resampled to 16kHz mono 16-bit.
    """
    try:
        import io
        import tempfile

        from gtts import gTTS  # type: ignore[import-untyped]
        from pydub import AudioSegment  # type: ignore[import-untyped]

        tts = gTTS(text=text, lang="en")
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)

        audio = AudioSegment.from_mp3(mp3_buf)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(SAMPLE_WIDTH)
        return audio.raw_data

    except Exception:
        # Fallback: 2s silence (enough to represent a short utterance gap).
        return generate_silence(2.0)
