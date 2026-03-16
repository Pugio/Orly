"""Client-side audio preprocessing: noise gate + echo suppression.

Sits between mic capture and the WebSocket send loop. Processes 16-bit
PCM chunks using only the stdlib ``struct`` module (no external deps).

Two stages:

1. **Noise gate** — suppresses chunks below a configurable RMS threshold.
   Removes background hum, fan noise, and low-level room tone so
   Gemini's server-side VAD doesn't false-trigger on silence.

2. **Echo suppression** — when the speaker is actively playing Gemini's
   audio response, attenuates the mic signal.  This is a simple
   energy-based approach (not full AEC): we know *when* audio is being
   played because AudioPlayer.is_playing tells us.  During playback we
   raise the noise gate threshold significantly, so only loud speech
   (actual user interruption) gets through.

Design choices:

- Pure Python + ``struct`` — no third-party dependencies.  Keeps the
  audio path lightweight and portable.
- Minimal state — a hold-open timer and echo holdover timestamp.  No
  adaptive filter coefficients or ring buffers that could diverge.
- Configurable thresholds so we can tune per-environment.
- The processor never *blocks* audio entirely during playback — it just
  raises the bar.  This preserves Gemini's ability to detect genuine
  user interruptions (loud speech over the speaker).
"""

from __future__ import annotations

import struct
import time


def compute_rms(pcm_bytes: bytes) -> float:
    """Compute RMS of 16-bit signed PCM bytes.

    Returns 0.0 for empty input.
    """
    n_samples = len(pcm_bytes) // 2
    if n_samples == 0:
        return 0.0
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes[:n_samples * 2])
    return (sum(s * s for s in samples) / n_samples) ** 0.5


def apply_gain(pcm_bytes: bytes, gain: float) -> bytes:
    """Apply a linear gain to 16-bit signed PCM, clamping to [-32768, 32767].

    gain=1.0 is passthrough, gain=0.0 is silence.
    Any trailing byte (odd-length input) is preserved unchanged.
    Output length always equals input length.
    """
    if gain == 1.0:
        return pcm_bytes
    n_bytes = len(pcm_bytes)
    n_samples = n_bytes // 2
    if n_samples == 0:
        return pcm_bytes
    tail = pcm_bytes[n_samples * 2:]  # trailing odd byte, if any
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes[:n_samples * 2])
    if gain == 0.0:
        return b"\x00" * (n_samples * 2) + tail
    clamped = [max(-32768, min(32767, int(s * gain))) for s in samples]
    return struct.pack(f"<{n_samples}h", *clamped) + tail


class AudioProcessor:
    """Processes mic audio chunks before sending to the backend.

    Parameters
    ----------
    noise_gate_rms : float
        RMS threshold below which audio is replaced with silence.
        Typical quiet room is 50-200; speech is 500-5000+.
        Default: 80 (conservative — suppresses room tone but allows
        soft speech and children's voices at table distance).
    gate_hold_ms : float
        Once audio exceeds the noise gate threshold, keep the gate
        open for this many ms even if RMS drops below threshold.
        This prevents inter-syllable gaps and soft consonants from
        being gated to silence, which fragments the audio stream
        and breaks Gemini's speech recognition.
        Default: 250 (covers typical syllable gaps of 100-200ms).
    echo_gate_rms : float
        Higher RMS threshold used when the speaker is playing.
        Only audio louder than this passes during playback — this
        should be set high enough that speaker echo is suppressed
        but a human speaking directly at the mic still gets through.
        Default: 800.
    echo_attenuation : float
        Gain multiplier applied to audio during speaker playback
        that exceeds echo_gate_rms.  < 1.0 attenuates, 1.0 passes.
        Default: 0.6 (modest attenuation).
    holdover_ms : float
        After the speaker stops playing, keep using the echo gate
        for this many ms to catch trailing echo/reverb.
        Default: 300.
    """

    def __init__(
        self,
        noise_gate_rms: float = 80.0,
        gate_hold_ms: float = 250.0,
        echo_gate_rms: float = 800.0,
        echo_attenuation: float = 0.6,
        holdover_ms: float = 300.0,
    ):
        self.noise_gate_rms = noise_gate_rms
        self.gate_hold_ms = gate_hold_ms
        self.echo_gate_rms = echo_gate_rms
        self.echo_attenuation = echo_attenuation
        self.holdover_ms = holdover_ms

        # State
        self._speaker_active = False
        self._speaker_stopped_at: float | None = None
        self._gate_open_until: float = 0.0  # monotonic time gate stays open

        # Stats (for diagnostics)
        self.chunks_processed = 0
        self.chunks_gated = 0  # silenced by noise gate
        self.chunks_echo_suppressed = 0  # silenced/attenuated by echo gate

    def set_speaker_active(self, active: bool) -> None:
        """Update speaker playback state.

        Call with True when AudioPlayer starts playing, False when it stops.
        The processor uses holdover_ms to handle trailing echo.
        """
        if active:
            self._speaker_active = True
            self._speaker_stopped_at = None
        elif self._speaker_active:
            self._speaker_active = False
            self._speaker_stopped_at = time.monotonic()

    def _in_echo_window(self) -> bool:
        """Check if we're in the echo suppression window.

        Returns True when the speaker is actively playing or within the
        holdover period after playback stopped. Clears stale holdover
        state as a side effect.
        """
        if self._speaker_active:
            return True
        if self._speaker_stopped_at is not None:
            elapsed_ms = (time.monotonic() - self._speaker_stopped_at) * 1000
            if elapsed_ms < self.holdover_ms:
                return True
            # Holdover expired — clear state.
            self._speaker_stopped_at = None
        return False

    def process(self, pcm_bytes: bytes) -> bytes:
        """Process a mic audio chunk, returning processed bytes.

        The returned bytes are always the same length as the input.
        Only complete 16-bit samples are considered; a trailing odd
        byte is preserved but does not affect RMS calculation.

        Returns silence (zero bytes) if the chunk is gated.
        Returns attenuated audio if echo-suppressed but above threshold.
        Returns original audio if it passes all gates.
        """
        n_bytes = len(pcm_bytes)
        if n_bytes < 2:
            # No complete samples — nothing to process.
            self.chunks_processed += 1
            return pcm_bytes

        self.chunks_processed += 1
        rms = compute_rms(pcm_bytes)
        now = time.monotonic()
        in_echo = self._in_echo_window()

        if in_echo:
            # Echo suppression mode: higher threshold.
            if rms < self.echo_gate_rms:
                self.chunks_echo_suppressed += 1
                return b"\x00" * n_bytes
            # Above echo gate — likely real speech (user interrupting).
            # Apply attenuation to reduce any residual echo component.
            self.chunks_echo_suppressed += 1
            return apply_gain(pcm_bytes, self.echo_attenuation)
        else:
            # Normal mode: noise gate with hold-open timer.
            # Once speech is detected, keep the gate open for
            # gate_hold_ms so inter-syllable gaps pass through.
            if rms >= self.noise_gate_rms:
                # Speech detected — extend the hold-open window.
                self._gate_open_until = now + self.gate_hold_ms / 1000.0
                return pcm_bytes
            if now < self._gate_open_until:
                # Below threshold but within hold-open window — pass through.
                return pcm_bytes
            # Below threshold, hold-open expired — gate to silence.
            self.chunks_gated += 1
            return b"\x00" * n_bytes

    def format_stats(self) -> str:
        """Return a one-line stats summary."""
        total = self.chunks_processed or 1
        gated_pct = self.chunks_gated / total * 100
        echo_pct = self.chunks_echo_suppressed / total * 100
        return (
            f"processed={self.chunks_processed} "
            f"gated={self.chunks_gated} ({gated_pct:.0f}%) "
            f"echo_suppressed={self.chunks_echo_suppressed} ({echo_pct:.0f}%)"
        )

    def reset_stats(self) -> None:
        """Reset per-interval stats counters."""
        self.chunks_processed = 0
        self.chunks_gated = 0
        self.chunks_echo_suppressed = 0
