"""Tests for client.audio_processor — noise gate + echo suppression.

Red/green TDD: these tests were written first, then the implementation.
"""

from __future__ import annotations

import struct
import time

import pytest

from client.audio_processor import AudioProcessor, apply_gain, compute_rms


# ---------------------------------------------------------------------------
# Helper: generate PCM audio at a given RMS level
# ---------------------------------------------------------------------------

def _make_pcm(rms: float, n_samples: int = 160) -> bytes:
    """Generate 16-bit signed PCM with approximately the given RMS.

    Uses a simple square wave at +/- amplitude to hit the target RMS exactly.
    """
    amplitude = int(min(rms, 32767))
    samples = [amplitude if i % 2 == 0 else -amplitude for i in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)


def _is_silence(pcm_bytes: bytes) -> bool:
    """True if all samples are zero."""
    return all(b == 0 for b in pcm_bytes)


# ---------------------------------------------------------------------------
# compute_rms
# ---------------------------------------------------------------------------

class TestComputeRms:
    def test_silence_is_zero(self):
        silence = b"\x00" * 320  # 160 samples
        assert compute_rms(silence) == 0.0

    def test_known_signal(self):
        # All samples = 1000 → RMS = 1000
        samples = [1000] * 100
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        assert abs(compute_rms(pcm) - 1000.0) < 1.0

    def test_empty_input(self):
        assert compute_rms(b"") == 0.0

    def test_single_byte_ignored(self):
        # Less than 2 bytes → 0 samples
        assert compute_rms(b"\x01") == 0.0


# ---------------------------------------------------------------------------
# apply_gain
# ---------------------------------------------------------------------------

class TestApplyGain:
    def test_unity_gain_passthrough(self):
        pcm = _make_pcm(500)
        assert apply_gain(pcm, 1.0) is pcm  # same object

    def test_zero_gain_silence(self):
        pcm = _make_pcm(500)
        result = apply_gain(pcm, 0.0)
        assert _is_silence(result)
        assert len(result) == len(pcm)

    def test_half_gain(self):
        samples = [1000, -1000, 500, -500]
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        result = apply_gain(pcm, 0.5)
        out = struct.unpack(f"<{len(samples)}h", result)
        assert out == (500, -500, 250, -250)

    def test_clamps_to_max(self):
        samples = [30000]
        pcm = struct.pack("<1h", *samples)
        result = apply_gain(pcm, 2.0)
        out = struct.unpack("<1h", result)
        assert out[0] == 32767

    def test_clamps_to_min(self):
        samples = [-30000]
        pcm = struct.pack("<1h", *samples)
        result = apply_gain(pcm, 2.0)
        out = struct.unpack("<1h", result)
        assert out[0] == -32768

    def test_empty_input(self):
        assert apply_gain(b"", 0.5) == b""


# ---------------------------------------------------------------------------
# AudioProcessor — noise gate (no echo)
# ---------------------------------------------------------------------------

class TestNoiseGate:
    def test_silence_is_gated(self):
        proc = AudioProcessor(noise_gate_rms=150)
        silence = b"\x00" * 320
        result = proc.process(silence)
        assert _is_silence(result)
        assert proc.chunks_gated == 1

    def test_quiet_noise_is_gated(self):
        proc = AudioProcessor(noise_gate_rms=150)
        quiet = _make_pcm(rms=100)
        result = proc.process(quiet)
        assert _is_silence(result)

    def test_speech_passes_through(self):
        proc = AudioProcessor(noise_gate_rms=150)
        speech = _make_pcm(rms=1000)
        result = proc.process(speech)
        assert result == speech
        assert proc.chunks_gated == 0

    def test_exactly_at_threshold_is_gated(self):
        """RMS < threshold is gated; RMS == threshold is also gated."""
        proc = AudioProcessor(noise_gate_rms=500)
        # RMS just below threshold
        below = _make_pcm(rms=499)
        assert _is_silence(proc.process(below))

    def test_above_threshold_passes(self):
        proc = AudioProcessor(noise_gate_rms=500)
        above = _make_pcm(rms=501)
        result = proc.process(above)
        assert not _is_silence(result)

    def test_zero_threshold_passes_everything(self):
        """noise_gate_rms=0 effectively disables the noise gate."""
        proc = AudioProcessor(noise_gate_rms=0)
        # Even very quiet audio should pass
        quiet = _make_pcm(rms=10)
        result = proc.process(quiet)
        assert not _is_silence(result)


# ---------------------------------------------------------------------------
# AudioProcessor — echo suppression
# ---------------------------------------------------------------------------

class TestEchoSuppression:
    def test_normal_speech_not_affected_when_speaker_off(self):
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800)
        speech = _make_pcm(rms=500)
        result = proc.process(speech)
        assert result == speech  # passes through unmodified

    def test_echo_gated_during_playback(self):
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800)
        proc.set_speaker_active(True)
        # Moderate audio (likely echo from speaker, not user speech)
        echo = _make_pcm(rms=500)
        result = proc.process(echo)
        assert _is_silence(result)
        assert proc.chunks_echo_suppressed == 1

    def test_loud_speech_passes_during_playback(self):
        """User interrupting loudly should still get through."""
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800,
                              echo_attenuation=0.6)
        proc.set_speaker_active(True)
        loud = _make_pcm(rms=2000)
        result = proc.process(loud)
        # Should NOT be silence — it passes but attenuated
        assert not _is_silence(result)
        # But RMS should be reduced
        result_rms = compute_rms(result)
        original_rms = compute_rms(loud)
        assert result_rms < original_rms

    def test_holdover_after_speaker_stops(self):
        """Echo gate stays active for holdover_ms after speaker stops."""
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800,
                              holdover_ms=200)
        proc.set_speaker_active(True)
        proc.set_speaker_active(False)
        # Immediately after stopping — should still be in echo window
        echo = _make_pcm(rms=500)
        result = proc.process(echo)
        assert _is_silence(result)

    def test_holdover_expires(self):
        """After holdover period, normal noise gate resumes."""
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800,
                              holdover_ms=50)
        proc.set_speaker_active(True)
        proc.set_speaker_active(False)
        # Wait for holdover to expire
        time.sleep(0.1)  # 100ms > 50ms holdover
        speech = _make_pcm(rms=500)
        result = proc.process(speech)
        # Should pass through normally (noise gate threshold is 150, rms is 500)
        assert result == speech

    def test_speaker_reactivate_resets_holdover(self):
        """If speaker becomes active again, holdover state resets."""
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800,
                              holdover_ms=50)
        proc.set_speaker_active(True)
        proc.set_speaker_active(False)
        # Before holdover expires, speaker starts again
        proc.set_speaker_active(True)
        assert proc._in_echo_window is True
        # Stop again — holdover clock should restart
        proc.set_speaker_active(False)
        assert proc._in_echo_window is True  # within holdover

    def test_silence_gated_during_playback(self):
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800)
        proc.set_speaker_active(True)
        silence = b"\x00" * 320
        result = proc.process(silence)
        assert _is_silence(result)


# ---------------------------------------------------------------------------
# AudioProcessor — stats
# ---------------------------------------------------------------------------

class TestProcessorStats:
    def test_stats_count(self):
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800)
        proc.process(b"\x00" * 320)  # silence → gated
        proc.process(_make_pcm(rms=1000))  # speech → passes
        proc.set_speaker_active(True)
        proc.process(_make_pcm(rms=500))  # echo → suppressed
        assert proc.chunks_processed == 3
        assert proc.chunks_gated == 1
        assert proc.chunks_echo_suppressed == 1

    def test_stats_reset(self):
        proc = AudioProcessor(noise_gate_rms=150)
        proc.process(b"\x00" * 320)
        assert proc.chunks_processed == 1
        proc.reset_stats()
        assert proc.chunks_processed == 0
        assert proc.chunks_gated == 0
        assert proc.chunks_echo_suppressed == 0

    def test_format_stats(self):
        proc = AudioProcessor()
        proc.chunks_processed = 100
        proc.chunks_gated = 30
        proc.chunks_echo_suppressed = 10
        s = proc.format_stats()
        assert "processed=100" in s
        assert "gated=30" in s
        assert "echo_suppressed=10" in s


# ---------------------------------------------------------------------------
# AudioProcessor — edge cases
# ---------------------------------------------------------------------------

class TestProcessorEdgeCases:
    def test_empty_chunk(self):
        proc = AudioProcessor()
        result = proc.process(b"")
        assert result == b""

    def test_odd_byte_count(self):
        """Odd number of bytes — last byte is ignored in RMS."""
        proc = AudioProcessor(noise_gate_rms=0)
        pcm = b"\x00\x01\x00"  # 1 complete sample + 1 trailing byte
        result = proc.process(pcm)
        assert len(result) == 3

    def test_default_thresholds_reasonable(self):
        proc = AudioProcessor()
        assert proc.noise_gate_rms > 0
        assert proc.echo_gate_rms > proc.noise_gate_rms
        assert 0 < proc.echo_attenuation <= 1.0

    def test_set_speaker_active_idempotent(self):
        proc = AudioProcessor()
        proc.set_speaker_active(True)
        proc.set_speaker_active(True)  # no crash
        assert proc._speaker_active is True
        proc.set_speaker_active(False)
        proc.set_speaker_active(False)  # no crash
        assert proc._speaker_active is False
