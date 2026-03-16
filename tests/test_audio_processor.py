"""Tests for client.audio_processor — noise gate + echo suppression."""

from __future__ import annotations

import asyncio
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

    def test_preserves_trailing_byte(self):
        """Odd-length input: trailing byte is preserved."""
        # 1 sample (2 bytes) + 1 trailing byte
        pcm = struct.pack("<1h", 1000) + b"\xab"
        result = apply_gain(pcm, 0.5)
        assert len(result) == 3
        sample = struct.unpack("<1h", result[:2])[0]
        assert sample == 500
        assert result[2:] == b"\xab"

    def test_zero_gain_preserves_trailing_byte(self):
        pcm = struct.pack("<1h", 1000) + b"\xcd"
        result = apply_gain(pcm, 0.0)
        assert len(result) == 3
        assert result[:2] == b"\x00\x00"
        assert result[2:] == b"\xcd"


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

    def test_below_threshold_is_gated(self):
        """RMS below threshold is gated to silence."""
        proc = AudioProcessor(noise_gate_rms=500)
        below = _make_pcm(rms=499)
        assert _is_silence(proc.process(below))

    def test_at_threshold_passes(self):
        """RMS exactly at threshold passes (strict less-than comparison)."""
        proc = AudioProcessor(noise_gate_rms=500)
        # _make_pcm(rms=500) produces amplitude=500, RMS=500.0
        # 500.0 < 500.0 is False, so it should pass.
        at = _make_pcm(rms=500)
        result = proc.process(at)
        assert not _is_silence(result)

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
# AudioProcessor — gate hold-open timer
# ---------------------------------------------------------------------------

class TestGateHoldOpen:
    def test_quiet_after_speech_passes_within_hold(self):
        """After speech is detected, quiet audio within hold window passes."""
        proc = AudioProcessor(noise_gate_rms=150, gate_hold_ms=200)
        speech = _make_pcm(rms=1000)
        quiet = _make_pcm(rms=50)
        # Speech triggers hold-open
        result1 = proc.process(speech)
        assert not _is_silence(result1)
        # Quiet chunk immediately after — within hold window, should pass
        result2 = proc.process(quiet)
        assert not _is_silence(result2)
        assert result2 == quiet

    def test_quiet_after_hold_expires_is_gated(self):
        """After hold window expires, quiet audio is gated again."""
        proc = AudioProcessor(noise_gate_rms=150, gate_hold_ms=30)
        speech = _make_pcm(rms=1000)
        quiet = _make_pcm(rms=50)
        proc.process(speech)
        time.sleep(0.06)  # 60ms > 30ms hold
        result = proc.process(quiet)
        assert _is_silence(result)
        assert proc.chunks_gated == 1

    def test_repeated_speech_extends_hold(self):
        """Each speech chunk resets the hold-open timer."""
        proc = AudioProcessor(noise_gate_rms=150, gate_hold_ms=100)
        speech = _make_pcm(rms=1000)
        quiet = _make_pcm(rms=50)
        # First speech
        proc.process(speech)
        time.sleep(0.06)  # 60ms into 100ms window
        # Second speech — resets the hold timer
        proc.process(speech)
        time.sleep(0.06)  # 60ms into new 100ms window (120ms total)
        # Should still be within hold (only 60ms since last speech)
        result = proc.process(quiet)
        assert not _is_silence(result)

    def test_zero_hold_ms_disables_hold(self):
        """gate_hold_ms=0 means no hold — every chunk gates independently."""
        proc = AudioProcessor(noise_gate_rms=150, gate_hold_ms=0)
        speech = _make_pcm(rms=1000)
        quiet = _make_pcm(rms=50)
        proc.process(speech)
        # Quiet chunk immediately after — but hold is 0ms, should gate
        result = proc.process(quiet)
        assert _is_silence(result)

    def test_hold_does_not_affect_echo_mode(self):
        """Hold-open timer is noise-gate only; echo mode is unaffected."""
        proc = AudioProcessor(noise_gate_rms=150, gate_hold_ms=500,
                              echo_gate_rms=800)
        speech = _make_pcm(rms=1000)
        echo_level = _make_pcm(rms=500)
        # Trigger hold-open
        proc.process(speech)
        # Now activate echo mode
        proc.set_speaker_active(True)
        # Echo-level audio should be gated by echo suppression,
        # NOT passed through by the noise gate hold-open
        result = proc.process(echo_level)
        assert _is_silence(result)

    def test_default_hold_ms(self):
        proc = AudioProcessor()
        assert proc.gate_hold_ms == 250.0

    def test_hold_preserves_inter_syllable_gaps(self):
        """Simulates speech pattern: loud-quiet-loud-quiet (syllables).
        All chunks should pass through thanks to hold-open."""
        proc = AudioProcessor(noise_gate_rms=150, gate_hold_ms=200)
        loud = _make_pcm(rms=800)
        soft = _make_pcm(rms=100)  # inter-syllable gap
        results = []
        for chunk in [loud, soft, loud, soft]:
            results.append(proc.process(chunk))
        # All four chunks should pass (hold keeps gate open between syllables)
        assert all(not _is_silence(r) for r in results)
        assert proc.chunks_gated == 0


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
        assert proc._in_echo_window() is True
        # Stop again — holdover clock should restart
        proc.set_speaker_active(False)
        assert proc._in_echo_window() is True  # within holdover

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
# AudioProcessor — attenuation precision
# ---------------------------------------------------------------------------

class TestAttenuationPrecision:
    def test_attenuation_factor_applied_correctly(self):
        """Verify that echo_attenuation multiplier is applied to samples."""
        proc = AudioProcessor(echo_gate_rms=100, echo_attenuation=0.5)
        proc.set_speaker_active(True)
        # Use a simple known signal: all 2000
        samples = [2000] * 160
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        result = proc.process(pcm)
        out = struct.unpack(f"<{len(samples)}h", result)
        assert all(s == 1000 for s in out)

    def test_attenuation_1_is_passthrough(self):
        """echo_attenuation=1.0 passes audio through unchanged."""
        proc = AudioProcessor(echo_gate_rms=100, echo_attenuation=1.0)
        proc.set_speaker_active(True)
        pcm = _make_pcm(rms=2000)
        result = proc.process(pcm)
        assert result == pcm

    def test_echo_gate_boundary_below(self):
        """RMS exactly at echo_gate_rms is gated (strict less-than)."""
        proc = AudioProcessor(echo_gate_rms=500)
        proc.set_speaker_active(True)
        # _make_pcm(rms=499) produces amplitude 499, RMS = 499
        at_threshold = _make_pcm(rms=499)
        result = proc.process(at_threshold)
        assert _is_silence(result)

    def test_echo_gate_boundary_above(self):
        """RMS above echo_gate_rms passes (with attenuation)."""
        proc = AudioProcessor(echo_gate_rms=500, echo_attenuation=0.8)
        proc.set_speaker_active(True)
        above = _make_pcm(rms=501)
        result = proc.process(above)
        assert not _is_silence(result)


# ---------------------------------------------------------------------------
# AudioProcessor — edge cases
# ---------------------------------------------------------------------------

class TestProcessorEdgeCases:
    def test_empty_chunk(self):
        proc = AudioProcessor()
        result = proc.process(b"")
        assert result == b""
        assert proc.chunks_processed == 1

    def test_single_byte_unchanged(self):
        """A single byte has no complete sample — returned as-is."""
        proc = AudioProcessor()
        result = proc.process(b"\xff")
        assert result == b"\xff"
        assert proc.chunks_processed == 1

    def test_odd_byte_count_preserves_length(self):
        """Odd number of bytes — output length matches input length."""
        proc = AudioProcessor(noise_gate_rms=0)
        pcm = b"\x00\x01\x00"  # 1 complete sample + 1 trailing byte
        result = proc.process(pcm)
        assert len(result) == 3

    def test_output_length_always_matches_input(self):
        """For any input, output length must equal input length."""
        proc = AudioProcessor(noise_gate_rms=150, echo_gate_rms=800)
        for size in [0, 1, 2, 3, 100, 319, 320, 321, 640]:
            chunk = b"\x00" * size
            result = proc.process(chunk)
            assert len(result) == size, f"size={size}: got {len(result)}"

    def test_output_length_echo_mode(self):
        """Output length matches input in echo suppression mode too."""
        proc = AudioProcessor(echo_gate_rms=100, echo_attenuation=0.5)
        proc.set_speaker_active(True)
        # Loud signal that will be attenuated (not silenced)
        loud = _make_pcm(rms=2000, n_samples=160)
        result = proc.process(loud)
        assert len(result) == len(loud)

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

    def test_set_speaker_false_when_already_false_no_holdover(self):
        """Calling set_speaker_active(False) when already False doesn't
        create a spurious holdover window."""
        proc = AudioProcessor(holdover_ms=1000)
        proc.set_speaker_active(False)
        assert proc._speaker_stopped_at is None
        assert proc._in_echo_window() is False

    def test_above_threshold_always_passes(self):
        """Consecutive above-threshold chunks always pass through."""
        proc = AudioProcessor(noise_gate_rms=150)
        pcm = _make_pcm(rms=1000)
        result1 = proc.process(pcm)
        result2 = proc.process(pcm)
        assert result1 == pcm
        assert result2 == pcm


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

class TestCliFlags:
    def test_noise_gate_flag_parsed(self):
        from client.main import parse_args
        args = parse_args(["--backend", "ws://x", "--noise-gate", "42"])
        assert args.noise_gate == 42.0

    def test_echo_gate_flag_parsed(self):
        from client.main import parse_args
        args = parse_args(["--backend", "ws://x", "--echo-gate", "600"])
        assert args.echo_gate == 600.0

    def test_flags_default_to_none(self):
        from client.main import parse_args
        args = parse_args(["--backend", "ws://x"])
        assert args.noise_gate is None
        assert args.echo_gate is None


# ---------------------------------------------------------------------------
# speaker_state_loop
# ---------------------------------------------------------------------------

class TestSpeakerStateLoop:
    @pytest.mark.asyncio
    async def test_activates_on_playing(self):
        """Loop sets processor active when player starts playing."""
        from client.main import speaker_state_loop

        proc = AudioProcessor()

        class FakePlayer:
            is_playing = False

        player = FakePlayer()
        task = asyncio.create_task(speaker_state_loop(proc, player, poll_interval=0.01))

        await asyncio.sleep(0.05)
        assert proc._speaker_active is False

        player.is_playing = True
        await asyncio.sleep(0.05)
        assert proc._speaker_active is True

        player.is_playing = False
        await asyncio.sleep(0.05)
        assert proc._speaker_active is False

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_no_spurious_transitions(self):
        """Loop only calls set_speaker_active on actual transitions."""
        from client.main import speaker_state_loop

        calls: list[bool] = []

        class TrackingProcessor(AudioProcessor):
            def set_speaker_active(self, active):
                calls.append(active)
                super().set_speaker_active(active)

        proc = TrackingProcessor()

        class FakePlayer:
            is_playing = False

        player = FakePlayer()
        task = asyncio.create_task(speaker_state_loop(proc, player, poll_interval=0.01))

        # Let it poll several times with no change
        await asyncio.sleep(0.05)
        assert len(calls) == 0  # no transitions

        # Trigger one transition
        player.is_playing = True
        await asyncio.sleep(0.05)
        assert calls == [True]

        # Several polls with same state — no additional calls
        await asyncio.sleep(0.05)
        assert calls == [True]

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Integration: audio_send_loop diagnostics see raw audio
# ---------------------------------------------------------------------------

class TestDiagnosticsSeesRawAudio:
    @pytest.mark.asyncio
    async def test_diagnostics_receives_raw_not_processed(self):
        """on_mic_chunk should see the original mic audio, not the
        processed (possibly silent) version."""
        import queue as queue_mod
        from client.audio import AudioCapture
        from client.main import audio_send_loop, AudioDiagnostics

        capture = AudioCapture.__new__(AudioCapture)
        capture._audio_queue = queue_mod.Queue()
        capture.chunk_size = 320

        # A chunk that will be noise-gated to silence
        quiet = _make_pcm(rms=50, n_samples=320)
        capture._audio_queue.put(quiet)

        sent_chunks: list[bytes] = []

        class FakeClient:
            async def send_audio(self, data):
                sent_chunks.append(data)

        # Custom diagnostics that records what chunks it receives
        received_by_diag: list[bytes] = []
        diag = AudioDiagnostics()
        original_on_mic = diag.on_mic_chunk

        def tracking_on_mic(chunk):
            received_by_diag.append(chunk)
            original_on_mic(chunk)

        diag.on_mic_chunk = tracking_on_mic

        proc = AudioProcessor(noise_gate_rms=150)

        task = asyncio.create_task(
            audio_send_loop(capture, FakeClient(), diag=diag, processor=proc)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Diagnostics should have received the raw chunk (not silence)
        assert len(received_by_diag) == 1
        assert received_by_diag[0] == quiet  # raw, not processed

        # But the backend should have received silence (processed)
        assert len(sent_chunks) == 1
        assert _is_silence(sent_chunks[0])
