"""Tests for the simulation harness — pure functions only, no API calls."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from simulation.fake_audio import (
    CHUNK_SAMPLES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    chunk_audio,
    generate_silence,
    generate_sine,
)
from simulation.fake_camera import (
    encode_frame_jpeg,
    generate_test_frame,
    generate_test_jpeg,
    load_image_as_jpeg,
)
from simulation.scenarios import (
    ALL_SCENARIOS,
    INTERRUPTION,
    SILENCE_ONLY,
    SIMPLE_QUESTION,
    WITH_TOOL_CALL,
    Scenario,
    ScenarioStep,
)
from simulation.sim_client import ScenarioResult, format_latency


# ===== fake_audio tests =====


class TestGenerateSilence:
    def test_correct_length_1s(self):
        pcm = generate_silence(1.0)
        expected_bytes = SAMPLE_RATE * SAMPLE_WIDTH  # 16000 * 2 = 32000
        assert len(pcm) == expected_bytes

    def test_correct_length_half_second(self):
        pcm = generate_silence(0.5)
        assert len(pcm) == 16000

    def test_all_zeros(self):
        pcm = generate_silence(0.1)
        assert all(b == 0 for b in pcm)

    def test_zero_duration(self):
        pcm = generate_silence(0.0)
        assert len(pcm) == 0


class TestGenerateSine:
    def test_correct_length(self):
        pcm = generate_sine(440.0, 1.0)
        expected = SAMPLE_RATE * SAMPLE_WIDTH
        assert len(pcm) == expected

    def test_not_silence(self):
        pcm = generate_sine(440.0, 0.1)
        assert not all(b == 0 for b in pcm)

    def test_amplitude_bounds(self):
        """Samples should not exceed the specified amplitude."""
        pcm = generate_sine(440.0, 0.1, amplitude=0.5)
        num_samples = len(pcm) // SAMPLE_WIDTH
        samples = struct.unpack(f"<{num_samples}h", pcm)
        max_expected = int(32767 * 0.5) + 1  # +1 for rounding
        assert all(abs(s) <= max_expected for s in samples)

    def test_frequency_period(self):
        """Check that the sine wave has approximately the right period."""
        freq = 1000.0  # 1kHz = 16 samples per period at 16kHz
        pcm = generate_sine(freq, 0.01)  # 10ms = 160 samples
        num_samples = len(pcm) // SAMPLE_WIDTH
        samples = struct.unpack(f"<{num_samples}h", pcm)

        # Count zero crossings (should be ~2 per period).
        crossings = 0
        for i in range(1, len(samples)):
            if (samples[i - 1] >= 0) != (samples[i] >= 0):
                crossings += 1

        # 10ms at 1kHz = 10 full cycles = ~20 zero crossings
        assert 18 <= crossings <= 22

    def test_short_duration(self):
        pcm = generate_sine(440.0, 0.001)
        assert len(pcm) == int(SAMPLE_RATE * 0.001) * SAMPLE_WIDTH


class TestChunkAudio:
    def test_exact_chunks(self):
        """1 second = exactly 20 chunks of 800 samples."""
        pcm = generate_silence(1.0)
        chunks = chunk_audio(pcm)
        assert len(chunks) == 20
        assert all(len(c) == CHUNK_SAMPLES * SAMPLE_WIDTH for c in chunks)

    def test_last_chunk_padded(self):
        """Partial last chunk is zero-padded."""
        # 900 samples = 1 full chunk + 100 samples leftover
        pcm = generate_silence(900 / SAMPLE_RATE)
        chunks = chunk_audio(pcm)
        assert len(chunks) == 2
        assert len(chunks[1]) == CHUNK_SAMPLES * SAMPLE_WIDTH

    def test_empty_input(self):
        chunks = chunk_audio(b"")
        assert chunks == []

    def test_single_sample(self):
        pcm = b"\x01\x00"  # one 16-bit sample
        chunks = chunk_audio(pcm)
        assert len(chunks) == 1
        assert len(chunks[0]) == CHUNK_SAMPLES * SAMPLE_WIDTH
        assert chunks[0][:2] == b"\x01\x00"
        assert chunks[0][2:] == b"\x00" * (CHUNK_SAMPLES * SAMPLE_WIDTH - 2)

    def test_custom_chunk_size(self):
        pcm = generate_silence(0.1)  # 1600 samples
        chunks = chunk_audio(pcm, chunk_samples=400)
        assert len(chunks) == 4
        assert all(len(c) == 400 * SAMPLE_WIDTH for c in chunks)

    def test_roundtrip_content(self):
        """Chunking should preserve the original audio content."""
        pcm = generate_sine(440.0, 0.05)  # exactly 800 samples = 1 chunk
        chunks = chunk_audio(pcm)
        assert len(chunks) == 1
        assert chunks[0] == pcm


# ===== fake_camera tests =====


class TestGenerateTestFrame:
    def test_default_shape(self):
        frame = generate_test_frame()
        assert frame.shape == (768, 768, 3)
        assert frame.dtype == np.uint8

    def test_custom_size(self):
        frame = generate_test_frame(width=640, height=480)
        assert frame.shape == (480, 640, 3)

    def test_not_uniform(self):
        """Frame should have variation (text, paper, desk)."""
        frame = generate_test_frame()
        assert frame.min() != frame.max()

    def test_custom_text(self):
        frame = generate_test_frame(text_lines=["Custom problem: x + 1 = 3"])
        assert frame.shape == (768, 768, 3)

    def test_has_white_region(self):
        """Paper area should contain white pixels."""
        frame = generate_test_frame()
        white_count = np.sum(np.all(frame == 255, axis=2))
        assert white_count > 0


class TestEncodeFrameJpeg:
    def test_jpeg_header(self):
        frame = generate_test_frame()
        jpeg = encode_frame_jpeg(frame)
        assert jpeg[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_jpeg_nonzero_size(self):
        frame = generate_test_frame()
        jpeg = encode_frame_jpeg(frame)
        assert len(jpeg) > 100

    def test_different_quality(self):
        frame = generate_test_frame()
        low = encode_frame_jpeg(frame, quality=10)
        high = encode_frame_jpeg(frame, quality=95)
        assert len(low) < len(high)


class TestGenerateTestJpeg:
    def test_returns_bytes(self):
        jpeg = generate_test_jpeg()
        assert isinstance(jpeg, bytes)
        assert jpeg[:2] == b"\xff\xd8"

    def test_custom_dimensions(self):
        jpeg = generate_test_jpeg(width=320, height=240)
        assert len(jpeg) > 50


class TestLoadImageAsJpeg:
    def test_missing_file(self):
        with pytest.raises(RuntimeError):
            load_image_as_jpeg("/nonexistent/path.png")


# ===== scenarios tests =====


class TestScenarios:
    def test_all_scenarios_present(self):
        assert "silence_only" in ALL_SCENARIOS
        assert "simple_question" in ALL_SCENARIOS
        assert "with_tool_call" in ALL_SCENARIOS
        assert "interruption" in ALL_SCENARIOS

    def test_scenario_has_steps(self):
        for name, scenario in ALL_SCENARIOS.items():
            assert len(scenario.steps) > 0, f"{name} has no steps"

    def test_silence_only_config(self):
        assert not SILENCE_ONLY.expect_transcript_in
        assert not SILENCE_ONLY.expect_transcript_out

    def test_simple_question_config(self):
        assert SIMPLE_QUESTION.expect_transcript_out

    def test_with_tool_call_config(self):
        assert WITH_TOOL_CALL.expect_tool_call
        assert WITH_TOOL_CALL.expect_transcript_out

    def test_interruption_has_two_sends(self):
        text_steps = [s for s in INTERRUPTION.steps if s.action == "send_text"]
        assert len(text_steps) == 2

    def test_scenario_step_defaults(self):
        step = ScenarioStep(action="wait")
        assert step.duration_s == 0.0
        assert step.text == ""
        assert step.image_path == ""
        assert step.video_text_lines == []

    def test_custom_scenario(self):
        s = Scenario(
            name="custom",
            description="A custom test",
            steps=[
                ScenarioStep(action="send_text", text="hello"),
                ScenarioStep(action="wait", duration_s=1.0),
            ],
        )
        assert s.name == "custom"
        assert len(s.steps) == 2


# ===== sim_client result tests =====


class TestScenarioResult:
    def test_latency_none_when_no_data(self):
        r = ScenarioResult(scenario_name="test")
        assert r.send_to_transcript_in_ms is None
        assert r.send_to_transcript_out_ms is None
        assert r.send_to_audio_response_ms is None
        assert r.send_to_tool_result_ms is None
        assert r.total_round_trip_ms is None

    def test_latency_calculation(self):
        r = ScenarioResult(scenario_name="test")
        r.last_send_time = 100.0
        r.first_transcript_out_time = 100.5
        r.first_audio_response_time = 100.7
        assert r.send_to_transcript_out_ms == pytest.approx(500.0)
        assert r.send_to_audio_response_ms == pytest.approx(700.0)
        # Total round-trip = min of transcript_out and audio = 500ms
        assert r.total_round_trip_ms == pytest.approx(500.0)

    def test_total_round_trip_uses_earliest(self):
        r = ScenarioResult(scenario_name="test")
        r.last_send_time = 100.0
        r.first_audio_response_time = 100.3
        r.first_transcript_out_time = 100.8
        assert r.total_round_trip_ms == pytest.approx(300.0)

    def test_tool_result_latency(self):
        r = ScenarioResult(scenario_name="test")
        r.last_send_time = 100.0
        r.first_tool_result_time = 102.0
        assert r.send_to_tool_result_ms == pytest.approx(2000.0)

    def test_transcript_in_latency(self):
        r = ScenarioResult(scenario_name="test")
        r.last_send_time = 50.0
        r.first_transcript_in_time = 50.1
        assert r.send_to_transcript_in_ms == pytest.approx(100.0)


class TestFormatLatency:
    def test_none(self):
        assert format_latency(None) == "N/A"

    def test_integer_ms(self):
        assert format_latency(500.0) == "500ms"

    def test_rounds_to_integer(self):
        assert format_latency(123.456) == "123ms"

    def test_zero(self):
        assert format_latency(0.0) == "0ms"
