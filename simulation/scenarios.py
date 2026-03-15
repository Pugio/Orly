"""Pre-defined test scenarios for the simulation harness.

Each scenario is a dataclass describing what audio/video to send and
what responses to expect. The sim_client executes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioStep:
    """A single step in a scenario.

    Attributes:
        action: "send_audio", "send_video", "send_text", "wait", "send_silence".
        duration_s: Duration in seconds (for audio/silence/wait).
        text: Text content (for send_text or TTS audio).
        image_path: Path to image file (for send_video with a file).
        video_text_lines: Custom text for generated test frames.
    """

    action: str
    duration_s: float = 0.0
    text: str = ""
    image_path: str = ""
    video_text_lines: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    """A complete test scenario.

    Attributes:
        name: Human-readable name.
        description: What the scenario tests.
        steps: Ordered list of steps to execute.
        expect_transcript_in: If True, expect input transcription in responses.
        expect_transcript_out: If True, expect output transcription in responses.
        expect_tool_call: If True, expect a tool_result in responses.
        max_wait_s: Maximum seconds to wait for all responses after steps complete.
    """

    name: str
    description: str
    steps: list[ScenarioStep]
    expect_transcript_in: bool = False
    expect_transcript_out: bool = False
    expect_tool_call: bool = False
    max_wait_s: float = 30.0


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

SILENCE_ONLY = Scenario(
    name="silence_only",
    description="Send 3s of silence + a video frame. Verify no phantom transcription.",
    steps=[
        ScenarioStep(action="send_video"),
        ScenarioStep(action="send_silence", duration_s=3.0),
        ScenarioStep(action="wait", duration_s=5.0),
    ],
    expect_transcript_in=False,
    expect_transcript_out=False,
    max_wait_s=10.0,
)

SIMPLE_QUESTION = Scenario(
    name="simple_question",
    description="Ask 'What is 2 plus 2?' via text and measure round-trip.",
    steps=[
        ScenarioStep(action="send_video"),
        ScenarioStep(action="send_silence", duration_s=1.0),
        ScenarioStep(action="send_text", text="What is 2 plus 2?"),
        ScenarioStep(action="wait", duration_s=15.0),
    ],
    expect_transcript_out=True,
    max_wait_s=20.0,
)

WITH_TOOL_CALL = Scenario(
    name="with_tool_call",
    description="Ask to graph y=x^2, which should trigger project_overlay.",
    steps=[
        ScenarioStep(action="send_video"),
        ScenarioStep(action="send_silence", duration_s=1.0),
        ScenarioStep(
            action="send_text",
            text="Please graph y equals x squared for me. Use the project_overlay tool with content_type graph.",
        ),
        ScenarioStep(action="wait", duration_s=20.0),
    ],
    expect_transcript_out=True,
    expect_tool_call=True,
    max_wait_s=30.0,
)

INTERRUPTION = Scenario(
    name="interruption",
    description="Send a question, then interrupt with new audio while model responds.",
    steps=[
        ScenarioStep(action="send_video"),
        ScenarioStep(action="send_silence", duration_s=1.0),
        ScenarioStep(action="send_text", text="Tell me a very long story about a dragon."),
        ScenarioStep(action="wait", duration_s=3.0),
        # Interrupt with a new question
        ScenarioStep(action="send_text", text="Stop. What is 5 plus 5?"),
        ScenarioStep(action="wait", duration_s=15.0),
    ],
    expect_transcript_out=True,
    max_wait_s=25.0,
)

# All built-in scenarios, keyed by name.
ALL_SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [SILENCE_ONLY, SIMPLE_QUESTION, WITH_TOOL_CALL, INTERRUPTION]
}
