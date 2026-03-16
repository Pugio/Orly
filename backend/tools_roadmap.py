"""Roadmap tools — temporarily removed for hackathon stability.

These tools work but increase the tool-declaration payload sent to
Gemini Live, which degrades function-calling reliability.  They can
be re-enabled by importing them in agent.py and adding them to
_TOOL_FUNCTIONS.

Covered features:
    - Interactive mini-programs (run/stop/list/generate_code)
    - AI video generation + playback (generate_video/play_video/stop_video)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------


def run_program(name: str, code: str = "", description: str = "") -> dict:
    """Run a mini-program on the table surface.

    Programs execute on the edge client with access to the `table` API for
    real-time interaction — tracking objects, placing overlays, playing sounds,
    and responding to the camera feed at frame rate.

    The code runs in a restricted Python environment with access to:
    - `table` — the TableAPI (overlays, tracking, sounds, notifications)
    - `np` — numpy for array/math operations
    - `cv2` — OpenCV for image processing
    - `math` — standard math functions
    - `time` — time.time() and time.sleep()

    To respond to camera frames, register an on_frame callback:
        def on_frame(frame):
            tracked = table.get_tracked("toy")
            if tracked and tracked["visible"]:
                y, x = tracked["center"]
                table.place_overlay("marker", "highlight",
                    [y-25, x-25, y+25, x+25], {"color": "#00ff00"})
        table.on_frame(on_frame)

    Args:
        name: Unique name for the program.
        code: Python source code to execute. If empty, loads previously
              saved code from the session (e.g. from generate_code).
        description: What the program does.

    Returns:
        dict with status.
    """
    return {
        "status": "started",
        "name": name,
        "description": description,
    }


def stop_program(name: str) -> dict:
    """Stop a running mini-program.

    Args:
        name: Name of the program to stop.

    Returns:
        dict with status.
    """
    return {
        "status": "stopping",
        "name": name,
    }


def list_programs() -> dict:
    """List all running mini-programs and their status.

    The actual program list is on the edge client. This tool triggers
    a query — the results will arrive as a notification shortly after.

    Returns:
        dict with status. Actual program list arrives as a notification.
    """
    return {
        "status": "fetching",
        "description": "Program list requested. Results will arrive as a notification.",
    }


def generate_code(name: str, description: str, context: str = "") -> dict:
    """Generate a mini-program using AI code generation.

    Uses a code-specialized model to write Python code that uses the TableAPI.
    The code is saved as a session artifact. After generation completes
    (you'll receive a notification), use run_program to execute it.

    Code generation is asynchronous — do NOT assume the code is ready until
    you receive a completion notification.

    Args:
        name: Name for the generated program.
        description: What the program should do. Be specific about the
                     interactive behaviour, tracking, overlays, and sounds.
        context: Additional context (e.g. what's on the table, the student's
                 question, or specific API calls to use).

    Returns:
        dict with status.
    """
    return {
        "status": "generating",
        "name": name,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def generate_video(
    name: str,
    prompt: str,
    placement: list[float],
    duration: int = 5,
    aspect_ratio: str = "16:9",
) -> dict:
    """Generate a short AI video and project it onto the table.

    Video generation is asynchronous and takes 1-6 minutes. A loading
    placeholder will be shown while generating. Do NOT tell the child
    the video is ready until you receive the completion notification.

    Args:
        name: Unique name for the video (e.g. "cat-piano-clip").
        prompt: Description of the video to generate.
        placement: Where to display it, [ymin, xmin, ymax, xmax] normalised 0-1000.
        duration: Video duration in seconds (4, 5, 6, or 8).
        aspect_ratio: "16:9" or "9:16".

    Returns:
        dict with status.
    """
    if duration not in (4, 5, 6, 8):
        return {"status": "error", "message": f"duration must be 4, 5, 6, or 8, got {duration}"}
    return {
        "status": "generating",
        "name": name,
        "prompt": prompt,
        "placement": placement,
        "duration": duration,
    }


def play_video(name: str, placement: list[float], loop: bool = False) -> dict:
    """Play a previously generated and saved video on the table.

    Args:
        name: Name of the video to play (must match a previous generate_video title).
        placement: Where to display it, [ymin, xmin, ymax, xmax] normalised 0-1000.
        loop: Whether to loop the video continuously.

    Returns:
        dict with status.
    """
    return {
        "status": "playing",
        "name": name,
        "placement": placement,
        "loop": loop,
    }


def stop_video(name: str) -> dict:
    """Stop a playing video.

    Args:
        name: Name of the video to stop.

    Returns:
        dict with status.
    """
    return {
        "status": "stopping",
        "name": name,
    }
