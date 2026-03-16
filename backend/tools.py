"""Overlay projection tool for the Lumi tutor agent."""

from client.renderer.registry import valid_types as _registry_valid_types


def project_overlay(
    content_type: str,
    placement: list[float],
    title: str,
    data: dict,
) -> dict:
    """Project a visual overlay onto the student's work surface via projector.

    Args:
        content_type: Type of visual — "graph", "annotation", "highlight", "markdown",
              "image", "number_line", "geometry", "chemistry", "steps", or "flashcard".
        placement: Where to place it on the table, [ymin, xmin, ymax, xmax] normalised 0-1000.
                   Choose empty space near relevant content. Never overlap existing work.
                   For markdown and annotation, use a LARGE box (at least 500 units wide
                   and 400 units tall) so text is readable on the projector.
        title: Label for the overlay.
        data: Content-specific data. For "graph": {"expression": "x**2 - 3*x + 2",
              "x_range": [-5, 5], "y_range": [-5, 10]}. For "annotation": {"text": "..."}.
              For "highlight": {"color": "#00ffff", "target": [ymin, xmin, ymax, xmax]}.
              For "markdown": {"text": "# Step 1\n\nFactor: $x^2+3x+2 = (x+1)(x+2)$\n\n- Root 1: **x = -1**\n- Root 2: **x = -2**"}.
              Markdown supports # headers, **bold**, - bullet lists, and $latex$ math. Prefer markdown over annotation for multi-step explanations.
              For "image": {"prompt": "a labeled unit circle showing sin and cos",
              "style": "technical", "include_view": true, "reference_previous": false}.
              Generates an image via AI and projects it.
              "style": "default" (generates exactly what you describe), "technical"
              (diagrams on black bg with bright colors), or "creative" (rich colorful
              illustrations). Default is "default".
              "include_view": true to pass the current camera view as reference (for
              incorporating what's on the table).
              "reference_previous": true to pass the last generated image as reference
              (for iterating on/modifying a previous image — e.g. "add a dragon to the
              scene", "make it more colorful", "now show the next page of the story").
              "reference_scene": "Scene 1" to use a specific named scene as reference
              instead of the most recent image. The name must match a previous title.
              For "flashcard": {"front": "What is 2+2?", "back": "4", "show_back": false}.

    Returns:
        dict with status of the projection.
    """
    _VALID_TYPES = _registry_valid_types()

    # Validate content_type.
    if content_type not in _VALID_TYPES:
        return {
            "status": "error",
            "message": (
                f"Invalid content_type '{content_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_TYPES))}."
            ),
        }

    # Validate placement length.
    if len(placement) != 4:
        return {
            "status": "error",
            "message": f"placement must have exactly 4 values, got {len(placement)}.",
        }

    # Validate placement range.
    for i, val in enumerate(placement):
        if val < 0 or val > 1000:
            return {
                "status": "error",
                "message": (
                    f"placement[{i}] = {val} is out of range. "
                    "All values must be between 0 and 1000."
                ),
            }

    # Validate geometric sense: [ymin, xmin, ymax, xmax].
    ymin, xmin, ymax, xmax = placement
    if ymin >= ymax:
        return {
            "status": "error",
            "message": f"placement ymin ({ymin}) must be less than ymax ({ymax}).",
        }
    if xmin >= xmax:
        return {
            "status": "error",
            "message": f"placement xmin ({xmin}) must be less than xmax ({xmax}).",
        }

    return {
        "status": "displayed",
        "content_type": content_type,
        "placement": placement,
        "title": title,
    }


def show_scene(scene_name: str, placement: list[float]) -> dict:
    """Display a previously generated image scene on the projector.

    Use this to switch between scenes in a story, or to show a previous
    image again. The scene_name must match the title used when the image
    was originally generated with project_overlay.

    Args:
        scene_name: The title/name of a previously generated image.
                    Must exactly match a title from a prior project_overlay
                    call with content_type "image".
        placement: Where to place it, [ymin, xmin, ymax, xmax] normalised 0-1000.

    Returns:
        dict with status.
    """
    return {
        "status": "showing_scene",
        "scene_name": scene_name,
        "placement": placement,
    }


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


def get_overlay_state() -> dict:
    """Get the current state of all overlays on the table.

    Returns a description of all active overlays with their names,
    types, positions, and an ASCII grid visualization of the table layout.
    Use this to understand what's currently projected before making changes.

    The actual state is on the edge client. This tool triggers a query —
    the full overlay state will arrive as a notification shortly after.

    Returns:
        dict with status. Actual overlay state arrives as a notification.
    """
    return {
        "status": "fetching",
        "description": "Overlay state requested. Results will arrive as a notification.",
    }


def refresh_view(reason: str) -> dict:
    """Temporarily hide overlays and capture a fresh view of the table.

    Call this when you need to see the current state of the student's work
    without your projected overlays blocking the view. The overlays will be
    restored after capture. The next video frame you receive will be the
    fresh clean view.

    Args:
        reason: Why you need a fresh view (e.g. "check student's new work",
                "verify overlay position"). Logged for debugging.

    Returns:
        dict with status of the refresh.
    """
    return {
        "status": "refreshing",
        "reason": reason,
        "description": "Overlays temporarily hidden. Next frame will be a fresh clean view.",
    }


def flip_flashcard(overlay_name: str) -> dict:
    """Flip a flashcard overlay to show the other side.

    Args:
        overlay_name: Name of the flashcard overlay to flip.

    Returns:
        dict with status of the flip.
    """
    return {
        "status": "flipping",
        "overlay_name": overlay_name,
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


def play_music(
    name: str,
    prompt: str,
    bpm: int = 120,
    temperature: float = 1.0,
    guidance: float = 3.0,
) -> dict:
    """Start playing AI-generated background music.

    Music plays at low volume alongside your voice. The student hears both.
    Music generation is asynchronous — wait for the notification before
    telling the student it's playing.

    Args:
        name: Name for the music track (e.g. "gentle-lullaby").
        prompt: Description of the music (e.g. "gentle piano lullaby with soft strings").
        bpm: Beats per minute, 60-200 (default 120).
        temperature: Randomness 0.0-3.0 (default 1.0).
        guidance: Prompt adherence 0.0-6.0 (default 3.0).

    Returns:
        dict with status.
    """
    return {
        "status": "starting",
        "name": name,
        "prompt": prompt,
        "bpm": bpm,
        "temperature": temperature,
        "guidance": guidance,
    }


def stop_music(name: str = "") -> dict:
    """Stop the currently playing background music.

    The music track is saved to session storage for later replay.

    Args:
        name: Optional name of the track to stop (currently only one track at a time).

    Returns:
        dict with status.
    """
    return {
        "status": "stopping",
        "name": name,
    }


def pause_music() -> dict:
    """Pause the background music (keeps session alive for resuming).

    Returns:
        dict with status.
    """
    return {"status": "pausing"}


def resume_music() -> dict:
    """Resume paused background music.

    Returns:
        dict with status.
    """
    return {"status": "resuming"}


def replay_music(name: str) -> dict:
    """Replay a previously generated and saved music track.

    Args:
        name: Name of the saved track to replay.

    Returns:
        dict with status.
    """
    return {
        "status": "replaying",
        "name": name,
    }


def get_session_manifest() -> dict:
    """Get the manifest of all session artifacts (images, music, videos, programs).

    The actual manifest is on the edge client. Results arrive as a notification.

    Returns:
        dict with status. Actual manifest arrives as a notification.
    """
    return {
        "status": "fetching",
        "description": "Session manifest requested. Results will arrive as a notification.",
    }


def advance_step(overlay_name: str, step_number: int) -> dict:
    """Advance a step-by-step overlay to show the next step.

    Use this after projecting a "steps" overlay to reveal steps one at a time.
    The overlay will animate to show the specified step number.

    Args:
        overlay_name: Name of the steps overlay to advance.
        step_number: Which step to advance to (1-based).

    Returns:
        dict with status of the advancement.
    """
    return {
        "status": "advancing",
        "overlay_name": overlay_name,
        "step_number": step_number,
    }
