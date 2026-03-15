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


def run_program(name: str, code: str, description: str = "") -> dict:
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
        code: Python source code to execute.
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
