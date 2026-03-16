"""Consolidated tools for the Orly agent (hackathon-stable set).

Three tools cover the full demo surface:
    overlay — create/show_scene/advance/flip/remove/clear overlays
    query   — refresh_view/overlay_state/session_manifest
    music   — play/stop/pause/resume/replay background music

Removed tools (programs, video) live in tools_roadmap.py.
"""

from client.renderer.registry import valid_types as _registry_valid_types


# ---------------------------------------------------------------------------
# Placement validation (shared by overlay actions that take placement)
# ---------------------------------------------------------------------------

def _validate_placement(placement: list[float]) -> dict | None:
    """Return an error dict if placement is invalid, else None."""
    if len(placement) != 4:
        return {
            "status": "error",
            "message": f"placement must have exactly 4 values, got {len(placement)}.",
        }
    for i, val in enumerate(placement):
        if val < 0 or val > 1000:
            return {
                "status": "error",
                "message": (
                    f"placement[{i}] = {val} is out of range. "
                    "All values must be between 0 and 1000."
                ),
            }
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
    return None


# ---------------------------------------------------------------------------
# 1. overlay — create, show_scene, advance_step, flip_flashcard, remove, clear
# ---------------------------------------------------------------------------

def overlay(
    action: str,
    placement: list[float] = None,
    title: str = "",
    content_type: str = "",
    data: dict = None,
    overlay_name: str = "",
    scene_name: str = "",
    step_number: int = 0,
) -> dict:
    """Project or manipulate a visual overlay on the student's table.

    Args:
        action: What to do — "create", "show_scene", "advance_step", "flip_flashcard", "remove", or "clear".
        placement: Where to place it, [ymin, xmin, ymax, xmax] normalised 0-1000. Required for "create" and "show_scene".
        title: Label for the overlay. Required for "create".
        content_type: Type of visual for "create" — "graph", "annotation", "highlight", "markdown", "image", "number_line", "geometry", "chemistry", "steps", or "flashcard".
        data: Content-specific data for "create". For "graph": {"expression": "x**2", "x_range": [-5,5], "y_range": [-5,10]}. For "annotation": {"text": "..."}. For "highlight": {"color": "#00ffff"}. For "markdown": {"text": "# Step 1\n..."}. For "image": {"prompt": "...", "style": "default|technical|creative", "include_view": false, "reference_previous": false}. For "steps": {"steps": [{"title": "Step 1", "content": "..."}]}. For "flashcard": {"front": "Q?", "back": "A", "show_back": false}. For "number_line": {"min_val": -5, "max_val": 5, "points": [], "ranges": []}. For "geometry": {"elements": [], "x_range": [-6,6], "y_range": [-6,6]}. For "chemistry": {"atoms": [], "bonds": []}.
        overlay_name: Name of existing overlay for "advance_step", "flip_flashcard", or "remove".
        scene_name: Name of a previously generated image for "show_scene".
        step_number: Which step to advance to (1-based) for "advance_step".

    Returns:
        dict with status.
    """
    if data is None:
        data = {}
    if placement is None:
        placement = []

    if action == "create":
        valid_types = _registry_valid_types()
        if content_type not in valid_types:
            return {
                "status": "error",
                "message": (
                    f"Invalid content_type '{content_type}'. "
                    f"Must be one of: {', '.join(sorted(valid_types))}."
                ),
            }
        err = _validate_placement(placement)
        if err:
            return err
        return {
            "status": "displayed",
            "content_type": content_type,
            "title": title,
        }

    if action == "show_scene":
        if not scene_name:
            return {"status": "error", "message": "scene_name is required for show_scene."}
        if placement:
            err = _validate_placement(placement)
            if err:
                return err
        return {
            "status": "showing_scene",
            "scene_name": scene_name,
        }

    if action == "advance_step":
        if not overlay_name:
            return {"status": "error", "message": "overlay_name is required for advance_step."}
        return {
            "status": "advancing",
            "overlay_name": overlay_name,
            "step_number": step_number,
        }

    if action == "flip_flashcard":
        if not overlay_name:
            return {"status": "error", "message": "overlay_name is required for flip_flashcard."}
        return {
            "status": "flipping",
            "overlay_name": overlay_name,
        }

    if action == "remove":
        if not overlay_name:
            return {"status": "error", "message": "overlay_name is required for remove."}
        return {
            "status": "removing",
            "overlay_name": overlay_name,
        }

    if action == "clear":
        return {"status": "clearing"}

    return {"status": "error", "message": f"Unknown overlay action '{action}'."}


# ---------------------------------------------------------------------------
# 2. query — refresh_view, overlay_state, session_manifest
# ---------------------------------------------------------------------------

def query(target: str, reason: str = "") -> dict:
    """Query the table state or request a fresh camera view.

    Args:
        target: What to query — "fresh_view", "overlay_state", or "session_manifest".
        reason: Why you need a fresh view (only for "fresh_view").

    Returns:
        dict with status.
    """
    if target == "fresh_view":
        return {
            "status": "refreshing",
            "reason": reason,
            "description": "Overlays temporarily hidden. Next frame will be a fresh clean view.",
        }

    if target == "overlay_state":
        return {
            "status": "fetching",
            "description": "Overlay state requested. Results will arrive as a notification.",
        }

    if target == "session_manifest":
        return {
            "status": "fetching",
            "description": "Session manifest requested. Results will arrive as a notification.",
        }

    return {"status": "error", "message": f"Unknown query target '{target}'."}


# ---------------------------------------------------------------------------
# 3. music — play, stop, pause, resume, replay
# ---------------------------------------------------------------------------

def music(
    action: str,
    name: str = "",
    prompt: str = "",
    bpm: int = 120,
    temperature: float = 1.0,
    guidance: float = 3.0,
) -> dict:
    """Control AI-generated background music on the table.

    Args:
        action: What to do — "play", "stop", "pause", "resume", or "replay".
        name: Name for the music track. Required for "play" and "replay".
        prompt: Description of the music to generate (only for "play").
        bpm: Beats per minute, 60-200 (only for "play", default 120).
        temperature: Randomness 0.0-3.0 (only for "play", default 1.0).
        guidance: Prompt adherence 0.0-6.0 (only for "play", default 3.0).

    Returns:
        dict with status.
    """
    if action == "play":
        if not name:
            return {"status": "error", "message": "name is required for play."}
        if not prompt:
            return {"status": "error", "message": "prompt is required for play."}
        return {
            "status": "starting",
            "name": name,
            "prompt": prompt,
        }

    if action == "stop":
        return {"status": "stopping", "name": name}

    if action == "pause":
        return {"status": "pausing"}

    if action == "resume":
        return {"status": "resuming"}

    if action == "replay":
        if not name:
            return {"status": "error", "message": "name is required for replay."}
        return {"status": "replaying", "name": name}

    return {"status": "error", "message": f"Unknown music action '{action}'."}
