"""Async code generation using Gemini 3 Flash for TableAPI programs."""

import logging
import re
import threading

logger = logging.getLogger(__name__)

CODE_GEN_MODEL = "gemini-3-flash-preview"

CODE_GEN_SYSTEM_PROMPT = """\
You are a Python code generator for a projected AR table called TableLight.
You write mini-programs that run on the table and interact with the student's
physical workspace through overlays, object tracking, and sound.

Your code runs in a restricted sandbox with access to the `table` API object,
plus `np` (numpy), `cv2` (OpenCV), `math`, and `time`.

## TableAPI Reference (available as `table`)

### Overlay Control
- table.place_overlay(name, content_type, placement, data)
    Place a named overlay. content_type: "annotation", "graph", "markdown",
    "highlight", "number_line", "geometry", "chemistry", "steps", "flashcard".
    placement: [ymin, xmin, ymax, xmax] normalised 0-1000.
    data: dict specific to content_type.
- table.remove_overlay(name) — Remove by name.
- table.clear_overlays() — Remove all overlays.
- table.get_overlay_state() — JSON description of current overlays.

### Camera
- table.get_frame() — Latest rectified BGR numpy array from the camera.

### Object Tracking
- table.track_color(name, region, hsv_range=None)
    Start tracking a colored object. region: [x, y, w, h] in pixel coords.
    hsv_range: optional dict {"lower": [h,s,v], "upper": [h,s,v]}.
- table.track_template(name, template) — Track by template image (numpy array).
- table.get_tracked(name) — Returns dict {name, center, bbox, visible, method} or None.
- table.get_all_tracked() — Returns dict of all tracked objects.
- table.add_zone(name, bbox, on_enter=None, on_exit=None)
    Add a trigger zone. bbox: [x, y, w, h]. Callbacks called when tracked
    objects enter/exit the zone.
- table.remove_zone(name) — Remove a zone.

### Sound
- table.play_tone(frequency, duration) — Play a sine wave tone.

### Communication
- table.notify(message) — Send a text notification back to the voice agent.
- table.log(message) — Log a debug message.

### Session
- table.save_image(name, image) — Save a BGR numpy array to session storage.
- table.load_image(name) — Load a saved image by name.

### Frame Callbacks
- table.on_frame(callback) — Register callback(frame) called each video frame.

### Lifecycle
- table.stop() — Signal this program to stop.
- table.stopped — True if program has been asked to stop.

## Rules
- Always check `table.stopped` in loops to allow graceful shutdown.
- Use `table.notify()` to communicate status back to the voice agent.
- Use `table.log()` for debugging.
- Do NOT import os, sys, subprocess, socket, or other forbidden modules.
- Write clean, well-structured code. Include comments for clarity.
- Output ONLY the Python code. No explanation, no markdown fences.
"""


def extract_code(text: str) -> str:
    """Extract Python code from a response, stripping markdown fences if present."""
    # Try to find a Python code block
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fences, assume the entire text is code
    return text.strip()


class CodeGenerator:
    """Generates Python code using Gemini 3 Flash for the TableAPI."""

    def __init__(self, session_store, validate_fn, notify_fn):
        """
        Args:
            session_store: SessionStore instance for saving generated code.
            validate_fn: Callable[[str], tuple[bool, str]] — validates code
                         (typically program_runtime.validate_code).
            notify_fn: Callable[[str], None] — sends notification to backend.
        """
        self._session_store = session_store
        self._validate_fn = validate_fn
        self._notify_fn = notify_fn

    def generate_async(self, name: str, description: str, context: str = "") -> None:
        """Start async code generation in a background thread."""
        thread = threading.Thread(
            target=self._generate_code_thread,
            args=(name, description, context),
            daemon=True,
        )
        thread.start()

    def _generate_code_thread(self, name: str, description: str, context: str) -> None:
        """Background thread: call Gemini, extract code, validate, save, notify."""
        try:
            from client.genai_utils import get_genai_client
            from google.genai import types

            client = get_genai_client()

            prompt = f"Write a TableAPI program that does the following:\n{description}"
            if context:
                prompt += f"\n\nAdditional context:\n{context}"

            response = client.models.generate_content(
                model=CODE_GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=CODE_GEN_SYSTEM_PROMPT,
                ),
            )

            raw_text = response.text or ""
            code = extract_code(raw_text)

            if not code:
                self._notify_fn(
                    f"Code generation for '{name}' failed: model returned empty response."
                )
                return

            # Validate
            valid, error = self._validate_fn(code)
            if not valid:
                # Retry once with the error
                logger.warning("Generated code failed validation: %s. Retrying.", error)
                retry_prompt = (
                    f"{prompt}\n\nYour previous attempt had a validation error: {error}\n"
                    "Please fix the issue and try again."
                )
                response = client.models.generate_content(
                    model=CODE_GEN_MODEL,
                    contents=retry_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=CODE_GEN_SYSTEM_PROMPT,
                    ),
                )
                code = extract_code(response.text or "")
                valid, error = self._validate_fn(code)
                if not valid:
                    self._notify_fn(
                        f"Code generation for '{name}' failed validation: {error}"
                    )
                    return

            # Save
            self._session_store.save_program(name, code)
            logger.info("Generated code '%s' saved.", name)

            self._notify_fn(
                f"Code '{name}' generated and saved. "
                f"Description: {description}. "
                f"Use run_program to execute it."
            )
        except Exception as e:
            logger.error("Code generation failed for '%s': %s", name, e)
            self._notify_fn(f"Code generation for '{name}' failed: {e}")
