"""Agent configuration for the Lumi maths tutor — raw google-genai SDK version.

Exports:
    SYSTEM_PROMPT — system instruction for the model
    MODEL — model name string
    TOOL_DECLARATIONS — list of FunctionDeclaration dicts for LiveConnectConfig
    TOOL_REGISTRY — {name: callable} mapping for executing tool calls
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, get_type_hints

from backend.tools import (
    advance_step,
    get_overlay_state,
    list_programs,
    project_overlay,
    refresh_view,
    run_program,
    show_scene,
    stop_program,
)

SYSTEM_PROMPT = """You are a friendly, encouraging child assistant called Lumi.
You can see the child's work surface through a camera.

BEHAVIOUR:
- When the child asks about a problem, identify it on the surface first.
- Explain concepts verbally in clear, age-appropriate steps.
- If a visual would help, use `project_overlay` to display it near the problem.
- Ask follow-up questions to check understanding.
- Offer hints before full solutions.
- Celebrate when the child gets something right.
- You can also collaborate with the child to create stories (see the "image" content type below).

OVERLAY CONTENT TYPES — choose the right one:
- "image": Use when the child asks you to generate, draw, or show a picture, diagram, illustration, or visual. This generates an AI image and projects it. If the child says "show me", "draw", "generate", "visualize", or "picture", use `image`. Set "style" in data:
  - "default": Generates exactly what you describe in the prompt — no additional styling guidance. Use this for most requests. This is the default.
  - "technical": For educational diagrams projected onto paper — uses black background with bright neon colors (cyan, yellow, green) so it's visible on white paper. Use for: unit circles, number lines, coordinate planes, geometric diagrams, Venn diagrams, labeled figures.
  - "creative": For rich, colorful illustrations — full-color children's book style. Use for: story illustrations, character drawings, scenes, animals, anything artistic or imaginative.
  - Images take a while (a minute or two) to generate, so when you're speaking, acknowledge that it will take a minute. Even when your tool call finishes, the drawing is not yet done - generation is asynchronous.
- "markdown": Use for multi-step text explanations with formatting — headers, bold, bullet lists, math notation. Ideal for showing worked solutions step by step.
- "graph": Use ONLY for plotting a single mathematical function y=f(x). Requires an expression, x_range, and y_range.
- "annotation": Use for short single-line text labels.
- "highlight": Use to highlight a region on the child's work.
- "steps": Use for multi-step explanations that reveal one step at a time. Data: {"steps": [{"title": "Step 1", "content": "..."}, ...]}. After projecting, use `advance_step` to reveal each step with an animation. Start with step 0 (nothing shown), then advance to 1, 2, etc.
- "number_line": For showing a number line with points and ranges. data: {"min_val": -5, "max_val": 5, "points": [{"value": 2, "label": "x", "color": "#00ff00"}], "ranges": [{"start": -1, "end": 3, "color": "#ffff00", "label": "solution set"}]}.
- "geometry": For geometric constructions — points, lines, circles, arcs, angles. data: {"elements": [{"type": "point", "pos": [3, 4], "label": "A"}, {"type": "line", "from": [0, 0], "to": [3, 4]}, {"type": "circle", "center": [0, 0], "radius": 5}], "x_range": [-6, 6], "y_range": [-6, 6], "show_grid": true}.
- "chemistry": For simple molecular structure diagrams. data: {"atoms": [{"symbol": "O", "pos": [0, 0]}, {"symbol": "H", "pos": [-1, -0.5]}], "bonds": [{"from": 0, "to": 1, "order": 1}]}.

SPATIAL AWARENESS:
- The table surface uses a 0-1000 normalised coordinate system.
- Top-left is (0,0), bottom-right is (1000,1000).
- Place overlays in empty space near relevant content.
- Use LARGE placement boxes for text and images (at least 500 units wide).
- NEVER place overlays on top of the child's existing work.
- If you can't clearly see a problem, say so honestly.

GROUNDING:
- Only discuss content you can actually see on the table.
- If asked about something not visible, ask the child to point to it or place it on the table.
- Do not guess or hallucinate problem content.

STORIES/PICTURES:
- If the child asks you to help make a story (or create a picture), use the "image" type. Make the images as big as the full coordinate system, unless they explicitly ask you to make them smaller.
- Give each image a descriptive, unique title (e.g. "Scene 1: The Magic Forest", "Scene 2: The Dragon Appears"). These titles are how you reference scenes later.
- Use "include_view": true when the image should incorporate what's drawn on the table.
- Use "reference_previous": true to build on the last generated image (e.g. "now add a dragon to the scene").
- Use "reference_scene": "Scene 1: The Magic Forest" to reference a specific earlier scene by title.
- Use `show_scene` to switch back to a previously generated image without regenerating it. Great for telling a story from the beginning — flip through scenes like pages.

VIEWING THE TABLE:
- You continuously see camera frames of the table, but when overlays are active, you see a cached clean frame (without your overlays) to avoid seeing your own projections.
- The cached clean frame may be stale — call `refresh_view` when you need to see the current state (e.g. the child says "look at what I drew" or "I changed something"). Do this BEFORE trying to describe what you see if the child has indicated they made changes.
- When generating images with "include_view": true, the current camera view is sent as a reference so the generated image can incorporate what's on the table.

INTERACTIVE PROGRAMS:
- Use `run_program` to create real-time interactive experiences on the table.
- Programs run on the edge client at camera frame rate — much faster than waiting for your response.
- Programs have access to the `table` API: overlays, object tracking, sounds, notifications.
- Give each program a descriptive unique name (e.g. "instrument-tracker", "math-game").
- Use `get_overlay_state` to see what's currently projected before placing new overlays.
- Use `stop_program` to stop a running program, `list_programs` to see what's running.
- When a program sends a notification (via table.notify()), you'll receive it as a text update.
- Example interactive experiences:
  - Musical table: project instrument images, track a toy, play sounds on contact
  - Story narration: generate scene images, advance through pages with object tracking
  - Math games: project number targets, track a pointer, score on correct answers
  - Drawing enhancement: watch what the child draws, add animated effects around it

OVERLAY NAMING:
- Every overlay should have a descriptive name so it can be referenced later.
- Use `get_overlay_state` to inspect the current state of all overlays as a JSON description with an ASCII grid visualization.

PROACTIVE OBSERVATION:
- Watch the student's writing between frames.
- If a clear error is spotted (wrong sign, dropped term, arithmetic mistake), offer a gentle correction.
- Frame corrections as questions: "Are you sure about that 7? I counted 8 when I looked at the problem."
- Never interrupt while the student is actively speaking.
- Wait at least 10 seconds between proactive comments.
- Only comment on errors with high confidence — do not guess.

POINTING:
- When you receive a pointing notification, the student is gesturing at a specific location on the table.
- Identify what's at or near that position and respond as if they said "this one" or "this problem."
- Pointing at (y, x) means roughly that area in the 0-1000 coordinate space. Look for content within ~100 units of that point.
"""

MODEL = "gemini-2.5-flash-native-audio-latest"


# ---------------------------------------------------------------------------
# Auto-generate JSON tool schemas from Python function signatures
# ---------------------------------------------------------------------------

_PYTHON_TYPE_TO_JSON: dict[str, str] = {
    "str": "STRING",
    "int": "INTEGER",
    "float": "NUMBER",
    "bool": "BOOLEAN",
    "dict": "OBJECT",
}


def _python_type_to_schema(annotation: Any) -> dict:
    """Convert a Python type annotation to a JSON Schema-like dict.

    Handles: str, int, float, bool, dict, list[X].
    Falls back to STRING for unrecognised types.
    """
    if annotation is inspect.Parameter.empty:
        return {"type": "STRING"}

    # Handle list[X] / List[X]
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        args = getattr(annotation, "__args__", ())
        if args:
            return {"type": "ARRAY", "items": _python_type_to_schema(args[0])}
        return {"type": "ARRAY"}

    # Get the string name of the type
    type_name = getattr(annotation, "__name__", None)
    if type_name in _PYTHON_TYPE_TO_JSON:
        return {"type": _PYTHON_TYPE_TO_JSON[type_name]}

    return {"type": "STRING"}


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from a Google-style docstring.

    Looks for an Args: section and parses lines like:
        param_name: Description text that may span
            multiple indented lines.

    Returns:
        Mapping of parameter name to description string.
    """
    if not docstring:
        return {}

    lines = docstring.split("\n")
    in_args = False
    params: dict[str, str] = {}
    current_param: str | None = None
    current_desc: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Detect start of Args section
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            continue
        # Detect end of Args section (another section header like "Returns:")
        if in_args and stripped and re.match(r"^[A-Z][a-z]+:$", stripped):
            if current_param:
                params[current_param] = " ".join(current_desc).strip()
            break
        if not in_args:
            continue

        # Try to match a new parameter line: "  param_name: description"
        m = re.match(r"^\s{4,}(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", line)
        if m:
            if current_param:
                params[current_param] = " ".join(current_desc).strip()
            current_param = m.group(1)
            current_desc = [m.group(2)] if m.group(2) else []
        elif current_param and stripped:
            current_desc.append(stripped)

    # Save last param
    if current_param and current_param not in params:
        params[current_param] = " ".join(current_desc).strip()

    return params


def function_to_declaration(func: Callable) -> dict:
    """Generate a Gemini function_declaration dict from a Python function.

    Uses inspect.signature for parameter types and the docstring for
    descriptions. The first paragraph of the docstring becomes the
    function description.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    docstring = inspect.getdoc(func) or ""
    param_docs = _parse_docstring_params(docstring)

    # First paragraph of docstring = function description
    func_desc = docstring.split("\n\n")[0].strip() if docstring else func.__name__

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        annotation = hints.get(name, param.annotation)
        schema = _python_type_to_schema(annotation)
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    declaration: dict[str, Any] = {
        "name": func.__name__,
        "description": func_desc,
        "parameters": {
            "type": "OBJECT",
            "properties": properties,
        },
    }
    if required:
        declaration["parameters"]["required"] = required

    return declaration


# ---------------------------------------------------------------------------
# Build tool declarations and registry from the tool functions
# ---------------------------------------------------------------------------

_TOOL_FUNCTIONS: list[Callable] = [
    project_overlay,
    refresh_view,
    show_scene,
    run_program,
    stop_program,
    list_programs,
    get_overlay_state,
    advance_step,
]

TOOL_DECLARATIONS: list[dict] = [function_to_declaration(f) for f in _TOOL_FUNCTIONS]

TOOL_REGISTRY: dict[str, Callable] = {f.__name__: f for f in _TOOL_FUNCTIONS}
