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

from backend.tools import project_overlay, refresh_view, show_scene

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

_TOOL_FUNCTIONS: list[Callable] = [project_overlay, refresh_view, show_scene]

TOOL_DECLARATIONS: list[dict] = [function_to_declaration(f) for f in _TOOL_FUNCTIONS]

TOOL_REGISTRY: dict[str, Callable] = {f.__name__: f for f in _TOOL_FUNCTIONS}
