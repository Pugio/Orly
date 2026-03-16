"""
PoC 5 — Gemini function calling for overlay generation.

Validates that Gemini can call project_overlay with correct parameters
(content_type, placement, title, data) when given a rectified table image
and a user question.

Uses the standard (non-streaming) Gemini API with an explicit tool declaration.

Usage:
    python poc/poc5_function_calling.py --image rectified_output.png --api-key YOUR_KEY
    python poc/poc5_function_calling.py --image rectified_output.png --api-key YOUR_KEY \
        --prompt "Can you help me graph y = x^2?"
"""

import argparse
import base64
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def get_tool_declaration() -> dict:
    """Return the project_overlay function declaration for Gemini.

    This is the JSON schema version needed for the standard API.
    Schema matches PROJECT_PLAN.md section 9.5.
    """
    return {
        "name": "project_overlay",
        "description": (
            "Project a visual overlay onto the student's work surface via projector. "
            "Placement in 0-1000 normalised coordinates [ymin, xmin, ymax, xmax]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": ["graph", "diagram", "annotation", "highlight"],
                },
                "placement": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "[ymin, xmin, ymax, xmax] 0-1000. "
                        "Choose empty space near relevant content."
                    ),
                },
                "title": {"type": "string"},
                "data": {
                    "type": "object",
                    "description": (
                        "For 'graph': {expression, x_range, y_range}. "
                        "For 'annotation': {text}. "
                        "For 'highlight': {color, target [ymin,xmin,ymax,xmax]}."
                    ),
                },
            },
            "required": ["content_type", "placement", "data"],
        },
    }


def get_system_prompt() -> str:
    """Return the Orly system prompt.

    Reads from backend/agent.py to keep in sync. Falls back to inline
    definition if the file cannot be parsed.
    """
    agent_path = Path(__file__).resolve().parent.parent / "backend" / "agent.py"
    try:
        source = agent_path.read_text()
        # Extract SYSTEM_PROMPT using regex — find the triple-quoted string
        match = re.search(
            r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""', source, re.DOTALL
        )
        if match:
            return match.group(1).strip()
    except (OSError, IOError):
        pass

    # Fallback
    return (
        "You are Orly, a friendly, encouraging maths tutor.\n"
        "You can see the student's work surface through a camera.\n\n"
        "BEHAVIOUR:\n"
        "- When the student asks about a problem, identify it on the surface first.\n"
        "- Explain concepts verbally in clear, age-appropriate steps.\n"
        "- If a visual would help, use project_overlay to display it near the problem.\n"
        "- Ask follow-up questions to check understanding.\n"
        "- Offer hints before full solutions.\n"
        "- Celebrate when the student gets something right.\n\n"
        "SPATIAL AWARENESS:\n"
        "- The table surface uses a 0-1000 normalised coordinate system.\n"
        "- Top-left is (0,0), bottom-right is (1000,1000).\n"
        "- Place overlays in empty space near relevant content.\n"
        "- NEVER place overlays on top of the student's existing work.\n"
        "- If you can't clearly see a problem, say so honestly.\n\n"
        "GROUNDING:\n"
        "- Only discuss content you can actually see on the table.\n"
        "- If asked about something not visible, ask the student to point to it "
        "or place it on the table.\n"
        "- Do not guess or hallucinate problem content."
    )


def parse_tool_calls(response) -> list[dict]:
    """Extract tool calls from a Gemini response.

    Returns list of dicts with keys: name, args (the parsed arguments dict).
    Returns empty list if no tool calls in the response.
    """
    calls = []
    if not response.candidates:
        return calls

    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.function_call is not None:
                fc = part.function_call
                # fc.args may be a dict or a proto MapComposite — convert to dict
                args = dict(fc.args) if not isinstance(fc.args, dict) else fc.args
                calls.append({"name": fc.name, "args": args})

    return calls


_VALID_CONTENT_TYPES = {"graph", "diagram", "annotation", "highlight"}


def validate_overlay_call(call: dict) -> list[str]:
    """Validate a project_overlay tool call's arguments.

    Returns list of error strings. Empty list = valid.
    """
    errors = []
    args = call.get("args", {})

    # content_type
    ct = args.get("content_type")
    if ct not in _VALID_CONTENT_TYPES:
        errors.append(
            f"Invalid content_type '{ct}'. "
            f"Must be one of: {', '.join(sorted(_VALID_CONTENT_TYPES))}."
        )

    # placement
    placement = args.get("placement")
    if placement is None:
        errors.append("Missing placement.")
    elif not isinstance(placement, (list, tuple)) or len(placement) != 4:
        errors.append(
            f"Placement must have exactly 4 values, got "
            f"{len(placement) if isinstance(placement, (list, tuple)) else type(placement).__name__}."
        )
    else:
        for i, val in enumerate(placement):
            if not isinstance(val, (int, float)) or val < 0 or val > 1000:
                errors.append(
                    f"Placement[{i}] = {val} is out of range (0-1000)."
                )
        if len(errors) == 0:  # only check ordering if values are valid
            ymin, xmin, ymax, xmax = placement
            if ymin >= ymax:
                errors.append(
                    f"Placement ymin ({ymin}) must be less than ymax ({ymax})."
                )
            if xmin >= xmax:
                errors.append(
                    f"Placement xmin ({xmin}) must be less than xmax ({xmax})."
                )

    # title
    title = args.get("title")
    if not title or not isinstance(title, str) or not title.strip():
        errors.append("Missing or empty title.")

    # data
    data = args.get("data")
    if data is None:
        errors.append("Missing data dict.")
    else:
        if ct == "graph" and "expression" not in data:
            errors.append("Graph data missing 'expression' field.")
        if ct == "annotation" and "text" not in data:
            errors.append("Annotation data missing 'text' field.")
        if ct == "highlight" and "color" not in data:
            errors.append("Highlight data missing 'color' field.")

    return errors


def render_overlay_preview(call: dict, image: np.ndarray) -> np.ndarray:
    """Render a preview of the overlay on the image.

    Uses the renderers from client/renderer/ to generate the overlay,
    then composites it onto the image at the specified placement coordinates.
    Returns annotated image copy.
    """
    from client.renderer.annotation import _render_annotation_impl as render_annotation
    from client.renderer.graph import _render_graph_impl as render_graph
    from client.renderer.highlight import _render_highlight_impl as render_highlight

    out = image.copy()
    args = call.get("args", {})
    ct = args.get("content_type")
    placement = args.get("placement", [0, 0, 500, 500])
    data = args.get("data", {})
    title = args.get("title", "")

    h, w = out.shape[:2]
    ymin, xmin, ymax, xmax = placement

    # Convert 0-1000 coords to pixel coords
    py1 = int(ymin * h / 1000)
    px1 = int(xmin * w / 1000)
    py2 = int(ymax * h / 1000)
    px2 = int(xmax * w / 1000)

    region_w = max(1, px2 - px1)
    region_h = max(1, py2 - py1)

    overlay = None

    if ct == "graph":
        expression = data.get("expression", "x")
        x_range = data.get("x_range", [-10, 10])
        y_range = data.get("y_range", [-10, 10])
        overlay = render_graph(expression, x_range, y_range, region_w, region_h)

    elif ct == "annotation":
        text = data.get("text", title)
        overlay = render_annotation(text, region_w, region_h)

    elif ct == "highlight":
        color_hex = data.get("color", "#00ffff")
        alpha = data.get("alpha", 0.3)
        overlay_bgra = render_highlight(region_w, region_h, color_hex, alpha)
        # Convert BGRA highlight to BGR for compositing with alpha blend
        bgr = overlay_bgra[:, :, :3]
        a = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
        region = out[py1:py2, px1:px2].astype(np.float32)
        blended = region * (1 - a) + bgr.astype(np.float32) * a
        out[py1:py2, px1:px2] = blended.astype(np.uint8)
        # Draw title label
        cv2.putText(
            out, title, (px1, max(py1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA,
        )
        return out

    elif ct == "diagram":
        # Diagram: render as annotation with the title for now
        overlay = render_annotation(title, region_w, region_h)

    if overlay is not None:
        # Composite: black pixels in overlay are transparent (projector logic)
        # For preview, blend non-black pixels onto the image
        mask = np.any(overlay > 10, axis=2)  # non-black pixels
        region = out[py1:py2, px1:px2]
        region[mask] = overlay[mask]
        out[py1:py2, px1:px2] = region

    # Draw title label above the overlay region
    cv2.putText(
        out, title, (px1, max(py1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA,
    )

    return out


async def call_gemini(
    image_path: str, api_key: str, prompt: str
) -> list[dict]:
    """Send image + prompt to Gemini with tool declaration, return tool calls."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Read and encode the image
    img_bytes = open(image_path, "rb").read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    ext = image_path.lower().rsplit(".", 1)[-1]
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")

    tool_decl = get_tool_declaration()
    system_prompt = get_system_prompt()

    # Build the tool config
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(**tool_decl),
        ]
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": img_b64}},
                    {"text": prompt},
                ]
            }
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[tool],
        ),
    )

    print(f"Raw response candidates: {len(response.candidates)}")
    for i, cand in enumerate(response.candidates):
        for j, part in enumerate(cand.content.parts):
            if part.function_call:
                print(f"  Candidate {i}, Part {j}: function_call({part.function_call.name})")
                print(f"    args: {dict(part.function_call.args)}")
            elif part.text:
                print(f"  Candidate {i}, Part {j}: text = {part.text[:200]}")

    return parse_tool_calls(response)


def main():
    parser = argparse.ArgumentParser(
        description="PoC 5: Gemini function calling for overlay generation"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to a rectified table image"
    )
    parser.add_argument("--api-key", type=str, required=True, help="Gemini API key")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Can you help me understand the equation y = 7x + b?",
        help="User question for the tutor",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="poc5_output.png",
        help="Output image path (default: poc5_output.png)",
    )
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: could not load image '{args.image}'")
        sys.exit(1)

    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"Prompt: {args.prompt}")
    print()

    # Call Gemini
    import asyncio

    calls = asyncio.run(call_gemini(args.image, args.api_key, args.prompt))

    if not calls:
        print("No tool calls returned. Gemini may have responded with text only.")
        sys.exit(0)

    print(f"\n{len(calls)} tool call(s) returned.\n")

    # Validate and render each call
    result = image.copy()
    for i, call in enumerate(calls):
        print(f"--- Call {i + 1}: {call['name']} ---")
        print(f"  content_type: {call['args'].get('content_type')}")
        print(f"  placement: {call['args'].get('placement')}")
        print(f"  title: {call['args'].get('title')}")
        print(f"  data: {call['args'].get('data')}")

        errors = validate_overlay_call(call)
        if errors:
            print(f"  VALIDATION ERRORS:")
            for err in errors:
                print(f"    - {err}")
            continue

        print("  Validation: OK")
        result = render_overlay_preview(call, result)

    # Save and display
    cv2.imwrite(args.output, result)
    print(f"\nSaved output to {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
