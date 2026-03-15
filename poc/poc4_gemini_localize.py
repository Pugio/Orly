"""
PoC 4 — Gemini spatial localization.

Validates that Gemini can accurately return bounding boxes for content
in a rectified table image. Uses the standard (non-streaming) Gemini API.

Usage:
    python poc/poc4_gemini_localize.py --image rectified_output.png --api-key YOUR_KEY
"""

import argparse
import asyncio
import base64
import json
import re
import sys

import cv2
import numpy as np


def build_localization_prompt() -> str:
    """Return the prompt that asks Gemini to identify and localize content on the table."""
    return (
        "You are analyzing a top-down photo of a student's desk/table.\n"
        "Identify every distinct piece of content visible on the table "
        "(e.g., equations, text paragraphs, diagrams, graphs, handwriting, printed text).\n\n"
        "For each item, return a JSON array of objects with these fields:\n"
        '  - "label": a short description of the content (string)\n'
        '  - "box_2d": bounding box as [ymin, xmin, ymax, xmax] '
        "with coordinates normalized to 0-1000\n\n"
        "Rules:\n"
        "- Coordinates are normalized: 0 is top/left edge, 1000 is bottom/right edge.\n"
        "- ymin < ymax and xmin < xmax.\n"
        "- Return ONLY the JSON array, no other text.\n"
        "- If nothing is detected, return an empty array: []\n"
    )


def parse_gemini_response(response_text: str) -> list[dict]:
    """Parse Gemini's response into a list of detected items.

    Each item: {"label": str, "box_2d": [ymin, xmin, ymax, xmax]}
    Handles markdown code fences, extra whitespace, etc.
    Returns empty list if parsing fails.
    """
    if not response_text or not response_text.strip():
        return []

    text = response_text.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:\w*)\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    return parsed


def validate_detections(detections: list[dict]) -> list[dict]:
    """Validate and clean detections.

    - Ensure box_2d has 4 values, all in 0-1000
    - Ensure ymin < ymax, xmin < xmax
    - Ensure label is non-empty
    Returns only valid detections.
    """
    valid = []
    for det in detections:
        # Check label
        label = det.get("label")
        if not label or not isinstance(label, str) or not label.strip():
            continue

        # Check box_2d
        box = det.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue

        # Check all values are numbers in range
        try:
            values = [float(v) for v in box]
        except (TypeError, ValueError):
            continue

        if any(v < 0 or v > 1000 for v in values):
            continue

        ymin, xmin, ymax, xmax = values
        if ymin >= ymax or xmin >= xmax:
            continue

        valid.append(det)

    return valid


# Colors for drawing bounding boxes (BGR)
_COLORS = [
    (0, 255, 0),    # green
    (255, 200, 0),   # cyan-ish
    (0, 200, 255),   # orange
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow
    (255, 100, 100), # light blue
]


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image.

    image is a rectified table view (OpenCV BGR).
    Boxes are drawn in bright colors with labels.
    Returns annotated copy of the image.
    """
    out = image.copy()
    if not detections:
        return out

    h, w = out.shape[:2]

    for i, det in enumerate(detections):
        color = _COLORS[i % len(_COLORS)]
        ymin, xmin, ymax, xmax = det["box_2d"]

        # Convert from 0-1000 normalized coords to pixel coords
        x1 = int(xmin * w / 1000)
        y1 = int(ymin * h / 1000)
        x2 = int(xmax * w / 1000)
        y2 = int(ymax * h / 1000)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = det.get("label", "")
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return out


async def localize_content(image_path: str, api_key: str) -> list[dict]:
    """Send a rectified table image to Gemini and get localized content.

    Returns list of validated detections.
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    # Read and encode the image
    img_bytes = open(image_path, "rb").read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Determine MIME type
    ext = image_path.lower().rsplit(".", 1)[-1]
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")

    prompt = build_localization_prompt()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            {
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": img_b64}},
                    {"text": prompt},
                ]
            }
        ],
    )

    response_text = response.text
    print(f"Raw Gemini response:\n{response_text}\n")

    detections = parse_gemini_response(response_text)
    valid = validate_detections(detections)

    print(f"Parsed {len(detections)} detections, {len(valid)} valid.")
    return valid


def main():
    parser = argparse.ArgumentParser(description="PoC 4: Gemini spatial localization")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to a rectified table image")
    parser.add_argument("--api-key", type=str, required=True,
                        help="Gemini API key")
    parser.add_argument("--output", type=str, default="poc4_output.png",
                        help="Output image path (default: poc4_output.png)")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: could not load image '{args.image}'")
        sys.exit(1)

    # Run localization
    detections = asyncio.run(localize_content(args.image, args.api_key))

    if not detections:
        print("No content detected on the table.")
    else:
        for det in detections:
            print(f"  {det['label']}: {det['box_2d']}")

    # Draw and save/show
    annotated = draw_detections(image, detections)
    cv2.imwrite(args.output, annotated)
    print(f"\nSaved annotated image to {args.output}")

    # Display if possible
    try:
        cv2.imshow("PoC 4: Gemini Localization", annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # No display available


if __name__ == "__main__":
    main()
