"""Generate images via Gemini and render for projector overlay.

Uses gemini-3.1-flash-image-preview to generate images from text prompts.
Falls back to a text annotation if generation fails.
"""

import os

import cv2
import numpy as np

ENHANCE_PREFIX = (
    "The student has drawn something on their paper. Enhance and build upon "
    "their drawing — do NOT replace it. Closely preserve their original lines, "
    "shapes, and intent. Add detail, color, and refinement while keeping their "
    "vision intact. "
)


_genai_client_cache = {}


def _get_genai_client():
    """Get or create a cached google.genai client.

    Checks env vars first, then falls back to `llm keys get gemini`.
    """
    if "client" in _genai_client_cache:
        return _genai_client_cache["client"]
    from google import genai
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        import subprocess
        try:
            api_key = subprocess.check_output(
                ["llm", "keys", "get", "gemini"], text=True
            ).strip()
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("No Gemini API key found. Set GOOGLE_API_KEY or run `llm keys set gemini`.")
    client = genai.Client(api_key=api_key)
    _genai_client_cache["client"] = client
    return client


def render_loading(prompt: str, width: int, height: int) -> np.ndarray:
    """Render a loading/spinner placeholder while image generates.

    Shows a pulsing border and "Generating..." text on black background.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw animated-looking border (dashed cyan rectangle)
    color = (255, 255, 0)  # cyan in BGR
    thickness = max(2, min(width, height) // 60)
    cv2.rectangle(img, (thickness, thickness),
                  (width - thickness, height - thickness), color, thickness)

    # "Generating..." text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, min(width, height) / 400)
    text = "Generating..."
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    tx = (width - tw) // 2
    ty = (height + th) // 2
    cv2.putText(img, text, (tx, ty), font, scale, color, 2, cv2.LINE_AA)

    # Show truncated prompt below
    prompt_display = prompt[:50] + "..." if len(prompt) > 50 else prompt
    scale_small = scale * 0.6
    (tw2, th2), _ = cv2.getTextSize(prompt_display, font, scale_small, 1)
    tx2 = (width - tw2) // 2
    ty2 = ty + th + int(th2 * 1.5)
    if ty2 < height - 10:
        cv2.putText(img, prompt_display, (tx2, ty2), font, scale_small,
                    (180, 180, 0), 1, cv2.LINE_AA)

    return img


_STYLE_PROMPTS = {
    "default": "{prompt}",
    "technical": (
        "Generate a technical diagram with a BLACK background and bright, "
        "vivid colors. This will be projected onto a table, so use cyan, "
        "yellow, green, magenta — no white or dark colors that won't show. "
        "Keep it clean, labeled, and educational. The diagram: {prompt}"
    ),
    "creative": (
        "Generate a vivid, colorful illustration. Make it rich, detailed, "
        "and appealing to a child. Use a style suitable for a children's "
        "book or classroom poster. The image: {prompt}"
    ),
}


def render_image(
    prompt: str,
    width: int,
    height: int,
    reference_frame: np.ndarray | None = None,
    style: str = "default",
    enhance: bool = False,
) -> np.ndarray:
    """Generate an image from a text prompt and return it sized for overlay.

    This is a blocking call that may take several seconds. Callers should
    show a loading placeholder first (see render_loading).

    Args:
        prompt: Text description of the image to generate.
        width: Output image width in pixels.
        height: Output image height in pixels.
        reference_frame: Optional BGR numpy array of the current camera view.
                         When provided, sent as a reference image so the model
                         can see what's on the table.
        style: "default" passes the prompt as-is, "technical" for diagrams
               on black bg, "creative" for rich illustrations (default: "default").

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
        On failure, returns a fallback annotation with the prompt text.
    """
    try:
        client = _get_genai_client()
        from google.genai import types

        template = _STYLE_PROMPTS.get(style, _STYLE_PROMPTS["default"])
        text_prompt = template.format(prompt=prompt)

        if enhance and style == "default":
            text_prompt = ENHANCE_PREFIX + text_prompt

        if reference_frame is not None:
            # Encode reference frame as JPEG for the API.
            _, ref_jpeg = cv2.imencode(".jpg", reference_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            contents = [
                types.Part.from_bytes(data=ref_jpeg.tobytes(), mime_type="image/jpeg"),
                text_prompt,
            ]
        else:
            contents = text_prompt

        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        # Safely extract image from response
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return _fallback("[No candidates in response]", width, height)

        content = getattr(candidates[0], "content", None)
        if content is None:
            return _fallback("[No content in response]", width, height)

        parts = getattr(content, "parts", None)
        if not parts:
            return _fallback("[No parts in response]", width, height)

        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                img_bytes = np.frombuffer(inline.data, dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.resize(img, (width, height),
                                      interpolation=cv2.INTER_AREA)

        return _fallback(f"[No image in response] {prompt}", width, height)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error("Image generation error: %s", e)
        return _fallback(f"[Image gen failed] {prompt}", width, height)


def _fallback(text: str, width: int, height: int) -> np.ndarray:
    """Render a text fallback when image generation fails."""
    from client.renderer.annotation import _render_annotation_impl
    return _render_annotation_impl(text, width, height)


def _render_image_registry(data: dict, width: int, height: int, title: str = "") -> np.ndarray:
    """Registry-compatible wrapper. Returns loading placeholder (actual gen is async)."""
    return render_loading(data.get("prompt", title), width, height)


SPEC = {
    "name": "image",
    "description": "AI-generated image projected onto the table.",
    "data_format": (
        '{"prompt": "a labeled unit circle", "style": "technical", '
        '"include_view": true, "reference_previous": false}.'
    ),
    "prompt_hint": (
        'Use when the child asks to generate, draw, or show a picture. '
        'Styles: "default", "technical" (black bg), "creative" (colorful).'
    ),
    "render": _render_image_registry,
}
