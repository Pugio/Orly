"""Generate images via Gemini and render for projector overlay.

Uses gemini-3.1-flash-image-preview to generate images from text prompts.
Falls back to a text annotation if generation fails.
"""

import cv2
import numpy as np

from client.genai_utils import get_genai_client

ENHANCE_PREFIX = (
    "The student has drawn something on their paper. Enhance and build upon "
    "their drawing — do NOT replace it. Closely preserve their original lines, "
    "shapes, and intent. Add detail, color, and refinement while keeping their "
    "vision intact. "
)


def render_loading(prompt: str, width: int, height: int) -> np.ndarray:
    """Render a static loading placeholder while image generates.

    Shows a border and "Generating..." text on black background.
    For animated loading, use render_loading_frame() instead.

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    return render_loading_frame(0.0, width, height, prompt)


def render_loading_frame(
    elapsed: float, width: int, height: int, prompt: str
) -> np.ndarray:
    """Render one frame of the animated loading indicator.

    Shows a cyan border, "Generating..." text, and a water-fill effect
    that rises from the bottom over ~60 seconds.

    Args:
        elapsed: Seconds since generation started.
        width: Frame width in pixels.
        height: Frame height in pixels.
        prompt: The generation prompt (displayed truncated).

    Returns:
        BGR numpy array (uint8) with shape (height, width, 3).
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    thickness = max(2, min(width, height) // 60)
    inset = thickness + 2  # inner edge of the border

    # Water fill: rises from bottom over 60 seconds
    fill_duration = 60.0
    fill_frac = min(elapsed / fill_duration, 1.0) if elapsed > 0 else 0.0
    fill_height = int((height - 2 * inset) * fill_frac)
    if fill_height > 0:
        # Dark teal water — visible on projector but doesn't overpower text
        water_color = (160, 100, 20)  # BGR: dark cyan/teal
        fill_top = height - inset - fill_height
        img[fill_top : height - inset, inset : width - inset] = water_color

    # Cyan border
    border_color = (255, 255, 0)  # cyan in BGR
    cv2.rectangle(img, (thickness, thickness),
                  (width - thickness, height - thickness),
                  border_color, thickness)

    # "Generating..." text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.8, min(width, height) / 400)
    text = "Generating..."
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    tx = (width - tw) // 2
    ty = (height + th) // 2
    cv2.putText(img, text, (tx, ty), font, scale, border_color, 2, cv2.LINE_AA)

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
        "Generate a simple children's image with a clear primary focus, large, "
        "easy to parse elements, and not too much visual noise. "
        "Use a style suitable for a children's book or classroom poster. The image: {prompt}"
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
        client = get_genai_client()
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
