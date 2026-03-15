"""Overlay projection tool for the Lumi tutor agent."""


def project_overlay(
    content_type: str,
    placement: list[float],
    title: str,
    data: dict,
) -> dict:
    """Project a visual overlay onto the student's work surface via projector.

    Args:
        content_type: Type of visual — "graph", "diagram", "annotation", or "highlight".
        placement: Where to place it on the table, [ymin, xmin, ymax, xmax] normalised 0-1000.
                   Choose empty space near relevant content. Never overlap existing work.
        title: Label for the overlay.
        data: Content-specific data. For "graph": {"expression": "x**2 - 3*x + 2",
              "x_range": [-5, 5], "y_range": [-5, 10]}. For "annotation": {"text": "..."}.
              For "highlight": {"color": "#00ffff", "target": [ymin, xmin, ymax, xmax]}.

    Returns:
        dict with status of the projection.
    """
    _VALID_TYPES = {"graph", "diagram", "annotation", "highlight"}

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
