"""Renderer registry — auto-discovers SPEC dicts from renderer modules."""

import importlib
import pkgutil
from pathlib import Path

_REGISTRY: dict[str, dict] = {}


def _discover():
    """Import all modules in client/renderer/ and collect SPEC dicts."""
    package_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name.startswith("_") or info.name == "registry":
            continue
        mod = importlib.import_module(f"client.renderer.{info.name}")
        spec = getattr(mod, "SPEC", None)
        if spec and "name" in spec and "render" in spec:
            _REGISTRY[spec["name"]] = spec


def get(name: str) -> dict | None:
    """Get a renderer spec by content_type name."""
    if not _REGISTRY:
        _discover()
    return _REGISTRY.get(name)


def all_specs() -> list[dict]:
    """Get all registered specs (sorted by name)."""
    if not _REGISTRY:
        _discover()
    return sorted(_REGISTRY.values(), key=lambda s: s["name"])


def valid_types() -> set[str]:
    """Get the set of all registered content_type names."""
    if not _REGISTRY:
        _discover()
    return set(_REGISTRY.keys())


def data_format_docs() -> str:
    """Generate the data format documentation for the project_overlay docstring."""
    lines = []
    for spec in all_specs():
        lines.append(f'For "{spec["name"]}": {spec["data_format"]}')
    return "\n              ".join(lines)


def prompt_overlay_docs() -> str:
    """Generate the OVERLAY CONTENT TYPES section for the system prompt."""
    lines = []
    for spec in all_specs():
        hint = spec.get("prompt_hint", "")
        lines.append(f'- "{spec["name"]}": {spec["description"]} {hint}')
    return "\n".join(lines)
