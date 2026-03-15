import json
import os
import re
import time

import cv2
import numpy as np


class SessionStore:
    """File-backed session storage for images, programs, and state."""

    def __init__(self, session_dir: str = "session"):
        self.session_dir = session_dir
        self.images_dir = os.path.join(session_dir, "images")
        self.programs_dir = os.path.join(session_dir, "programs")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.programs_dir, exist_ok=True)
        self._created_at = time.time()

    def save_image(self, name: str, image: np.ndarray) -> str:
        """Save a BGR image as PNG. Returns the file path."""
        safe = self.sanitize_name(name)
        path = os.path.join(self.images_dir, f"{safe}.png")
        cv2.imwrite(path, image)
        return path

    def load_image(self, name: str) -> np.ndarray | None:
        """Load a previously saved image by name."""
        safe = self.sanitize_name(name)
        path = os.path.join(self.images_dir, f"{safe}.png")
        if not os.path.exists(path):
            return None
        return cv2.imread(path)

    def list_images(self) -> list[str]:
        """List all saved image names (without extension)."""
        if not os.path.exists(self.images_dir):
            return []
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.images_dir)
            if f.endswith(".png")
        )

    def delete_image(self, name: str) -> bool:
        """Delete a saved image. Returns True if it existed."""
        safe = self.sanitize_name(name)
        path = os.path.join(self.images_dir, f"{safe}.png")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def save_program(self, name: str, code: str) -> str:
        """Save program source code. Returns file path."""
        safe = self.sanitize_name(name)
        path = os.path.join(self.programs_dir, f"{safe}.py")
        with open(path, "w") as f:
            f.write(code)
        return path

    def load_program(self, name: str) -> str | None:
        """Load program source code by name."""
        safe = self.sanitize_name(name)
        path = os.path.join(self.programs_dir, f"{safe}.py")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return f.read()

    def list_programs(self) -> list[str]:
        """List all saved program names (without extension)."""
        if not os.path.exists(self.programs_dir):
            return []
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.programs_dir)
            if f.endswith(".py")
        )

    def _state_path(self) -> str:
        return os.path.join(self.session_dir, "state.json")

    def _read_state_file(self) -> dict:
        path = self._state_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _write_state_file(self, data: dict) -> None:
        with open(self._state_path(), "w") as f:
            json.dump(data, f, indent=2)

    def save_overlay_state(self, state: dict) -> None:
        """Save overlay state dict to session/state.json.

        Replaces existing overlay keys (preserves scene_order).
        """
        existing = self._read_state_file()
        # Preserve scene_order if present, replace everything else
        scene_order = existing.get("scene_order")
        new_data = dict(state)
        if scene_order is not None:
            new_data["scene_order"] = scene_order
        self._write_state_file(new_data)

    def load_overlay_state(self) -> dict:
        """Load overlay state from session/state.json. Returns {} if missing."""
        data = self._read_state_file()
        if not data:
            return {}
        # Return everything except scene_order
        result = {k: v for k, v in data.items() if k != "scene_order"}
        return result if result else {}

    def save_scene_order(self, order: list[str]) -> None:
        """Save scene order to session/state.json (merges with existing)."""
        existing = self._read_state_file()
        existing["scene_order"] = order
        self._write_state_file(existing)

    def load_scene_order(self) -> list[str]:
        """Load scene order from session/state.json. Returns [] if missing."""
        data = self._read_state_file()
        return data.get("scene_order", [])

    def get_manifest(self) -> dict:
        """Return a manifest of all session assets."""
        return {
            "images": self.list_images(),
            "programs": self.list_programs(),
            "created_at": self._created_at,
            "session_dir": self.session_dir,
        }

    def clear(self) -> None:
        """Delete all session data."""
        import shutil

        if os.path.exists(self.session_dir):
            shutil.rmtree(self.session_dir)
        # Recreate empty dirs
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.programs_dir, exist_ok=True)

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Convert a display name to a filesystem-safe filename.

        - Lowercases
        - Replaces spaces and special chars with hyphens
        - Collapses multiple hyphens
        - Strips leading/trailing hyphens
        - Truncates to 100 chars
        - Returns 'unnamed' if result is empty
        """
        safe = name.lower()
        safe = re.sub(r"[^a-z0-9]+", "-", safe)
        safe = re.sub(r"-+", "-", safe)
        safe = safe.strip("-")
        safe = safe[:100]
        return safe or "unnamed"
