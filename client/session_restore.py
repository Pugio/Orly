"""Session persistence: save and restore session state across restarts."""

import threading


def save_session_state(session_store, overlay_state, scene_order) -> None:
    """Save current session state for later restoration."""
    session_store.save_overlay_state(overlay_state.to_json())
    session_store.save_scene_order(scene_order)


def restore_session_state(session_store, overlay_manager) -> int:
    """Restore saved session state. Returns number of overlays restored.

    Re-renders non-image overlays from saved data. Image overlays are
    restored from disk via session_store.load_image().
    """
    state = session_store.load_overlay_state()
    if not state:
        return 0
    overlays = state.get("overlays", [])
    restored = 0
    for entry in overlays:
        name = entry.get("name", "")
        content_type = entry.get("content_type", "")
        placement = entry.get("placement", [0, 0, 1000, 1000])
        title = entry.get("title", "")
        data = entry.get("data", {})

        if content_type == "image":
            img = session_store.load_image(name)
            if img is not None:
                overlay_manager.scenes[title] = img
                overlay_manager._show_overlay(img, placement, content_type)
                restored += 1
        else:
            overlay_manager.handle_tool_result("project_overlay", {
                "content_type": content_type,
                "placement": placement,
                "title": title,
                "data": data,
            })
            restored += 1

    # Restore scene order
    scene_order = session_store.load_scene_order()
    if scene_order:
        overlay_manager._scene_order = scene_order

    return restored


class DebouncedSaver:
    """Saves session state after a quiet period (default 2s).

    Each call to trigger() resets the timer. The save_fn is only called
    once the timer expires without another trigger.
    """

    def __init__(self, save_fn, delay=2.0):
        self._save_fn = save_fn
        self._delay = delay
        self._timer: threading.Timer | None = None
        self._pending = False
        self._lock = threading.Lock()

    def trigger(self):
        """Request a save. Actual save happens after `delay` seconds of no triggers."""
        with self._lock:
            self._pending = True
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._execute)
            self._timer.daemon = True
            self._timer.start()

    def flush(self):
        """Force immediate save if a trigger is pending."""
        with self._lock:
            if not self._pending:
                return
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending = False
        self._save_fn()

    def _execute(self):
        with self._lock:
            self._pending = False
            self._timer = None
        self._save_fn()
