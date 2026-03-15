"""Mini-program runtime: executes agent-authored Python in a restricted namespace."""

import ast
import logging
import math
import threading
import time as time_mod
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code validation
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "urllib", "requests", "http",
    "importlib", "ctypes", "signal",
}

FORBIDDEN_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open",
    "breakpoint", "exit", "quit",
}


def validate_code(code: str) -> tuple[bool, str]:
    """Validate code before execution.

    Returns (valid, error_message).
    """
    # Check syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Walk AST for forbidden constructs
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_BUILTINS:
                    return False, f"Forbidden builtin: {node.func.id}"

    return True, ""


# ---------------------------------------------------------------------------
# TableAPI
# ---------------------------------------------------------------------------


class TableAPI:
    """API surface available to mini-programs via the ``table`` variable."""

    def __init__(self, overlay_state_manager, object_tracker, session_store,
                 notify_fn, get_frame_fn):
        """
        Args:
            overlay_state_manager: OverlayStateManager instance
            object_tracker: ObjectTracker instance
            session_store: SessionStore instance
            notify_fn: Callable[[str], None] — sends notification to backend
            get_frame_fn: Callable[[], np.ndarray | None] — gets latest camera frame
        """
        self._osm = overlay_state_manager
        self._tracker = object_tracker
        self._session = session_store
        self._notify_fn = notify_fn
        self._get_frame_fn = get_frame_fn
        self._frame_callbacks: list = []
        self._frame_callbacks_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._log_messages: list[str] = []

    # --- Overlay Control ---
    def place_overlay(self, name, content_type, placement, data):
        """Place or update a named overlay on the table."""
        om = self._osm._om
        overlay_img = om.render_overlay(content_type, placement, name, data)
        self._osm.add(name, content_type, placement, name, data, overlay_img)

    def remove_overlay(self, name):
        """Remove a named overlay."""
        return self._osm.remove(name)

    def clear_overlays(self):
        """Remove all overlays."""
        self._osm.clear()

    def get_overlay_state(self):
        """Get JSON description of current overlays."""
        return self._osm.to_json()

    # --- Camera ---
    def get_frame(self):
        """Get the latest rectified camera frame (BGR numpy array)."""
        return self._get_frame_fn()

    # --- Object Tracking ---
    def track_color(self, name, region, hsv_range=None):
        """Start tracking a colored object."""
        frame = self._get_frame_fn()
        if frame is not None:
            self._tracker.track_color(name, frame, region, hsv_range)

    def track_template(self, name, template):
        """Start tracking an object by template image."""
        self._tracker.track_template(name, template)

    def get_tracked(self, name):
        """Get position of a tracked object."""
        obj = self._tracker.get_object(name)
        if obj is None:
            return None
        return {
            "name": obj.name,
            "center": obj.center,
            "bbox": obj.bbox,
            "visible": obj.visible,
            "method": obj.method,
        }

    def get_all_tracked(self):
        """Get all tracked objects."""
        result = {}
        for name, obj in self._tracker.get_all().items():
            result[name] = {
                "name": obj.name,
                "center": obj.center,
                "bbox": obj.bbox,
                "visible": obj.visible,
            }
        return result

    def add_zone(self, name, bbox, on_enter=None, on_exit=None):
        """Add a trigger zone."""
        from client.object_tracker import Zone
        self._tracker.add_zone(Zone(name=name, bbox=bbox,
                                     on_enter=on_enter, on_exit=on_exit))

    def remove_zone(self, name):
        """Remove a trigger zone."""
        self._tracker.remove_zone(name)

    # --- Sound ---
    def play_tone(self, frequency, duration):
        """Play a sine wave tone. Currently a stub — logs the request."""
        self.log(f"play_tone({frequency}Hz, {duration}s)")

    # --- Communication ---
    def notify(self, message):
        """Send a text notification back to the Gemini agent."""
        self._notify_fn(message)

    def log(self, message):
        """Log a debug message."""
        self._log_messages.append(str(message))
        logger.info("[Program] %s", message)

    # --- Session ---
    def save_image(self, name, image):
        """Save an image to the session store."""
        self._session.save_image(name, image)

    def load_image(self, name):
        """Load an image from the session store."""
        return self._session.load_image(name)

    # --- Frame Callbacks ---
    def on_frame(self, callback):
        """Register a callback to be called each frame.

        callback(frame: np.ndarray) -> None
        """
        with self._frame_callbacks_lock:
            self._frame_callbacks.append(callback)

    # --- Lifecycle ---
    def stop(self):
        """Signal this program to stop."""
        self._stop_event.set()

    @property
    def stopped(self):
        return self._stop_event.is_set()


# ---------------------------------------------------------------------------
# ProgramRuntime
# ---------------------------------------------------------------------------


@dataclass
class ProgramStatus:
    name: str
    description: str
    state: str           # "running", "stopped", "error"
    started_at: float
    error: str | None = None
    frame_count: int = 0


class _RunningProgram:
    def __init__(self, name, description, api, thread):
        self.name = name
        self.description = description
        self.api = api
        self.thread = thread
        self.started_at = time_mod.time()
        self.error = None
        self.frame_count = 0

    @property
    def state(self):
        if self.error:
            return "error"
        if self.api.stopped:
            if self.thread.is_alive():
                return "stopping"
            return "stopped"
        if self.thread.is_alive():
            return "running"
        # Thread finished but has registered frame callbacks — still active.
        with self.api._frame_callbacks_lock:
            has_callbacks = bool(self.api._frame_callbacks)
        if has_callbacks:
            return "running"
        return "stopped"

    def to_status(self):
        return ProgramStatus(
            name=self.name,
            description=self.description,
            state=self.state,
            started_at=self.started_at,
            error=self.error,
            frame_count=self.frame_count,
        )


class ProgramRuntime:
    """Executes mini-programs in a restricted namespace."""

    def __init__(self, table_api_factory):
        """
        Args:
            table_api_factory: Callable that creates a new TableAPI for each program.
                              This allows each program to have its own stop event etc.
                              Signature: () -> TableAPI
        """
        self._api_factory = table_api_factory
        self._programs: dict[str, _RunningProgram] = {}
        self._latest_frame: np.ndarray | None = None  # updated by video_loop
        self._object_tracker = None  # set externally to share tracker across programs

    def run(self, name, code, description=""):
        """Parse, validate, and start a program.

        Returns ProgramStatus.
        """
        # Stop existing program with same name
        if name in self._programs:
            self.stop(name)

        # Validate
        valid, error = validate_code(code)
        if not valid:
            status = ProgramStatus(
                name=name, description=description,
                state="error", started_at=time_mod.time(), error=error,
            )
            return status

        # Create API for this program
        api = self._api_factory()

        # Safe __import__ that blocks forbidden modules at runtime
        _real_import = __import__

        def _safe_import(mod_name, *args, **kwargs):
            top = mod_name.split(".")[0]
            if top in FORBIDDEN_IMPORTS:
                raise ImportError(f"Forbidden import: {mod_name}")
            return _real_import(mod_name, *args, **kwargs)

        # Build restricted namespace
        namespace = {
            "__name__": f"program_{name}",
            "table": api,
            "np": np,
            "cv2": cv2,
            "math": math,
            "time": time_mod,
            "print": api.log,
            "__builtins__": {
                "True": True, "False": False, "None": None,
                # Types
                "int": int, "float": float, "str": str, "bool": bool,
                "list": list, "dict": dict, "tuple": tuple, "set": set,
                "bytes": bytes, "bytearray": bytearray, "frozenset": frozenset,
                "object": object, "property": property, "super": super,
                "__build_class__": __build_class__,
                # Iteration & sequences
                "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "iter": iter, "next": next,
                "any": any, "all": all,
                "sorted": sorted, "reversed": reversed,
                # Math & comparison
                "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
                "pow": pow, "divmod": divmod,
                # Inspection
                "isinstance": isinstance, "issubclass": issubclass,
                "type": type, "callable": callable,
                "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
                "id": id, "hash": hash, "repr": repr, "chr": chr, "ord": ord,
                # I/O (redirected)
                "print": api.log,
                # Import (restricted)
                "__import__": _safe_import,
                # Exceptions
                "Exception": Exception, "RuntimeError": RuntimeError,
                "ValueError": ValueError, "TypeError": TypeError,
                "KeyError": KeyError, "IndexError": IndexError,
                "AttributeError": AttributeError, "NameError": NameError,
                "StopIteration": StopIteration, "ZeroDivisionError": ZeroDivisionError,
                "NotImplementedError": NotImplementedError, "OSError": OSError,
            },
        }

        def _run_code():
            try:
                exec(code, namespace)
            except Exception as e:
                prog.error = f"{type(e).__name__}: {e}"
                logger.error("Program '%s' error: %s", name, prog.error)

        thread = threading.Thread(target=_run_code, daemon=True, name=f"program-{name}")
        prog = _RunningProgram(name, description, api, thread)
        self._programs[name] = prog

        # Save program code to session
        if api._session:
            api._session.save_program(name, code)

        thread.start()
        return prog.to_status()

    def stop(self, name):
        """Stop a running program. Returns True if found."""
        prog = self._programs.get(name)
        if not prog:
            return False
        prog.api.stop()
        prog.thread.join(timeout=2)
        return True

    def stop_all(self):
        """Stop all running programs."""
        for name in list(self._programs.keys()):
            self.stop(name)

    def get_status(self, name):
        """Get status of a program."""
        prog = self._programs.get(name)
        if not prog:
            return None
        return prog.to_status()

    def list_programs(self):
        """List all programs."""
        return [p.to_status() for p in self._programs.values()]

    def process_frame(self, frame):
        """Called each video frame. Updates object tracker, then dispatches to programs."""
        # Update shared latest frame for table.get_frame().
        self._latest_frame = frame

        # Update object tracker so positions and zones are current.
        if self._object_tracker is not None:
            self._object_tracker.update(frame)

        for prog in self._programs.values():
            if prog.state != "running":
                continue
            with prog.api._frame_callbacks_lock:
                callbacks = list(prog.api._frame_callbacks)
            for cb in callbacks:
                try:
                    cb(frame)
                    prog.frame_count += 1
                except Exception as e:
                    prog.error = f"on_frame error: {type(e).__name__}: {e}"
                    logger.error("Program '%s' frame callback error: %s", prog.name, e)
                    break
