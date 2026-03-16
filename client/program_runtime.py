"""Mini-program runtime: executes agent-authored Python in a restricted namespace."""

import ast
import logging
import math
import threading
import time as time_mod
from dataclasses import dataclass

import cv2
import numpy as np

from client.paint_canvas import PaintCanvas

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
                 notify_fn, get_frame_fn, audio_player=None):
        """
        Args:
            overlay_state_manager: OverlayStateManager instance
            object_tracker: ObjectTracker instance
            session_store: SessionStore instance
            notify_fn: Callable[[str], None] — sends notification to backend
            get_frame_fn: Callable[[], np.ndarray | None] — gets latest camera frame
            audio_player: Optional AudioPlayer instance for play_tone.
        """
        self._osm = overlay_state_manager
        self._tracker = object_tracker
        self._session = session_store
        self._notify_fn = notify_fn
        self._get_frame_fn = get_frame_fn
        self._audio_player = audio_player
        self._frame_callbacks: list = []
        self._frame_callbacks_lock = threading.Lock()
        self._paint_canvases: list[PaintCanvas] = []
        self._stop_event = threading.Event()
        self._log_messages: list[str] = []

    # --- Overlay Control ---
    def place_overlay(self, name, content_type, placement, data):
        """Place or update a named overlay on the table."""
        om = self._osm._om
        overlay_img = om.render_overlay(content_type, placement, name, data)
        # Orient for the human's viewing angle. overlay_state.add stores
        # the oriented image and uses it for recomposite.
        oriented = om.transform.orient_overlay(overlay_img)
        self._osm.add(name, content_type, placement, name, data, oriented)

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

    # --- Paint Canvas ---
    def create_canvas(self) -> PaintCanvas:
        """Create a paint canvas for direct pixel drawing.

        The canvas is sized to the projector/screen resolution.
        Paint canvases are composited onto the display each frame.
        Black pixels are transparent (projector paradigm).

        Returns a PaintCanvas object with drawing methods.
        """
        om = self._osm._om
        canvas = PaintCanvas(om.proj_width, om.proj_height)
        with self._frame_callbacks_lock:
            self._paint_canvases.append(canvas)
        return canvas

    # --- Color Analysis ---
    def get_dominant_color(self, region=None):
        """Get the dominant BGR color from the camera frame.

        Args:
            region: Optional (y, x, h, w) in pixel coords. If None, uses full frame.

        Returns (b, g, r) tuple of the dominant color, or None if no frame.
        """
        frame = self._get_frame_fn()
        if frame is None:
            return None
        if region is not None:
            y, x, h, w = region
            roi = frame[y:y+h, x:x+w]
        else:
            roi = frame

        if roi.size == 0:
            return None

        # Use k-means with k=3 to find dominant color (ignoring dark background)
        pixels = roi.reshape(-1, 3).astype(np.float32)
        # Filter out very dark pixels (likely background)
        bright_mask = np.any(pixels > 30, axis=1)
        bright_pixels = pixels[bright_mask]
        if len(bright_pixels) < 10:
            # Fallback: use mean of all pixels
            mean = np.mean(pixels, axis=0).astype(int)
            return tuple(int(c) for c in mean)

        k = min(3, len(bright_pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(bright_pixels, k, None, criteria, 3,
                                         cv2.KMEANS_PP_CENTERS)
        # Pick the cluster with the most pixels
        counts = np.bincount(labels.flatten(), minlength=k)
        dominant_idx = np.argmax(counts)
        color = centers[dominant_idx].astype(int)
        return tuple(int(c) for c in color)

    def get_object_size(self, name):
        """Get the size of a tracked object in normalised 0-1000 coords.

        Returns (height, width) or None if object not found/not visible.
        """
        obj = self._tracker.get_object(name)
        if obj is None or not obj.visible:
            return None
        y, x, h, w = obj.bbox
        fh, fw = self._tracker.frame_size
        nh = (h / fh) * 1000.0 if fh > 0 else 0
        nw = (w / fw) * 1000.0 if fw > 0 else 0
        return (round(nh, 1), round(nw, 1))

    def capture_baseline(self, region=None, settle_time=0.5):
        """Capture a baseline frame (or region) for change detection.

        Blanks the entire projection (overlay canvas + paint canvases) so
        the camera sees only the physical scene. Waits settle_time for the
        projector to go dark and the camera to capture a clean frame.

        Args:
            region: Optional (y, x, h, w) in pixel coords. If given, returns
                    only the grayscale ROI. If None, returns full grayscale frame.
            settle_time: Seconds to wait after blanking projection.

        Returns grayscale numpy array, or None if no frame.
        """
        om = self._osm._om

        # Hide paint canvases
        with self._frame_callbacks_lock:
            for c in self._paint_canvases:
                c._was_visible = c.visible
                c.visible = False

        # Blank the overlay manager canvas (stash current state)
        om.request_refresh()

        # Wait for projector to go dark + camera to capture clean frame
        time_mod.sleep(settle_time)

        frame = self._get_frame_fn()

        # Restore everything
        om.complete_refresh()
        with self._frame_callbacks_lock:
            for c in self._paint_canvases:
                c.visible = getattr(c, '_was_visible', True)

        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if region is not None:
            y, x, h, w = region
            gray = gray[y:y+h, x:x+w]
        return gray

    def wait_for_object_in_region(self, region, timeout=30.0, check_interval=0.3,
                                   diff_threshold=30, change_ratio=0.10,
                                   baseline=None):
        """Wait for a new object to appear in a region via change detection.

        Compares each frame against a baseline. Detects when a significant
        portion of the region has changed — meaning something new was placed
        there. This avoids false triggers from projected content or table texture.

        Args:
            region: (y, x, h, w) in pixel coords — area to watch.
            timeout: Max seconds to wait.
            check_interval: How often to check (seconds).
            diff_threshold: Per-pixel intensity change to count as "different".
            change_ratio: Fraction of pixels that must change (0.0–1.0).
            baseline: Pre-captured baseline (grayscale ROI). If None, captures one
                      automatically (briefly hides overlays).

        Returns True if an object was detected, False if timed out.
        """
        if baseline is None:
            baseline = self.capture_baseline(region, settle_time=0.5)
        if baseline is None:
            return False  # no camera

        start = time_mod.time()
        while not self._stop_event.is_set():
            if time_mod.time() - start > timeout:
                return False
            frame = self._get_frame_fn()
            if frame is not None:
                y, x, h, w = region
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(roi_gray, baseline)
                changed = np.count_nonzero(diff > diff_threshold)
                ratio = changed / max(1, diff.size)
                if ratio > change_ratio:
                    return True
            time_mod.sleep(check_interval)
        return False

    def wait_for_hands_clear(self, region, timeout=15.0, stable_time=1.0,
                              check_interval=0.2):
        """Wait for hands/large motion to clear from a region.

        Detects when the region has been stable (low frame-to-frame diff)
        for at least stable_time seconds. Used after "place object here"
        to wait for the user to remove their hands.

        Args:
            region: (y, x, h, w) in pixel coords.
            timeout: Max seconds to wait.
            stable_time: How long the region must be still.
            check_interval: How often to check.

        Returns True if hands cleared, False if timed out.
        """
        start = time_mod.time()
        prev_roi = None
        stable_since = None

        while not self._stop_event.is_set():
            if time_mod.time() - start > timeout:
                return False
            frame = self._get_frame_fn()
            if frame is not None:
                y, x, h, w = region
                roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                if prev_roi is not None:
                    diff = np.mean(np.abs(roi.astype(float) - prev_roi.astype(float)))
                    if diff < 5.0:  # very stable
                        if stable_since is None:
                            stable_since = time_mod.time()
                        elif time_mod.time() - stable_since >= stable_time:
                            return True
                    else:
                        stable_since = None
                prev_roi = roi.copy()
            time_mod.sleep(check_interval)
        return False

    def init_color_tracking(self, name, region):
        """Initialize color tracking from a camera region.

        Blanks the entire projection so the camera captures only the physical
        object (not projected overlays). Extracts the object's color histogram
        from the specified region and starts CamShift tracking.

        Args:
            name: Name for the tracked object.
            region: (y, x, h, w) in pixel coords.

        Returns the dominant BGR color of the object, or None on failure.
        """
        om = self._osm._om

        # Blank everything so camera sees only the physical scene
        with self._frame_callbacks_lock:
            for c in self._paint_canvases:
                c._was_visible = c.visible
                c.visible = False
        om.request_refresh()

        time_mod.sleep(0.5)  # let projector go dark + camera settle

        frame = self._get_frame_fn()

        # Restore
        om.complete_refresh()
        with self._frame_callbacks_lock:
            for c in self._paint_canvases:
                c.visible = getattr(c, '_was_visible', True)

        if frame is None:
            return None
        self._tracker.track_color(name, frame, region)
        color = self.get_dominant_color(region)
        return color

    # --- Sound ---
    def play_tone(self, frequency, duration, volume=0.5):
        """Play a sine wave tone through the speaker.

        Args:
            frequency: Tone frequency in Hz.
            duration: Duration in seconds.
            volume: Volume 0.0-1.0.
        """
        self.log(f"play_tone({frequency}Hz, {duration}s)")
        try:
            sample_rate = 16000
            n_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, n_samples, endpoint=False)
            wave = (volume * 32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
            pcm_data = wave.tobytes()

            if self._audio_player is not None:
                self._audio_player.play(pcm_data)
            else:
                self.log("No audio player available for play_tone")
        except Exception as e:
            self.log(f"play_tone error: {e}")

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
        """Called each video frame. Updates object tracker, dispatches to programs,
        and composites paint canvases onto the overlay manager canvas."""
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

        # Composite paint canvases onto overlay manager canvas
        self._composite_paint_canvases()

    def _composite_paint_canvases(self):
        """Composite all active paint canvases onto the overlay manager canvas."""
        for prog in self._programs.values():
            if prog.state not in ("running",):
                continue
            with prog.api._frame_callbacks_lock:
                canvases = list(prog.api._paint_canvases)
            for canvas in canvases:
                if canvas.has_content() and canvas.visible:
                    om = prog.api._osm._om
                    canvas.composite_onto(om.canvas)
