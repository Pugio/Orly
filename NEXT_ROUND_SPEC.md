# Next Round: Interactive Programs, Object Tracking & Session Management

## Overview

Transform TableLight from a reactive tool-calling system into a programmable interactive platform. The Gemini agent should be able to write and run mini-programs that respond to the camera feed in real-time, track objects, trigger events, and compose rich interactive experiences — all with much lower latency than a full Gemini round-trip.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Gemini Live Agent                              │
│  - Calls run_program(code=...) tool             │
│  - Receives async notifications via text channel│
│  - Calls get_overlay_state() to inspect canvas  │
└─────────┬──────────────────────────┬────────────┘
          │ tool calls               │ text updates
          ▼                          ▲
┌─────────────────────────────────────────────────┐
│  Backend (Cloud Run)                            │
│  - Validates programs                           │
│  - Forwards to client via WebSocket             │
│  - Relays async notifications to Gemini         │
└─────────┬──────────────────────────┬────────────┘
          │ WebSocket                │ WebSocket
          ▼                          ▲
┌─────────────────────────────────────────────────┐
│  Client (Edge)                                  │
│  ┌──────────────┐  ┌─────────────────────────┐  │
│  │ ProgramRuntime│  │ OverlayStateManager     │  │
│  │ - Sandboxed   │  │ - Named overlays        │  │
│  │ - Frame loop  │  │ - ASCII grid            │  │
│  │ - TableAPI    │  │ - JSON state            │  │
│  └──────┬───────┘  └─────────────────────────┘  │
│         │                                       │
│  ┌──────▼───────┐  ┌─────────────────────────┐  │
│  │ ObjectTracker│  │ SessionStore             │  │
│  │ - Color HSV  │  │ - session/images/        │  │
│  │ - Template   │  │ - session/programs/      │  │
│  │ - Zones      │  │ - Named references       │  │
│  └──────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Component 1: Overlay State Manager

**File:** `client/overlay_state.py`

Extends OverlayManager with named overlay tracking, CRUD operations, and state serialization.

### API

```python
class OverlayEntry:
    """A single named overlay on the canvas."""
    name: str
    content_type: str         # graph, annotation, markdown, highlight, image
    placement: list[float]    # [ymin, xmin, ymax, xmax] in 0-1000
    title: str
    data: dict
    image: np.ndarray         # The rendered BGR image (before placement)
    created_at: float         # time.time()

class OverlayStateManager:
    """Tracks named overlays and provides state queries."""

    def __init__(self, overlay_manager: OverlayManager):
        self._om = overlay_manager
        self._overlays: dict[str, OverlayEntry] = {}  # name → entry

    def add(self, name: str, content_type: str, placement: list[float],
            title: str, data: dict, image: np.ndarray) -> None:
        """Add or replace a named overlay."""

    def remove(self, name: str) -> bool:
        """Remove overlay by name. Returns True if found."""

    def get(self, name: str) -> OverlayEntry | None:
        """Get overlay entry by name."""

    def clear(self) -> None:
        """Remove all overlays."""

    def list_names(self) -> list[str]:
        """Return all overlay names."""

    def to_json(self) -> dict:
        """Serialize current state to JSON-safe dict."""
        # Returns: {"overlays": [...], "count": N, "dimensions": [1000, 1000]}

    def to_ascii(self, width: int = 40, height: int = 20) -> str:
        """Render ASCII grid showing overlay positions."""
        # '.' = empty, first char of name = overlay, '#' = overlap

    def recomposite(self) -> None:
        """Re-render all overlays onto a fresh canvas."""
        # Called after remove() to rebuild without the removed overlay.
```

### New Backend Tool

```python
def get_overlay_state() -> dict:
    """Get the current state of all overlays on the table.

    Returns a JSON description of all active overlays with their names,
    types, positions, and an ASCII grid visualization of the table layout.
    Use this to understand what's currently projected before making changes.

    Returns:
        dict with overlay list, count, and ASCII visualization.
    """
```

### Test Plan

1. **CRUD operations**: add/get/remove/clear, duplicate names overwrite
2. **JSON serialization**: correct structure, all fields present, timestamps
3. **ASCII grid**: empty grid, single overlay, multiple overlays, overlapping overlays, full coverage
4. **Recomposite**: after remove, canvas reflects remaining overlays only
5. **Edge cases**: remove nonexistent, empty state, overlay at boundary coords

---

## Component 2: Session Storage

**File:** `client/session_store.py`

Persists generated images, program code, and overlay state to a `session/` directory for cross-reference.

### API

```python
class SessionStore:
    """File-backed session storage for images, programs, and state."""

    def __init__(self, session_dir: str = "session"):
        self.session_dir = session_dir
        self.images_dir = os.path.join(session_dir, "images")
        self.programs_dir = os.path.join(session_dir, "programs")
        # Creates directories on init

    def save_image(self, name: str, image: np.ndarray) -> str:
        """Save a BGR image. Returns the file path."""
        # Sanitizes name for filesystem safety
        # Saves as PNG for lossless quality

    def load_image(self, name: str) -> np.ndarray | None:
        """Load a previously saved image by name."""

    def list_images(self) -> list[str]:
        """List all saved image names."""

    def delete_image(self, name: str) -> bool:
        """Delete a saved image."""

    def save_program(self, name: str, code: str) -> str:
        """Save program source code. Returns file path."""

    def load_program(self, name: str) -> str | None:
        """Load program source code by name."""

    def list_programs(self) -> list[str]:
        """List all saved program names."""

    def get_manifest(self) -> dict:
        """Return a manifest of all session assets."""
        # {"images": [...], "programs": [...], "created_at": ...}

    def clear(self) -> None:
        """Delete all session data."""

    @staticmethod
    def sanitize_name(name: str) -> str:
        """Convert a display name to a filesystem-safe filename."""
        # Lowercase, replace spaces/special chars with hyphens
        # Truncate to 100 chars
```

### Test Plan

1. **Image CRUD**: save/load/list/delete, PNG format verified
2. **Program CRUD**: save/load/list
3. **Name sanitization**: spaces, unicode, special chars, length limits
4. **Manifest**: correct structure, reflects actual files
5. **Clear**: removes all files and directories
6. **Concurrent access**: thread-safe saves
7. **Edge cases**: load nonexistent, duplicate names overwrite, empty session

---

## Component 3: Object Tracking

**File:** `client/object_tracker.py`

Frame-by-frame object tracking using color histograms and template matching. Reports positions in normalized 0-1000 coordinates.

### API

```python
@dataclass
class TrackedObject:
    """State of a tracked object."""
    name: str
    bbox: tuple[int, int, int, int]  # (y, x, h, w) in pixel coords
    center: tuple[float, float]      # (y, x) in 0-1000 normalized
    visible: bool
    method: str                       # "color" or "template"
    last_seen: float                  # time.time()

@dataclass
class Zone:
    """A named rectangular zone on the table."""
    name: str
    bbox: tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax) in 0-1000
    on_enter: Callable | None = None
    on_exit: Callable | None = None

class ObjectTracker:
    """Track objects across video frames."""

    def __init__(self, frame_size: tuple[int, int] = (768, 768)):
        self._tracked: dict[str, _TrackerState] = {}
        self._zones: dict[str, Zone] = {}
        self._zone_occupancy: dict[str, set[str]] = {}  # zone → set of object names inside
        self.frame_size = frame_size

    def track_color(self, name: str, initial_frame: np.ndarray,
                    region: tuple[int, int, int, int],
                    hsv_range: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        """Start tracking an object by color histogram.

        Args:
            name: Unique name for this tracked object.
            initial_frame: BGR frame containing the object.
            region: (y, x, h, w) bounding box of the object in the initial frame.
            hsv_range: Optional (lower, upper) HSV bounds. If None, auto-computed
                       from the region's dominant colors.
        """

    def track_template(self, name: str, template: np.ndarray) -> None:
        """Start tracking an object by template matching.

        Args:
            name: Unique name for this tracked object.
            template: BGR image of the object to track.
        """

    def remove(self, name: str) -> bool:
        """Stop tracking an object."""

    def update(self, frame: np.ndarray) -> dict[str, TrackedObject]:
        """Process a new frame and update all tracked object positions.

        Returns current state of all tracked objects.
        """

    def get_object(self, name: str) -> TrackedObject | None:
        """Get current state of a tracked object."""

    def get_all(self) -> dict[str, TrackedObject]:
        """Get all tracked objects."""

    def add_zone(self, zone: Zone) -> None:
        """Add a trigger zone."""

    def remove_zone(self, name: str) -> bool:
        """Remove a trigger zone."""

    def _check_zones(self) -> list[tuple[str, str, str]]:
        """Check zone enter/exit events after an update().

        Returns list of (event_type, object_name, zone_name) tuples.
        event_type is "enter" or "exit".
        """

    def _normalize_position(self, pixel_y: int, pixel_x: int) -> tuple[float, float]:
        """Convert pixel coordinates to 0-1000 normalized coords."""

    @staticmethod
    def compute_color_histogram(frame: np.ndarray,
                                 region: tuple[int, int, int, int]) -> np.ndarray:
        """Compute HSV histogram for a region (for color tracking)."""

    @staticmethod
    def match_template(frame: np.ndarray,
                       template: np.ndarray) -> tuple[int, int, float]:
        """Find best template match. Returns (y, x, confidence)."""
```

### Tracking Methods

**Color tracking (CamShift):**
1. Convert region to HSV
2. Compute hue histogram (or use provided HSV range)
3. On each frame: back-project histogram → CamShift to find new position
4. Report lost if tracking confidence drops below threshold

**Template tracking:**
1. Store template image
2. On each frame: cv2.matchTemplate with TM_CCOEFF_NORMED
3. Report position of best match if above confidence threshold
4. Report lost if confidence drops below 0.5

### Test Plan

1. **Color tracking**: synthetic frames with colored rectangle, verify position updates
2. **Template tracking**: synthetic frames with known pattern, verify match
3. **Position normalization**: pixel → 0-1000 coords, edge cases at boundaries
4. **Zone triggers**: object enters zone → enter callback, leaves → exit callback
5. **Multiple objects**: track 3+ objects simultaneously, independent positions
6. **Object lost**: object removed from frame → visible=False
7. **Remove tracker**: stop tracking, verify cleanup
8. **Histogram computation**: verify HSV histogram shape and values
9. **Edge cases**: zero-size region, template larger than frame, empty frame

---

## Component 4: Mini-Program Runtime

**File:** `client/program_runtime.py`

Executes agent-authored Python code in a restricted namespace with access to the TableAPI.

### API

```python
@dataclass
class ProgramStatus:
    """Status of a running program."""
    name: str
    description: str
    state: str           # "running", "stopped", "error"
    started_at: float
    error: str | None
    frame_count: int     # number of frames processed

class TableAPI:
    """API surface available to mini-programs via the `table` variable.

    All methods are synchronous from the program's perspective.
    Coordinates are in 0-1000 normalized space.
    """

    def __init__(self, overlay_state: OverlayStateManager,
                 object_tracker: ObjectTracker,
                 session_store: SessionStore,
                 notify_fn: Callable[[str], None],
                 get_frame_fn: Callable[[], np.ndarray | None]):
        ...

    # --- Overlay Control ---
    def place_overlay(self, name: str, content_type: str,
                      placement: list[float], data: dict) -> None:
        """Place or update a named overlay."""

    def remove_overlay(self, name: str) -> bool:
        """Remove a named overlay."""

    def clear_overlays(self) -> None:
        """Remove all overlays."""

    def get_overlay_state(self) -> dict:
        """Get JSON description of current overlays."""

    # --- Camera ---
    def get_frame(self) -> np.ndarray | None:
        """Get the latest rectified camera frame (BGR numpy array)."""

    # --- Object Tracking ---
    def track_color(self, name: str, region: tuple[int, int, int, int],
                    hsv_range: tuple | None = None) -> None:
        """Start tracking a colored object in the given region."""

    def track_template(self, name: str, template: np.ndarray) -> None:
        """Start tracking an object by template image."""

    def get_tracked(self, name: str) -> dict | None:
        """Get position of a tracked object. Returns dict with center, bbox, visible."""

    def get_all_tracked(self) -> dict:
        """Get all tracked objects."""

    def add_zone(self, name: str, bbox: tuple[float, float, float, float],
                 on_enter: Callable | None = None,
                 on_exit: Callable | None = None) -> None:
        """Add a trigger zone (bbox in 0-1000 coords)."""

    def remove_zone(self, name: str) -> None:
        """Remove a trigger zone."""

    # --- Sound ---
    def play_tone(self, frequency: float, duration: float) -> None:
        """Play a sine wave tone (Hz, seconds)."""

    # --- Communication ---
    def notify(self, message: str) -> None:
        """Send a text notification back to the Gemini agent."""

    def log(self, message: str) -> None:
        """Log a debug message."""

    # --- Session ---
    def save_image(self, name: str, image: np.ndarray) -> None:
        """Save an image to the session store."""

    def load_image(self, name: str) -> np.ndarray | None:
        """Load an image from the session store."""

    # --- Lifecycle ---
    def stop(self) -> None:
        """Stop this program."""

class ProgramRuntime:
    """Executes mini-programs in a restricted namespace."""

    def __init__(self, table_api: TableAPI):
        self._api = table_api
        self._programs: dict[str, _RunningProgram] = {}

    def run(self, name: str, code: str, description: str = "") -> ProgramStatus:
        """Parse, validate, and start a program.

        The code runs in a thread with access to `table` (TableAPI instance),
        `np` (numpy), `cv2` (OpenCV), `math`, and `time`.

        Programs typically define an on_frame(frame) callback:
            def on_frame(frame):
                # process frame, update overlays, etc.
                pass
            table.on_frame(on_frame)  # register frame callback

        Or run a simple one-shot script.
        """

    def stop(self, name: str) -> bool:
        """Stop a running program."""

    def stop_all(self) -> None:
        """Stop all running programs."""

    def get_status(self, name: str) -> ProgramStatus | None:
        """Get status of a program."""

    def list_programs(self) -> list[ProgramStatus]:
        """List all programs (running and stopped)."""

    def process_frame(self, frame: np.ndarray) -> None:
        """Called each video frame. Dispatches to all running programs' on_frame callbacks."""

    @staticmethod
    def validate_code(code: str) -> tuple[bool, str]:
        """Validate code before execution.

        Checks:
        - Valid Python syntax (ast.parse)
        - No dangerous imports (os, sys, subprocess, etc.)
        - No file operations (open, read, write)
        - No network access (socket, urllib, requests)
        - No exec/eval/compile

        Returns (valid, error_message).
        """
```

### Restricted Namespace

Programs execute with access to:
- `table` — TableAPI instance (the main programming surface)
- `np` — numpy
- `cv2` — OpenCV (for image processing only)
- `math` — standard math functions
- `time` — time.time() and time.sleep()
- `print` — redirected to table.log()

Programs do NOT have access to:
- `os`, `sys`, `subprocess` — no system access
- `open`, `read`, `write` — no direct file I/O
- `socket`, `urllib`, `requests` — no network
- `exec`, `eval`, `compile` — no meta-execution
- `__import__` — no dynamic imports

### New Backend Tools

```python
def run_program(name: str, code: str, description: str = "") -> dict:
    """Run a mini-program on the table surface.

    Programs execute on the edge client with access to the `table` API.
    They can track objects, place overlays, play sounds, and respond to
    the camera feed in real-time — much faster than a full agent round-trip.

    The code runs in a restricted Python environment with access to:
    - `table` — the TableAPI (overlays, tracking, sounds, notifications)
    - `np` — numpy for array/math operations
    - `cv2` — OpenCV for image processing
    - `math` — standard math functions
    - `time` — time.time() and time.sleep()

    To respond to camera frames, define an on_frame callback:
        def on_frame(frame):
            # frame is a BGR numpy array
            tracked = table.get_tracked("toy")
            if tracked and tracked["visible"]:
                table.place_overlay("marker", "highlight",
                    tracked["bbox_norm"], {"color": "#00ff00"})
        table.on_frame(on_frame)

    Args:
        name: Unique name for the program (used to stop/reference it).
        code: Python source code to execute.
        description: What the program does (for user/debugging).

    Returns:
        dict with status of the program launch.
    """

def stop_program(name: str) -> dict:
    """Stop a running mini-program.

    Args:
        name: Name of the program to stop.

    Returns:
        dict with status.
    """

def list_programs() -> dict:
    """List all running mini-programs.

    Returns:
        dict with list of program statuses.
    """
```

### Test Plan

1. **Code validation**: valid code passes, syntax errors caught, dangerous imports blocked
2. **Namespace restriction**: can't access os/sys/subprocess/open, can access np/cv2/math/time
3. **Program lifecycle**: run → running, stop → stopped, error → error state
4. **Frame callback**: on_frame registered, called each frame, receives correct frame
5. **TableAPI overlay ops**: place/remove/clear from program code
6. **TableAPI tracking**: track_color/track_template/get_tracked from program code
7. **TableAPI zones**: add_zone with enter/exit callbacks, verify triggers
8. **TableAPI notify**: sends message back through notification channel
9. **Multiple programs**: run 2+ programs, independent lifecycles
10. **Program timeout**: long-running on_frame doesn't block other programs
11. **Error handling**: runtime error in program → error state, other programs unaffected
12. **Stop all**: stops all running programs
13. **Edge cases**: empty code, code with only comments, stop nonexistent program

---

## Component 5: Async Notification Channel

**File:** Extensions to `backend/main.py` and `client/ws_client.py`

Bi-directional text channel for async task completion notifications.

### Protocol Extension

New WebSocket message types:

```python
# Client → Backend (async notification from program or system)
{"type": "notification", "source": "program:instrument-tracker", "text": "Toy placed on piano!"}
{"type": "notification", "source": "image_gen", "text": "Image 'dragon-scene' is ready."}

# Backend → Client (program execution request)
{"type": "run_program", "name": "...", "code": "...", "description": "..."}
{"type": "stop_program", "name": "..."}
```

### Backend Changes

When backend receives a `notification` message from client:
- Forward it to Gemini via `session.send_client_content()` as a system/user message
- Format: `"[NOTIFICATION from {source}]: {text}"`

When Gemini calls `run_program` tool:
- Forward the program to client via `{"type": "run_program", ...}` WebSocket message
- Tool returns immediately with `{"status": "started"}`

### Client Changes

- `TableLightClient` gains `on_run_program(callback)` and `on_stop_program(callback)` callbacks
- `send_notification(source, text)` method added
- Notification callback sends `{"type": "notification", ...}` to backend

### Image Generation Integration

When image generation completes in `_generate_image_async`:
- Send notification: `{"source": "image_gen", "text": "Image '{title}' is ready and displayed."}`
- Gemini receives this and can acknowledge to the user

### Test Plan

1. **Notification round-trip**: client sends notification → backend receives → forwards to Gemini
2. **Program forwarding**: Gemini calls run_program → backend sends to client → client executes
3. **Image gen notification**: image completes → notification sent → correct format
4. **Multiple notifications**: rapid notifications don't get lost
5. **Source formatting**: program notifications include program name
6. **Edge cases**: notification with empty text, very long text, special characters

---

## Component 6: Image Generation Improvements

**File:** Updates to `client/renderer/image.py` and `backend/agent.py`

### Prompt Tuning

When `include_view` is true and the user has drawn something:
- Prompt prefix: "The student has drawn something on their paper. Enhance and build upon their drawing — do NOT replace it. Closely preserve their original lines, shapes, and intent. Add detail, color, and refinement while keeping their vision intact."
- This ensures AI enhancement, not replacement

### Session Integration

- All generated images auto-saved to `SessionStore`
- Image names are the overlay title (sanitized)
- `reference_scene` looks up from SessionStore (not just in-memory)
- Previous session images available after restart

### System Prompt Update

Add to SYSTEM_PROMPT:
```
INTERACTIVE PROGRAMS:
- Use `run_program` to create interactive experiences that run in real-time.
- Programs can track objects, respond to movement, play sounds, and update overlays — all at camera frame rate, much faster than waiting for you to respond.
- Use `get_overlay_state` to see what's currently on the table before placing new overlays.
- Give programs descriptive names so you can reference and manage them.
- Example use cases:
  - Musical instruments: project instrument images, track a toy, play sounds when it touches each instrument
  - Story narration: project scene images, advance through pages
  - Math games: project number targets, track a pointer, score when it hits the right answer
```

### Test Plan

1. **Enhancement prompt**: include_view=true generates correct prompt prefix
2. **Session auto-save**: generated images saved to SessionStore
3. **Session reference**: reference_scene loads from SessionStore
4. **System prompt**: contains interactive program instructions
5. **Named overlays**: all tool calls use name for state tracking

---

## Fun Use Cases (Design Validation)

These use cases validate the API design covers real scenarios:

### 1. Musical Table
```python
# Agent writes this program:
instruments = [
    ("piano", [100, 100, 400, 400]),
    ("drum", [100, 600, 400, 900]),
    ("guitar", [600, 100, 900, 400]),
    ("flute", [600, 600, 900, 900]),
]

# Project instrument images
for name, placement in instruments:
    table.place_overlay(name, "image", placement,
                        {"prompt": f"a {name}, top-down view", "style": "creative"})

# Add zones for each instrument
for name, placement in instruments:
    ymin, xmin, ymax, xmax = placement
    def make_handler(inst_name):
        def on_enter(obj, zone):
            table.play_tone({"piano": 440, "drum": 200, "guitar": 330, "flute": 880}[inst_name], 0.3)
            table.notify(f"Toy touched the {inst_name}!")
        return on_enter
    table.add_zone(name, (ymin, xmin, ymax, xmax), on_enter=make_handler(name))

# Track the toy (user places toy on table first)
frame = table.get_frame()
table.track_color("toy", (400, 400, 100, 100))  # center region

def on_frame(frame):
    tracked = table.get_tracked("toy")
    if tracked and tracked["visible"]:
        y, x = tracked["center"]
        table.place_overlay("toy-marker", "highlight",
            [y-25, x-25, y+25, x+25], {"color": "#ff00ff"})

table.on_frame(on_frame)
```

### 2. Interactive Story
```python
# Agent pre-generates scene images, then writes a program to flip through them
scenes = ["Scene 1: Forest", "Scene 2: Dragon", "Scene 3: Castle"]
current_scene = [0]

table.place_overlay("story", "annotation", [0, 0, 100, 1000],
    {"text": f"Page {current_scene[0]+1}: {scenes[current_scene[0]]}"})

# Track a "next page" token on the right side
table.add_zone("next-page", (400, 800, 600, 1000),
    on_enter=lambda obj, zone: advance_page())

def advance_page():
    current_scene[0] = (current_scene[0] + 1) % len(scenes)
    # show_scene equivalent via overlay
    img = table.load_image(scenes[current_scene[0]])
    if img is not None:
        table.place_overlay("scene", "image", [100, 0, 1000, 1000],
            {"prompt": scenes[current_scene[0]]})
    table.notify(f"Turned to page {current_scene[0]+1}")
```

### 3. Math Target Game
```python
import random

score = [0]
target = [random.randint(1, 10)]

def new_target():
    target[0] = random.randint(1, 10)
    # Place number targets around the table
    positions = [(200, 200), (200, 500), (200, 800),
                 (500, 200), (500, 500), (500, 800)]
    for i, (y, x) in enumerate(positions):
        num = random.randint(1, 10)
        is_answer = (i == 0)  # first position is always the answer
        if is_answer:
            num = target[0]
        table.place_overlay(f"num-{i}", "annotation",
            [y-50, x-50, y+50, x+50], {"text": str(num)})
        if is_answer:
            table.add_zone(f"answer-{i}", (y-50, x-50, y+50, x+50),
                on_enter=lambda o, z: on_correct())

def on_correct():
    score[0] += 1
    table.notify(f"Correct! Score: {score[0]}")
    table.play_tone(880, 0.2)
    new_target()

new_target()
table.notify(f"Find the number {target[0]}! Point to it.")
```

---

## Implementation Order & Dependencies

```
1. OverlayStateManager (no deps)     ──┐
2. SessionStore (no deps)              ──┼── can build in parallel
3. ObjectTracker (no deps)             ──┤
4. Async Notification Channel          ──┘
5. Mini-Program Runtime (depends on 1-4)
6. Image Gen Improvements (depends on 2, 4)
7. Integration + System Prompt Updates (depends on all)
```

## Testing Strategy

- **Unit tests first** (red/green TDD): Each component tested in isolation with synthetic data
- **Integration tests**: Components wired together with mock WebSocket
- **All tests run without hardware**: synthetic frames, mock audio, no Gemini API calls
- **Target: 100+ new tests** across all components
