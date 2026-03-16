# Orly Setup & Lessons Learned

## Physical Setup

```
                  PROJECTOR (beaming down)
                      |
        ID 1 ———————————————— ID 2     (far edge)
         |                      |
         |    WORKING AREA      |
         |    (math goes here)  |
         |                      |
        ID 0 ———————————————— ID 3     (near edge)
                      |
                    HUMAN (sitting here)

        CAMERA (off to the side, looking across the mat from the 0/1 short edge)
```

### Key positions:
- **Human** sits at the long edge between markers 0 and 3
- **Projector** is behind the mat (long edge between markers 1 and 2), beaming down onto the table
- **Camera** (phone on IP Webcam) is off to one side, at the short edge between markers 0 and 1, looking across the mat
- **Calibration mat** is A4, printed with 4 ArUco markers (DICT_4X4_50, IDs 0-3) at corners

### Why this matters:
The camera sees the mat sideways relative to how the human reads it. The rectified image has marker 0 at top-left, but from the human's perspective marker 0 is bottom-left. This means text written to be read by the human appears **rotated 90°** in the rectified image sent to Gemini.

---

## Coordinate Systems

### 1. Camera pixel space
Raw pixels from the phone camera (1920x1080 typical). The mat appears as a trapezoid due to perspective.

### 2. Table space (0-1000 normalized)
Defined by the ArUco markers:
- `(0, 0)` = inner corner of marker 0
- `(1000, 0)` = inner corner of marker 1
- `(1000, 1000)` = inner corner of marker 2
- `(0, 1000)` = inner corner of marker 3

This is the shared coordinate system that bridges camera, Gemini, and projector.

### 3. Gemini's coordinate space
Gemini returns bounding boxes as `[ymin, xmin, ymax, xmax]` in 0-1000 normalized to the image it receives. Since the rectified image maps markers to (0,0)→(1000,1000), Gemini's coordinates align with table space — **but only if the image orientation matches**.

### 4. Projector pixel space
1280x720 pixels. `H_proj` transforms table coordinates → projector pixels. The projector is at an angle, so this is a full perspective transform (not just scaling).

---

## The Rotation Problem

The camera is on the 0/1 short edge. The human writes text readable from the 0/3 long edge. In the rectified image (marker 0 = top-left), the text appears rotated 90° CCW.

**Solution:** The `--rotate 270` flag on the client rotates the rectified image 270° CW (= 90° CCW) before sending to Gemini. This makes text upright for Gemini.

**Coordinate un-rotation:** When Gemini returns coordinates from the rotated image, they must be un-rotated back to marker space before being fed to H_proj. The `OverlayManager._unrotate_placement()` handles this.

**Important:** The `--rotate` value depends on the camera position. If the camera moves to a different edge, the rotation changes. For camera on the 0/1 edge looking toward 2/3: use `--rotate 270`.

---

## Projector Calibration

### What we learned the hard way:

1. **Auto-calibration (white dots on black) is unreliable** because:
   - Camera auto-exposure adjusts to ambient light, making projected dots too faint
   - ArUco marker edges create false bright blobs in the diff image
   - Wall reflections can be larger/brighter than the actual dot
   - IP Webcam MJPEG stream buffers frames, causing timing issues (wrong frame captured)
   - Red/night filters on the display dim the projected dots

2. **Manual calibration works:** `calibration/manual_calibrate.py` projects dots one at a time, takes HTTP snapshots (`/photo.jpg` endpoint), and the user clicks where each dot lands in the camera view. The click is transformed through H_cam to get table coordinates.

3. **Need coverage across the full projector:** If calibration dots only land on part of the mat, the homography extrapolates badly to uncovered areas. Use `--margin 0.25` to avoid the projector edges where dots may not appear.

4. **Projector orientation flips content:** The projector beams from behind the mat, so rendered content (graphs, text) appears upside down from the viewer's perspective. Solution: rotate overlay content 180° before projection (but NOT highlights, which are symmetric).

### Running calibration:
```bash
# Manual calibration (recommended)
uv run python calibration/manual_calibrate.py \
    --url http://<PHONE_IP>:8080 \
    --margin 0.25 --rows 3 --cols 4

# Auto calibration (if lighting conditions are good)
uv run python calibration/projector_calibrate.py \
    --url http://<PHONE_IP>:8080 \
    --rows 5 --cols 5 --margin 0.05
```

### Verifying calibration:
```bash
# Test overlay pipeline (no backend needed)
uv run python poc/test_overlay_pipeline.py --projector --h-proj projector_homography.npz
# Type: dots, h, hfull, g, a, clear, q
```

---

## Architecture Insights

### OpenCV on macOS
- **Must run on the main thread.** OpenCV's highgui windows don't render from asyncio tasks or background threads. Solution: run asyncio in a background thread, keep the main thread for the OpenCV display loop.
- **`cv2.waitKey()` doesn't receive keyboard input** when the terminal has focus. Always use a threaded stdin reader for interactive input.
- **Fullscreen window placement:** Create window → move to projector display → wait → set fullscreen. If you set fullscreen before moving, it fullscreens on the primary display.

### Gemini Live API
- **Native-audio model required** for `bidiGenerateContent`: `gemini-2.5-flash-native-audio-latest`
- **Always use AUDIO modality** — the native-audio model rejects TEXT mode
- **For no-audio testing:** Send silence (zero PCM bytes at 16kHz) to keep the audio stream alive. Use `send_content()` for text input alongside the silence stream.
- **Tool calling is supported but flaky** — sometimes the session crashes after tool execution. The `before_tool_callback` on the Agent can help by intercepting and returning results directly.
- **Thinking blocks** appear as `**Bold Header**` text — filter these from transcript output
- **Audio control tokens** appear as `<ctrl46>` — filter these too
- **Event structure:** ADK `Event` has flat fields (`input_transcription`, `output_transcription`, `interrupted`), not nested `server_content`. Tool calls come as `function_call` parts in `event.content.parts`.

### Clean frame caching
When an overlay is projected onto the table, the camera sees it. If that frame is sent to Gemini, the model sees its own overlay and can't read the underlying content. Solution: cache the last clean (pre-overlay) frame and send that instead while overlays are active.

---

## Running the System

### Prerequisites
```bash
# Install dependencies
uv sync --extra dev

# Start IP Webcam on phone, note the IP address
# Connect projector via HDMI (extended display, NOT mirrored)
# Print and place calibration_mat.png on table
```

### Step 1: Calibrate projector
```bash
uv run python calibration/manual_calibrate.py --url http://<PHONE_IP>:8080 --margin 0.25
```

### Step 2: Verify
```bash
uv run python poc/test_overlay_pipeline.py --projector --h-proj projector_homography.npz
```

### Step 3: Start backend
```bash
GOOGLE_API_KEY=<key> uv run uvicorn backend.main:app --port 8080
```

### Step 4: Start client
```bash
uv run python -m client.main \
    --backend ws://localhost:8080/ws/session \
    --url http://<PHONE_IP>:8080 \
    --no-audio \
    --mode projector \
    --h-proj projector_homography.npz \
    --rotate 270
```

### Step 5: Interact
Type questions in the terminal. Gemini will see the table, speak/text responses, and project overlays.

---

## Saved Artifacts

| File | Purpose | When to regenerate |
|------|---------|--------------------|
| `calibration_mat.png` | Printed ArUco mat | Never (unless markers change) |
| `camera_homography.npz` | H_cam matrix | If camera position changes |
| `projector_homography.npz` | H_proj matrix | If projector or mat moves |
| `rectified_output.png` | Test image for offline Gemini testing | When content on mat changes |

---

## Test Suite

173 tests covering all pure functions. Run with:
```bash
uv run pytest -v
```

All rendering, coordinate transforms, message parsing, tool validation, and overlay management are tested. Hardware-dependent code (camera capture, audio, WebSocket connections) is tested via mocks.
