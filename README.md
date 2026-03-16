# Orly

A real-time AI agent that lives on your desk. Point a camera at your table, talk naturally, and Orly sees your world, speaks back, and projects images, diagrams, music, stories, and more directly onto the physical surface through a mini projector.

Help your kid with homework. Create a story together with AI-generated illustrations. Explore the solar system projected onto your kitchen table. Generate background music while you work. No screen. No headset. Just your desk, your voice, and light.

**Orly** (from **O**ve**RL**a**Y**) is a seamless blend of digital and material — powered by Gemini's Live API.

## What Can Orly Do?

- 📈 **Help with homework** — graph equations, explain step-by-step, highlight problems, quiz with flashcards
- 🎨 **Create images** — ask Orly to draw anything and it appears on your table (Gemini image generation)
- 📖 **Tell stories** — collaboratively build illustrated stories, scene by scene, projected onto the desk
- 🎵 **Generate music** — AI-composed background music while you work or study (Google Lyria)
- 🎬 **Generate videos** — create short videos projected onto your surface (Google Veo)
- 🔬 **Explore subjects** — chemistry molecules, geometry constructions, historical timelines, vocabulary cards
- 🌍 **Explore the world** — ask about anything on the table and Orly explains it with visuals
- ✏️ **Annotate & highlight** — Orly marks up your physical materials with projected labels and regions

## How It Works

1. **Camera** sees your table (via local webcam or IP Webcam phone)
2. **Microphone** captures your voice
3. **Backend** bridges everything to a Gemini Live API session — audio + video streamed in real time
4. **Gemini** sees your surface, hears you, speaks back, and calls tools to project overlays
5. **Projector** (or screen fallback) renders overlays onto the table via calibrated homography

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A Gemini API key (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)
- A webcam (local or IP Webcam app)
- A mini projector (optional (highly recommended) — screen mode works without one)
- A printed calibration mat (generated in setup)

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Print the calibration mat

```bash
uv run python -m calibration.generate_mat
```

This generates `calibration/calibration_mat.png` — a page with 4 ArUco markers at the corners. Print it and lay it flat on your table. The markers define the table coordinate system.

Options:
```bash
uv run python -m calibration.generate_mat --paper letter    # US Letter (default is A4)
uv run python -m calibration.generate_mat --paper a3        # A3
uv run python -m calibration.generate_mat --dpi 300         # Higher resolution
```

### 3. Calibrate the projector

> **Skip this step** if you're using `--mode screen` (no projector). Only needed for projector output.

The calibration computes a homography that maps table coordinates to projector pixels. You have two options:

**Manual calibration** (recommended — you click where each projected dot lands):
```bash
uv run python calibration/manual_calibrate.py --webcam 0
```

**Automatic calibration** (camera detects the dots — can be finicky):
```bash
uv run python calibration/projector_calibrate.py --webcam 0
```

Both will:
1. Open a fullscreen black window on the projector
2. Project bright dots one at a time onto the mat
3. You click (manual) or the camera detects (auto) where each dot landed
4. Compute the homography and save it to `projector_homography.npz`

If you're using an IP Webcam phone instead of a local webcam, replace `--webcam 0` with `--url http://<phone-ip>:8080`.

### 4. Verify the calibration

After calibrating, verify everything works with the test pattern viewer:

```bash
uv run python calibration/projector_verify.py
```

This auto-detects `projector_homography.npz` and cycles through test patterns with `Space`/`n` (next), `p` (previous), `q` (quit):

1. **Rectangle** — basic projector output test
2. **Graph** — matplotlib-rendered graph overlay
3. **Annotation** — text rendering
4. **Highlight** — semi-transparent colored region
5. **Calibration grid** — dots at every 200 table units, color-coded (red corners, green center, cyan elsewhere)
6. **Crosshair** — cross at center (500, 500) with red corner markers

Patterns 5–6 require a homography file and verify calibration accuracy — the dots should line up with the corresponding positions on your printed mat. If they're off, recalibrate. You can also pass `--homography path/to/file.npz` explicitly.

> **Recalibrate** whenever you move the projector, camera, or mat. If overlays land in the wrong spot, recalibrate.

## Running

You need two terminals.

### Terminal 1 — Backend

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8080
```

### Terminal 2 — Edge client

**Screen mode** (no projector, overlays shown in a laptop window):
```bash
uv run python -m client.main \
  --backend ws://localhost:8080/ws/session \
  --webcam 0 \
  --mode screen
```

**Projector mode** (overlays projected onto the table):
```bash
uv run python -m client.main \
  --backend ws://localhost:8080/ws/session \
  --webcam 0 \
  --h-proj projector_homography.npz \
  --mode projector
```

### Client options

| Flag | Description |
|------|-------------|
| `--webcam N` | Local webcam index (e.g. `0`) |
| `--url URL` | IP Webcam URL (alternative to `--webcam`) |
| `--backend URL` | Backend WebSocket URL |
| `--mode screen` | Show overlays on laptop (default) |
| `--mode projector` | Output overlays to projector |
| `--h-proj FILE` | Projector homography file |
| `--fps FLOAT` | Video frame rate sent to backend (default: 1.0) |
| `--no-audio` | Disable mic/speaker (useful for testing) |

## Testing

```bash
uv run pytest
```

### Simulation (no hardware needed)

Run the synthetic audio/video benchmark without a camera or projector:
```bash
uv run python -m simulation.latency_benchmark
```

## Architecture

```
┌─────────────┐     WebSocket      ┌─────────────────┐     Live API     ┌─────────┐
│ Edge Client  │ ◄──────────────► │  Backend (Cloud   │ ◄────────────► │  Gemini  │
│ camera, mic, │   audio/video/    │  Run / local)     │  audio/video/   │  Live    │
│ projector    │   overlays        │  FastAPI + genai   │  tool calls     │          │
└─────────────┘                    └─────────────────┘                   └─────────┘
```

- **Backend** (`backend/`) — FastAPI + raw `google-genai` SDK. Maintains a bidirectional Gemini Live session with separate audio and video streams.
- **Client** (`client/`) — Captures camera + mic, sends to backend, receives audio responses + tool results, renders overlays via matplotlib, maps to projector coordinates via homography.
- **Calibration** (`calibration/`) — Mat generation + projector homography calibration.
- **Simulation** (`simulation/`) — Synthetic audio/video pipeline for testing without hardware.

## License

MIT — see [LICENSE](LICENSE).

Built for the [Gemini Live Agent Challenge](https://geminiliveagentchallenge.devpost.com/).
