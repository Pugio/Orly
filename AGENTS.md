# AGENTS.md

## What This Is

TableLight is a projected AR tutoring system. A student puts homework on a table. A camera sees it. A Gemini Live API session hears the student and sees the table through a single streaming session. The agent speaks explanations and calls a `project_overlay` tool to physically project graphs/diagrams onto the table via a mini projector. No screen. No headset. Output lands on the physical desk.

Being built for the [Gemini Live Agent Challenge](https://geminiliveagentchallenge.devpost.com/) hackathon (Live Agents category). Full plan in `PROJECT_PLAN.md`.

## Architecture

Two components connected by WebSocket:

**Cloud Run backend** (`backend/`) — FastAPI + raw google-genai SDK. Uses `client.aio.live.connect()` to establish a bidirectional Gemini Live session. Audio and video are sent as separate concurrent streams (no FIFO queue). Tool calls are handled directly in the receive loop. Session resumption and context window compression are configured natively. Deployed on GCP.

**Local edge client** (`client/`) — Python asyncio app. Captures camera via IP Webcam, runs ArUco marker detection + homography (OpenCV), captures mic audio, sends rectified frames + PCM audio to backend over WebSocket. Receives audio responses (plays through speakers) and tool results (renders overlays via matplotlib, maps to projector coordinates via homography, displays on projector or screen).

**Simulation harness** (`simulation/`) — Synthetic audio/video pipeline for testing without hardware. Generates fake PCM audio (silence, sine waves, TTS), fake camera frames (JPEG), connects to the real backend, and measures end-to-end latency. Pre-defined scenarios for smoke testing and benchmarking.

The coordinate system is the key insight: Gemini's 0–1000 bounding boxes map directly to normalised table coordinates, which map to projector pixels through a calibrated homography. See `PROJECT_PLAN.md` §3.2.

## Stack

- Python 3.12+
- `google-genai` — Raw Gemini SDK (Live API for bidirectional audio/video streaming)
- `fastapi` + `uvicorn` — WebSocket server
- `opencv-contrib-python` — ArUco detection, homography, image warping
- `numpy` — matrix math
- `matplotlib` — overlay rendering (graphs, diagrams)
- `pyaudio` — mic capture + speaker playback
- `websockets` — edge client WS connection

## Key Technical Decisions

- **Raw GenAI SDK, not ADK.** ADK's `LiveRequestQueue` serializes all input (audio, video, text) into a single FIFO queue. Audio gets stuck behind video frames, adding ~5s speech-to-transcription latency. The raw SDK natively supports separate `audio=`/`video=` streams via `session.send_realtime_input()`. The backend went from 437 lines of workarounds to ~250 lines of clean code.
- **Tools as plain Python functions.** `function_to_declaration()` in `agent.py` auto-generates JSON tool schemas from Python function signatures + docstrings. Same convenience as ADK, no dependency.
- **Vertex AI on Cloud Run, not Google AI API key.** Service account auth, no key management, counts as a Google Cloud service for hackathon requirements. Google AI API key for local dev.
- **Black background for all overlays.** Projectors add light — black is transparent. All rendering must use dark backgrounds with bright content.
- **Homography caching.** Camera homography is cached on last-good-detection so brief marker occlusion doesn't break the system.
- **Screen overlay as fallback.** If no projector is connected, overlays composite onto the rectified camera view in a laptop window. Same code path, different output target.

## Coding Rules

1. **Red/green TDD.** Write a failing test first. Make it pass. Then refactor. No untested code lands in main.
2. **Commit at logical intervals.** One commit per coherent change — a new function, a bug fix, a refactor. Not "end of day" dumps. Write descriptive commit messages.
3. **Review and refactor regularly.** After each PoC milestone, review the code. Extract shared utilities. Remove dead code. Improve names. Don't let tech debt compound.

## File Layout

```
tablelight/
├── CLAUDE.md              ← you are here
├── PROJECT_PLAN.md        ← full plan, architecture, hackathon strategy
├── backend/               ← Cloud Run service (FastAPI + raw GenAI SDK)
│   ├── main.py            ← WebSocket endpoint + Gemini Live session
│   ├── agent.py           ← System prompt + tool schema generation
│   └── tools.py           ← project_overlay, refresh_view, show_scene, run_program, stop_program, list_programs, get_overlay_state
├── client/                ← Local edge client (camera, audio, projector)
│   ├── overlay_state.py   ← Named overlay tracking (CRUD, JSON/ASCII state)
│   ├── session_store.py   ← File-backed session storage (images, programs)
│   ├── object_tracker.py  ← Color/template object tracking with zone triggers
│   ├── program_runtime.py ← Mini-program runtime (sandboxed exec, TableAPI)
│   └── ...                ← camera, audio, overlay_manager, ws_client, renderers
├── simulation/            ← Synthetic audio/video for testing + benchmarks
├── calibration/           ← Mat generation + projector calibration
├── poc/                   ← Proof-of-concept scripts
├── infra/                 ← Terraform / deploy scripts
├── docs/                  ← Architecture diagram, demo script, blog
└── tests/                 ← 991 tests
```

## Current State

- End-to-end system: **working** — camera → backend → projector overlay
- ADK → raw GenAI SDK migration: **done** — separate audio/video streams, no FIFO
- Simulation harness: **done** — `uv run python -m simulation.latency_benchmark`
- Test coverage: **991 tests** passing
- Manual projector calibration: **done**
- Image generation (Gemini): **done** — with enhance mode for user drawings
- Markdown/annotation rendering: **done**
- Mini-program runtime: **done** — agent writes Python, runs on client with TableAPI
- Object tracking: **done** — color (CamShift) + template matching, zone triggers
- Named overlay state: **done** — CRUD, JSON export, ASCII grid visualization
- Session storage: **done** — images + programs persisted to `session/`
- Async notification channel: **done** — bi-directional text for task completion
- Renderer registry: **done** — auto-discovery, one-file renderer additions
- Projector perspective warp: **done** — proper warpPerspective, axis order fix
- Error recovery UX: **done** — grace periods, priority messages, error overlays
- Multi-subject support: **done** — math, science, language, history
- Session persistence: **done** — save/restore overlays, debounced auto-save
- Flashcard renderer: **done** — front/back with flip tool
- Perf: **done** — JPEG round-trip eliminated, tracker offloaded to thread pool
