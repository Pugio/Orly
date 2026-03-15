# AGENTS.md

## What This Is

TableLight is a projected AR tutoring system. A student puts homework on a table. A camera sees it. An ADK agent powered by Gemini Live API hears the student and sees the table through a single streaming session. The agent speaks explanations and calls a `project_overlay` tool to physically project graphs/diagrams onto the table via a mini projector. No screen. No headset. Output lands on the physical desk.

Being built for the [Gemini Live Agent Challenge](https://geminiliveagentchallenge.devpost.com/) hackathon (Live Agents category). Full plan in `PROJECT_PLAN.md`.

## Architecture

Two components connected by WebSocket:

**Cloud Run backend** (`backend/`) — FastAPI + ADK. An `Agent` definition with `project_overlay` as a tool. `LiveRequestQueue` receives audio/video from the edge client. `Runner.run_live()` manages the Gemini Live session, dispatches tool calls automatically, handles session resumption and context compression. Deployed on GCP. This is where all agent logic lives.

**Local edge client** (`client/`) — Python asyncio app. Captures camera via IP Webcam, runs ArUco marker detection + homography (OpenCV), captures mic audio, sends rectified frames + PCM audio to backend over WebSocket. Receives audio responses (plays through speakers) and tool results (renders overlays via matplotlib, maps to projector coordinates via homography, displays on projector or screen).

The coordinate system is the key insight: Gemini's 0–1000 bounding boxes map directly to normalised table coordinates, which map to projector pixels through a calibrated homography. See `PROJECT_PLAN.md` §3.2.

## Stack

- Python 3.12+
- `google-adk` — Agent Development Kit (includes `google-genai`, `fastapi`, `pydantic`)
- `opencv-contrib-python` — ArUco detection, homography, image warping
- `numpy` — matrix math
- `matplotlib` — overlay rendering (graphs, diagrams)
- `pyaudio` — mic capture + speaker playback
- `websockets` — edge client WS connection

## Key Technical Decisions

- **ADK, not raw GenAI SDK.** ADK's `LiveRequestQueue.send_realtime()` accepts raw `types.Blob` — our homography-corrected JPEG frames pass through without issue. ADK handles session lifecycle, tool dispatch, reconnection, and state persistence automatically. We write agent definition + tools, not plumbing. The hackathon explicitly lists ADK as a first-class option and using it signals ecosystem fluency. See `PROJECT_PLAN.md` Appendix A for the full decision rationale.
- **Tools as plain Python functions.** ADK infers the function schema from type annotations and docstrings. No manual JSON schema — the docstring IS the schema. ADK calls the function automatically when the model invokes the tool.
- **Vertex AI on Cloud Run, not Google AI API key.** Service account auth, no key management, counts as a Google Cloud service for hackathon requirements. Google AI API key for local dev.
- **FastAPI for the WebSocket layer.** ADK includes FastAPI. We add a WebSocket endpoint that bridges the edge client to the ADK `LiveRequestQueue` / `Runner.run_live()` event stream.
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
├── AGENTS.md              ← you are here
├── PROJECT_PLAN.md        ← full plan, architecture, hackathon strategy
├── backend/               ← Cloud Run service (FastAPI + ADK Runner)
│   ├── main.py            ← WebSocket endpoint + run_live() loop
│   ├── agent.py           ← Agent definition + system prompt
│   └── tools.py           ← project_overlay (plain Python function)
├── client/                ← Local edge client (camera, audio, projector)
├── calibration/           ← Mat generation + projector calibration
├── poc/                   ← Proof-of-concept scripts
├── infra/                 ← Terraform / deploy scripts
├── docs/                  ← Architecture diagram, demo script, blog
└── tests/
```

See `PROJECT_PLAN.md` §11 for the full file tree.

## Current State

- PoC 1 (camera homography): **done** — `poc/poc1_rectify.py`
- Calibration mat: **done** — `calibration/calibration_mat.png`
- Everything else: **not started**

Next priority: PoC 2 (projector calibration), then PoC 4 (Gemini spatial localization) — these can be done in parallel.
