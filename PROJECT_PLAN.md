# TableLight: Projected Learning Overlay System

## Project Plan & Implementation Guide

**Hackathon:** [Gemini Live Agent Challenge](https://geminiliveagentchallenge.devpost.com/)
**Category:** Live Agents 🗣️
**Submission URL:** https://geminiliveagentchallenge.devpost.com/

---

## 1. Hackathon Context & Strategy

### 1.1 Category Fit

The hackathon's Live Agents category describes exactly what we're building: *"Build an agent that users can talk to naturally and can be interrupted. This could be a real-time translator, a **vision-enabled customized tutor that 'sees' your homework**, or a customer support voice agent that handles interruptions gracefully."*

TableLight is the bolded example — taken to its logical conclusion. Instead of overlaying annotations on a screen, we project them directly onto the physical homework using a projector. The agent sees the table through a camera, hears the student through a microphone, speaks explanations through speakers, and physically manifests visual aids next to the relevant problems through a projector. Every input and output channel is a different modality. There is no screen-based chat interface at all.

### 1.2 Competitive Positioning

Most hackathon entries will be browser-based voice agents — a webcam feed in one panel, chat responses in another, maybe some tool calls. That's valuable but familiar. TableLight's differentiator is that **the AI's output exists in the physical world**. The student never looks at a screen. They look at their desk, their paper, their textbook — and the AI's responses appear there, spatially anchored to the content they're studying. This is genuinely novel and should score well on the Innovation criterion.

### 1.3 Judging Criteria Alignment

The hackathon judges score on three axes. Here's how we target each:

**Innovation & Multimodal User Experience — 40% of score**

> *"Does the project break the 'text box' paradigm? Does the agent help 'See, Hear, and Speak' in a way that feels seamless? Does it have a distinct persona/voice? Is the experience 'Live' and context-aware?"*

This is our strongest axis. Key points to make in the demo and writeup:
- No screen, no chat box — output is projected onto the physical world.
- The agent sees (camera → rectified table image), hears (microphone → voice), speaks (audio response → speakers), and *acts* (function call → projected overlay on the table). Four distinct modalities, none of which is text-on-screen.
- Gemini's spatial understanding (bounding boxes in 1000×1000 space) maps directly to physical table coordinates through calibrated homographies. The AI reasons about physical space.
- The student interacts by placing physical materials on the table and talking. Their input modalities are also non-digital: voice + physical arrangement.
- The tutor persona should be warm, patient, and pedagogically sound — not just "helpful assistant." It should ask probing questions, give hints before answers, and celebrate progress. The voice (we pick one of Gemini's voices — Kore or Aoede for warmth) reinforces this.
- Natural interruption handling: the student can say "wait, go back" mid-explanation and the Live API's VAD handles it natively.
- Proactive audio: the tutor can observe the student writing (through periodic video frames) and offer unsolicited hints when it notices an error — like a real tutor leaning over their shoulder.

**Technical Implementation & Agent Architecture — 30% of score**

> *"Does the code effectively utilize the Google GenAI SDK or ADK? Is the backend robustly hosted on Google Cloud? Is the agent logic sound? Does it handle errors gracefully? Does the agent avoid hallucinations?"*

Key points:
- Clean cloud/edge split: the agent brain (raw GenAI SDK Live session, tool orchestration) runs on Cloud Run; the physical I/O layer (camera, projector, audio routing) runs on a local edge client. The two communicate via WebSocket.
- Raw GenAI SDK with separate audio/video streams for minimal latency — no FIFO queue serialization.
- Function calling with spatial semantics: the `project_overlay` tool is a plain Python function; schemas auto-generated from type annotations.
- Dual homography pipeline with fiducial marker calibration — this is real computer vision, not a toy demo.
- Grounding: the tutor only answers about content it can see on the table. The system prompt explicitly instructs it to say "I can't see that problem clearly" rather than guess.
- Error handling: cached homographies for marker occlusion, session resumption with transparent handles for network drops, context window compression for long study sessions.
- Infrastructure as Code: Terraform or `gcloud` deployment scripts in the repo (bonus points).

**Demo & Presentation — 30% of score**

> *"Does the video define the problem and solution? Is the architecture diagram clear? Is there visual proof of Cloud deployment? Does the video show the actual software working?"*

Key points:
- The demo video needs to show the physical setup: camera, projector, table, printed mat, real homework.
- The "wow moment" is when the student says something and a graph physically appears on the table next to the equation. This needs to be clearly visible in the video — good lighting, good camera angle on the table.
- Architecture diagram should emphasise the cloud/edge split and the coordinate system mapping.
- Include a brief shot of the Cloud Run console showing the backend service running.

### 1.4 Mandatory Technical Requirements Checklist

| Requirement | How We Satisfy It |
|-------------|-------------------|
| Must use a Gemini model | `gemini-2.5-flash-native-audio-latest` via Live API |
| Must use Google GenAI SDK or ADK | `google-genai` SDK with `client.aio.live.connect()` for bidirectional streaming |
| Must use at least one Google Cloud service | Cloud Run (agent backend) + Vertex AI Gemini API |
| Agents hosted on Google Cloud | FastAPI + raw GenAI SDK on Cloud Run manages the agent session |

### 1.5 Bonus Points Opportunities

| Bonus | Plan |
|-------|------|
| Blog post / content piece | Write a technical blog covering the homography + Gemini spatial understanding pipeline. Publish on dev.to or Medium with `#GeminiLiveAgentChallenge` hashtag and disclosure. |
| Automated cloud deployment | Terraform config or `deploy.sh` script using `gcloud run deploy`. Include Dockerfile and cloudbuild.yaml in repo. |
| GDG membership | Sign up for nearest Google Developer Group and link profile. |

### 1.6 Submission Deliverables Checklist

| Deliverable | Description | Notes |
|-------------|-------------|-------|
| Text description | Summary of features, tech, data sources, learnings | Section 2 of this doc is the draft |
| Public code repository | GitHub repo with full source | Include README with spin-up instructions |
| Proof of GCP deployment | Screen recording of Cloud Run console showing backend running, or link to deployment code | Record 30-second GCP console walkthrough |
| Architecture diagram | Visual of system: Gemini ↔ backend ↔ frontend, cloud vs. edge | Create clean diagram — see section 3 |
| Demo video | <4 minutes, real software working, pitch the problem/value | See section 12 for the video plan |

---

## 2. What We're Building

TableLight is a projected augmented reality system for learning. A mini projector casts visual overlays — graphs, diagrams, annotations — directly onto physical surfaces like textbooks, homework sheets, or blank paper on a table. A smartphone camera observes the table, and Google's Gemini Live API powers a real-time voice and vision agent that understands what's on the surface. The user speaks naturally, the agent reasons about the content, and produces projected overlays in response.

**The core experience:** A student places their homework on the table. They say, "Can you show me what the graph of this equation looks like?" The system locates the equation on the page, generates a graph, and projects it onto the table right next to the relevant problem — while simultaneously explaining the concept aloud.

This is spatial AR without a headset. Everything stays physical and shared. Multiple people can see the projections. The student can write on real paper while the system annotates around their work.

### Why This Matters

Traditional AR requires wearing a headset or looking through a phone screen. Both break the natural physicality of studying — you can't easily write notes, flip pages, or collaborate while holding a device or wearing goggles. Screen-based AI tutors confine their output to a chat window that competes for the student's attention with their actual work.

By projecting onto the actual surface, we preserve the physical workspace and add to it rather than replacing it. The student's eyes stay on their paper. The AI's responses appear *in context*, spatially anchored to the content they reference.

### Dual-Mode Operation

The system supports two output modes:

1. **Projector mode** (full experience) — overlays are physically projected onto the table surface via a calibrated mini projector. This is the intended experience and the primary demo path.

2. **Screen overlay mode** (development / fallback) — overlays are composited onto the rectified camera image and displayed on the laptop screen. This mode requires no projector and is useful for development, testing, and situations where a projector isn't available. In the demo video, we can show both modes — the screen overlay to prove the spatial reasoning works, and the projector output for the "wow" factor.

---

## 3. System Architecture

### 3.1 Cloud/Edge Split

The system is split into a **cloud-hosted agent backend** and a **local edge client**. This separation is both architecturally clean and a hackathon requirement (backend must run on Google Cloud).

```
┌─────────────────────────────────────────────────────────────┐
│  GOOGLE CLOUD (Cloud Run)                                    │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Agent Backend (FastAPI + raw google-genai SDK)          │ │
│  │                                                          │ │
│  │  ┌──────────────┐    ┌──────────────────────────────┐  │ │
│  │  │ WebSocket     │    │  client.aio.live.connect()    │  │ │
│  │  │ endpoint      │    │                                │  │ │
│  │  │ /ws/session   │◄──▶│  send_realtime_input(audio=)  │  │ │
│  │  │               │    │  send_realtime_input(video=)  │  │ │
│  │  │ Binary frames:│    │  send_client_content(text)    │  │ │
│  │  │  0x01+PCM     │    │                                │  │ │
│  │  │  0x02+JPEG    │    │  session.receive() → events   │  │ │
│  │  │  0x03+PCM out │    │    audio, transcripts, tools  │  │ │
│  │  │               │    │                                │  │ │
│  │  │ JSON frames:  │    │  Tool schemas auto-generated   │  │ │
│  │  │  text, close  │    │  from Python function sigs     │  │ │
│  │  │  transcripts  │    │                                │  │ │
│  │  │  tool results │    │  model=gemini-2.5-flash-       │  │ │
│  │  │  interrupted  │    │    native-audio-latest         │  │ │
│  │  └──────────────┘    └──────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │ WebSocket (wss://)
                         │ Binary + JSON mixed frames
┌────────────────────────▼────────────────────────────────────┐
│  LOCAL EDGE CLIENT (Python)                                  │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────────┐   │
│  │ Camera      │  │ Microphone  │  │ ArUco Detection     │   │
│  │ (IP Webcam) │  │ (laptop)    │  │ + Homography        │   │
│  │ MJPEG       │  │ 16kHz/20ms │  │ (OpenCV)            │   │
│  └──────┬─────┘  └──────┬─────┘  └──────────┬──────────┘   │
│         │               │                     │              │
│         ▼               ▼                     ▼              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  WebSocket Client                                     │   │
│  │  - Binary frames for audio/video (zero-copy, no b64) │   │
│  │  - JSON frames for text/tool results/transcripts      │   │
│  │  - Plays audio responses through speakers             │   │
│  └──────────────────────────────┬───────────────────────┘   │
│                                  │                           │
│              ┌───────────────────▼────────────────────┐     │
│              │  Overlay Renderer + Projection Manager  │     │
│              │  - Renders graphs (matplotlib)          │     │
│              │  - Applies H_proj homography            │     │
│              │  - Composites onto projector canvas     │     │
│              │  - OR composites onto screen overlay    │     │
│              └────────────────────┬───────────────────┘     │
│                                   │                          │
│              ┌────────────────────▼────────────────────┐    │
│              │  Output                                  │    │
│              │  [Projector mode] → USB projector display │    │
│              │  [Screen mode]    → Laptop window overlay │    │
│              └─────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**Why this split?**

- The hackathon requires the backend to be hosted on Google Cloud. By putting the Gemini Live session on Cloud Run, we satisfy this cleanly.
- The raw `google-genai` SDK gives us direct control over audio/video streams — they travel as separate concurrent channels with no FIFO serialization, achieving ~1.6s speech-to-response latency.
- The edge client handles latency-sensitive physical I/O (camera capture, audio at 16kHz, projector rendering) that would be impractical to route through the cloud.
- Binary WebSocket frames for audio/video eliminate base64 encode/decode overhead. JSON is used only for text messages.
- This architecture would let multiple edge clients connect to a single cloud backend in the future (e.g., multiple desks in a classroom).

### 3.2 The Shared Coordinate System

The phone and projector can be at completely different positions and angles. They never need to know about each other. Instead, both are independently calibrated to the same physical table surface through a printed calibration mat with fiducial markers.

**Three coordinate spaces:**

- **Camera pixel space** — where things are in the phone's image sensor.
- **Table coordinate space** — a normalized 2D coordinate system defined by the calibration mat. We normalize to a 1000×1000 grid to match Gemini's bounding box convention.
- **Projector pixel space** — where things are in the projector's output buffer.

**Two homographies bridge them:**

- `H_cam`: Camera pixels → table coordinates. Computed by detecting ArUco fiducial markers on the mat.
- `H_proj`: Table coordinates → projector pixels. Computed by projecting known calibration points and photographing them.

**Why Gemini's coordinate system aligns perfectly:** Gemini returns bounding boxes in a 0–1000 normalized coordinate system (`[ymin, xmin, ymax, xmax]`). Our rectified table image is a top-down view of the mat surface. When we send this to Gemini and it returns spatial coordinates, those map directly to physical table positions. The model can both read content at specific locations and specify where to place output — all in the same coordinate frame. This is not a hack or a workaround; it's a natural alignment between how Gemini understands images and how our system represents the physical world.

---

## 4. Gemini Live API — Technical Reference

### 4.1 Model Selection

| Model | Purpose | Notes |
|-------|---------|-------|
| `gemini-2.5-flash-native-audio-preview-12-2025` | Primary Live API model | Native audio, affective dialog, proactive audio, thinking. 128k context window. Recommended for all Live API use cases. |

### 4.2 ADK Agent Definition

```python
from google.adk.agents import Agent
from google.genai import types

def project_overlay(
    content_type: str,
    placement: list[float],
    title: str,
    data: dict,
) -> dict:
    """Project a visual overlay onto the student's work surface via projector.

    Args:
        content_type: Type of visual — "graph", "diagram", "annotation", or "highlight".
        placement: Where to place it on the table, [ymin, xmin, ymax, xmax] normalised 0-1000.
                   Choose empty space near relevant content. Never overlap existing work.
        title: Label for the overlay.
        data: Content-specific data. For "graph": {"expression": "x**2 - 3*x + 2",
              "x_range": [-5, 5], "y_range": [-5, 10]}. For "annotation": {"text": "..."}.
              For "highlight": {"color": "#00ffff", "target": [ymin, xmin, ymax, xmax]}.

    Returns:
        dict with status of the projection.
    """
    # ADK calls this automatically when the model invokes the tool.
    # The actual rendering + projection happens here.
    overlay_img = render_overlay(content_type, data)
    project_to_surface(overlay_img, placement)
    return {"status": "displayed", "content_type": content_type}


root_agent = Agent(
    name="lumi_tutor",
    model="gemini-2.5-flash-native-audio-preview-12-2025",
    instruction=SYSTEM_PROMPT,
    tools=[project_overlay],  # ADK wraps plain functions as tools via docstrings
)
```

ADK infers the function schema from the type annotations and docstring. No manual JSON schema definition needed — the docstring IS the schema description. ADK handles tool dispatch, execution, and response routing automatically.

### 4.3 ADK Runner + LiveRequestQueue

```python
from google.adk.agents import LiveRequestQueue
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.sessions import InMemorySessionService

# One-time setup
session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name="tablelight",
    session_service=session_service,
)

# Per-connection setup
live_request_queue = LiveRequestQueue()

run_config = RunConfig(
    streaming_mode=StreamingMode.BIDI,
    response_modalities=["AUDIO"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
        )
    ),
    output_audio_transcription={},
    input_audio_transcription={},
    context_window_compression=types.ContextWindowCompressionConfig(
        sliding_window=types.SlidingWindow()
    ),
    session_resumption=types.SessionResumptionConfig(handle=None),
)
```

### 4.4 Sending Inputs via LiveRequestQueue

```python
# Video frame (JPEG-encoded bytes) — our rectified, homography-corrected frame
live_request_queue.send_realtime(
    types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
)

# Audio chunk (16-bit PCM, 16kHz, mono)
live_request_queue.send_realtime(
    types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000")
)

# Text (context injection)
live_request_queue.send_content(
    types.Content(role="user", parts=[types.Part(text="Student placed a new worksheet.")])
)

# Graceful shutdown
live_request_queue.close()
```

The key insight for our use case: `send_realtime()` accepts raw `types.Blob` objects. ADK doesn't care where the bytes come from. Our ArUco detection → homography → rectification pipeline produces JPEG bytes of a top-down table view; those bytes go straight into `send_realtime()`. ADK never knows they were perspective-corrected.

### 4.5 Receiving Events via run_live()

```python
async for event in runner.run_live(
    session=session,
    live_request_queue=live_request_queue,
    run_config=run_config,
):
    # Audio response → forward to edge client for speaker playback
    if event.content and event.content.parts:
        for part in event.content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                await websocket.send_bytes(part.inline_data.data)

    # Transcriptions → forward to edge client for display/logging
    if event.server_content:
        if event.server_content.input_transcription:
            await websocket.send_json({
                "type": "transcript_in",
                "text": event.server_content.input_transcription.text
            })
        if event.server_content.output_transcription:
            await websocket.send_json({
                "type": "transcript_out",
                "text": event.server_content.output_transcription.text
            })
        if event.server_content.interrupted:
            await websocket.send_json({"type": "interrupted"})

    # Tool calls are executed AUTOMATICALLY by ADK.
    # project_overlay() runs server-side. We forward the result to the edge
    # client so it can render and project the overlay.
    if event.actions and event.actions.tool_results:
        for result in event.actions.tool_results:
            await websocket.send_json({
                "type": "tool_result",
                "name": result.function_name,
                "result": result.response
            })
```

**What ADK handles that we don't have to:**
- Tool dispatch and execution — `project_overlay()` is called automatically when the model requests it.
- Session resumption — transparent reconnection on WebSocket drops.
- Context window management — sliding window compression keeps sessions running indefinitely.
- VAD and interruption — the student can interrupt mid-explanation; ADK signals this via events.
- `before_tool_callback` / `after_tool_callback` hooks are available if we need custom logic around tool execution (e.g., sending "rendering..." to the edge client before the overlay renders).

### 4.6 Key Constraints

- Only one response modality per session: TEXT or AUDIO. We use AUDIO + transcription configs for text logging.
- Video is processed at ~1 FPS. Don't exceed ~1-2 frames/second.
- Audio-video sessions without compression: ~2 minutes. With compression: indefinite.
- Connection lifetime: ~10 minutes, but session resumption extends transparently.
- Context window: 128k tokens. Audio ≈ 25 tokens/sec, video ≈ 258 tokens/frame.
- Recommended video resolution: 768×768 JPEG for best results.

---

## 5. Google Cloud Deployment Architecture

### 5.1 Cloud Run Backend

The agent backend is a FastAPI application deployed on Cloud Run. It uses the raw `google-genai` SDK with `client.aio.live.connect()` to establish a bidirectional Gemini Live session. Audio and video are sent as separate concurrent streams (no FIFO queue). Tool calls are handled directly in the receive loop. Session resumption and context window compression are configured natively.

```
tablelight-backend/
├── Dockerfile
├── requirements.txt
├── main.py              # FastAPI app with WebSocket endpoint
├── agent.py             # ADK Agent definition
├── tools.py             # project_overlay tool (plain Python function)
└── deploy.sh            # gcloud run deploy script
```

**`agent.py`** — System prompt, tool schemas auto-generated from Python functions:

```python
from backend.tools import project_overlay, refresh_view, show_scene

SYSTEM_PROMPT = """..."""  # See section 9.6
MODEL = "gemini-2.5-flash-native-audio-latest"

# Auto-generate JSON tool schemas from Python function signatures
TOOL_DECLARATIONS = [function_to_declaration(f) for f in [project_overlay, refresh_view, show_scene]]
TOOL_REGISTRY = {f.__name__: f for f in [project_overlay, refresh_view, show_scene]}
```

**`main.py`** — Raw GenAI SDK, separate audio/video streams:

```python
from google import genai
from google.genai import types

@app.websocket("/ws/session")
async def session_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = genai.Client()

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=SYSTEM_PROMPT,
        tools=[{"function_declarations": TOOL_DECLARATIONS}],
        speech_config=types.SpeechConfig(...),
        realtime_input_config=types.RealtimeInputConfig(...),
        input_audio_transcription={},
        output_audio_transcription={},
        context_window_compression=types.ContextWindowCompressionConfig(...),
        session_resumption=types.SessionResumptionConfig(transparent=True),
    )

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        async def send_from_client():
            # Audio and video sent as SEPARATE concurrent streams
            await session.send_realtime_input(audio=blob)   # no FIFO queue
            await session.send_realtime_input(video=blob)   # independent stream

        async def receive_from_gemini():
            async for msg in session.receive():
                # Audio output, transcriptions, tool calls, interruptions
                if msg.tool_call:
                    for fc in msg.tool_call.function_calls:
                        result = execute_tool(fc.name, fc.args, TOOL_REGISTRY)
                        await session.send_tool_response(function_responses=[...])
```

The tool function runs server-side. The result is forwarded to the edge client for physical rendering and projection. Session resumption handles reconnection transparently.

### 5.2 Deployment

**Dockerfile:**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**`deploy.sh`** (automated deployment for bonus points):

```bash
#!/bin/bash
set -e

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="tablelight-backend"

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com

# Build and deploy
gcloud run deploy $SERVICE_NAME \
    --source . \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}" \
    --session-affinity \
    --min-instances=1 \
    --timeout=3600

echo "Backend deployed. WebSocket URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)'
```

Key Cloud Run settings:
- `--session-affinity`: Critical for WebSocket connections — ensures reconnects hit the same instance.
- `--min-instances=1`: Avoids cold start latency on the first connection.
- `--timeout=3600`: Allow long-running WebSocket sessions.

### 5.3 Vertex AI vs. Google AI API

The `google-genai` SDK supports both backends:

1. **Vertex AI** — set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` env vars. Authenticates via service account on Cloud Run. No API key needed. This is the production path and counts as a Google Cloud service for hackathon requirements.
2. **Google AI API** — set `GOOGLE_API_KEY` env var. Simpler for local development. The backend checks for this env var first.

For the hackathon, use Vertex AI on Cloud Run and Google AI API locally during development.

---

## 6. Hardware Requirements

### Minimum Required

| Component | Purpose | Notes |
|-----------|---------|-------|
| Android smartphone | Camera + microphone input | Any modern phone with decent camera. Same WiFi as laptop. |
| USB mini projector | Overlay output | Even 480p works for graphs/diagrams. |
| Laptop | Edge client + development | Runs Python, connects to phone stream and Cloud Run backend. |
| Printed A4 calibration mat | Defines the table coordinate system | `calibration_mat.png` — print on regular paper. |

### Recommended

| Component | Purpose | Notes |
|-----------|---------|-------|
| 3D-printed phone stand | Stable camera position | Clamp or gorillapod also works. |
| Good ambient lighting | Reliable marker detection | ArUco detection needs contrast. |
| Headphones | Audio output without echo | Prevents VAD feedback loop. |
| Second camera / tripod | Record the demo video | You need to film the table while the system runs. |

### Software Dependencies

```bash
pip install google-adk opencv-contrib-python numpy matplotlib pyaudio websockets
```

`google-adk` includes `google-genai`, `fastapi`, and `pydantic` as transitive dependencies.

Also: **IP Webcam** (free Android app) and a **Google Cloud project** with billing enabled.

---

## 7. Proof-of-Concept Roadmap

| PoC | What It Proves | Key Risk |
|-----|---------------|----------|
| 1 | Camera → table homography works from phone at angle | Marker detection reliability |
| 2 | Projector → table mapping works | Projector brightness, resolution |
| 3 | Camera → projector round trip | Geometric pipeline composes correctly |
| 4 | Gemini localizes content from rectified table image | AI spatial understanding accuracy |
| 5 | Gemini produces overlays via function calling | Prompt/schema design |
| 6 | Cloud Run backend + edge client end-to-end | Full integration with voice |

PoCs 1–4 are independent. PoC 5 depends on 4. PoC 6 depends on all.

---

## 8. Existing Artifacts

### `calibration/generate_mat.py`

Generates a printable calibration mat with four ArUco markers (IDs 0–3, `DICT_4X4_50` dictionary, 25mm square, 5mm margin). Supports A4, US Letter, A3, Tabloid, Legal, and custom dimensions via CLI args. Run with `python -m calibration.generate_mat [--paper letter|a3|...]`.

### `poc1_rectify.py`

Complete PoC 1 implementation. Connects to IP Webcam or local webcam, detects markers, computes homography, displays live rectified top-down view. Includes debug overlay, homography caching, and keyboard controls (`q` quit, `s` save snapshot, `d` toggle debug).

---

## 9. Detailed PoC Implementation Steps

### 9.1. PoC 1 — Camera-to-Table Homography

**Goal:** Reliably detect the calibration mat from a phone at an oblique angle and produce a clean rectified top-down image.

**Status: Code complete.**

#### Setup

```bash
python -m calibration.generate_mat     # Creates calibration/calibration_mat_a4.png
# Print on A4, "actual size" / no scaling
# Place on table, mount phone on stand, start IP Webcam
python poc1_rectify.py --url http://<PHONE_IP>:8080
```

#### Success criteria

- `LOCKED` status (all 4 markers detected) at your mounting distance/angle.
- Rectified view is clean, readable, minimal distortion.
- Text on paper is legible in the rectified view.
- Stable through brief marker occlusion.

#### Troubleshooting

| Problem | Fix |
|---------|-----|
| No markers detected | Increase `MARKER_SIZE_MM` to 40-50mm, reprint |
| Intermittent detection | Better lighting, reduce camera distance, 720p in IP Webcam |
| Distorted rectified image | Move phone further back or closer to overhead (45° sweet spot) |
| Can't connect to stream | Check IP, same WiFi, try `http://<IP>:8080/video` in browser |

---

### 9.2. PoC 2 — Projector-to-Table Mapping

**Goal:** Compute a mapping from table coordinates to projector pixels.

**Status: Done.** Manual click-based calibration in `calibration/manual_calibrate.py`.

#### Approach

Project a grid of bright dots, photograph each with the calibrated camera, compute correspondences from projector pixels to table coordinates, derive `H_proj` via `cv2.findHomography`.

#### Key implementation steps

1. Connect projector as extended display. Open fullscreen window on it (OpenCV or pygame).
2. Project dots one at a time (white circle on black, 4×3 grid).
3. For each: wait 500ms, capture frame, subtract background, find centroid, apply `H_cam` to get table coordinates.
4. Compute `H_proj = cv2.findHomography(table_points, projector_points)`.
5. Save to `projector_homography.npz`.
6. Verify: click in rectified view → dot appears at correct physical location.

#### Success criteria

- Dots land within ~5mm of intended position across full projectable area.

---

### 9.3. PoC 3 — Camera-to-Projector Round Trip

**Goal:** Full geometric pipeline: detect physical object → project annotation at correct position.

**Status: Done.** Integrated in the full edge client.

#### Approach

Place a coloured sticky note or extra ArUco marker on the table. Detect it in the camera, compute table coordinates, apply `H_proj`, project an annotation next to it. Run in a loop so the annotation tracks the object.

#### Success criteria

- Annotation within ~10mm of intended position.
- Tracks moving object at 5+ FPS.

---

### 9.4. PoC 4 — Gemini Spatial Localization

**Goal:** Gemini accurately returns bounding boxes for content in a rectified table image.

**Status: Done.** Gemini places overlays via project_overlay tool with 0-1000 coordinate system.

#### Approach (Jupyter notebook)

PoC 4 uses the standard (non-streaming) Gemini API via `genai.Client` to test spatial localization in isolation. This is intentional — no need for ADK or Live API here; we just need to validate that Gemini can return accurate bounding boxes from a single image.

```python
from google import genai
from PIL import Image

client = genai.Client(api_key="YOUR_KEY")
img = Image.open("rectified_output.png")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        img,
        "This is a top-down photo of a student's work surface. "
        "Identify each distinct problem, equation, or figure. "
        "Return a JSON array: {\"label\": str, \"box_2d\": [ymin, xmin, ymax, xmax]} "
        "normalized to 0-1000. Return ONLY JSON."
    ],
)
```

Parse, draw boxes on image, verify accuracy visually.

#### Success criteria

- 80%+ of distinct content areas correctly identified and localized on a printed page.
- Works for at least two content types.

---

### 9.5. PoC 5 — Function Calling for Overlay Generation

**Goal:** Gemini calls `project_overlay` with correct content and placement.

**Status: Done.** Tool schemas auto-generated from Python functions via `function_to_declaration()`.

#### Function definition

In the full system (PoC 6), ADK infers the schema from the Python function's type annotations and docstring — see `PROJECT_PLAN.md` §4.2 for the `project_overlay` function definition. No manual JSON needed.

For PoC 5, we test with the standard (non-Live) Gemini API first, which requires an explicit schema:

```python
project_overlay_declaration = {
    "name": "project_overlay",
    "description": (
        "Project a visual overlay onto the student's work surface via projector. "
        "Placement in 0-1000 normalised coordinates [ymin, xmin, ymax, xmax]."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content_type": {
                "type": "string",
                "enum": ["graph", "diagram", "annotation", "highlight"],
            },
            "placement": {
                "type": "array",
                "items": {"type": "number"},
                "description": "[ymin, xmin, ymax, xmax] 0-1000. Choose empty space near relevant content."
            },
            "title": {"type": "string"},
            "data": {
                "type": "object",
                "description": (
                    "For 'graph': {expression, x_range, y_range}. "
                    "For 'annotation': {text}. "
                    "For 'highlight': {color, target [ymin,xmin,ymax,xmax]}."
                )
            }
        },
        "required": ["content_type", "placement", "data"]
    }
}
```

Rendering uses matplotlib with black backgrounds (projector transparent). Overlays composited onto a preview image to validate before integrating with the projector.

#### Success criteria

- Gemini calls tool with correct parameters for the visible content.
- Placement is in empty space near the relevant problem.

---

### 9.6. PoC 6 — End-to-End with Raw GenAI SDK

**Goal:** Full integration: student speaks → agent sees table → speaks + projects overlay.

**Status: Done.** Backend uses raw `google-genai` SDK with `client.aio.live.connect()`. Edge client unchanged.

#### Cloud Run backend

FastAPI + ADK Runner on Cloud Run. `LiveRequestQueue` receives audio/video from edge client; `Runner.run_live()` manages the Gemini session and dispatches tool calls automatically. Edge client connects via `wss://`.

See section 5 for the full backend code and deployment.

#### Edge client

`asyncio`-based Python app running four concurrent tasks:

1. **Video capture + rectification** — IP Webcam → ArUco → homography → rectified JPEG → WebSocket to backend at ~1 FPS.
2. **Audio capture** — laptop mic → 16kHz PCM → WebSocket to backend continuously.
3. **Response handler** — WebSocket from backend → audio playback + tool call routing + transcription logging.
4. **Projection manager** — renders overlays, applies `H_proj`, composites onto projector canvas (or screen overlay).

#### System prompt

```python
SYSTEM_PROMPT = """You are a friendly, encouraging maths tutor called Lumi.
You can see the student's work surface through a camera.

BEHAVIOUR:
- When the student asks about a problem, identify it on the surface first.
- Explain concepts verbally in clear, age-appropriate steps.
- If a visual would help, use project_overlay to display it near the problem.
- Ask follow-up questions to check understanding.
- Offer hints before full solutions.
- Celebrate when the student gets something right.

SPATIAL AWARENESS:
- The table surface uses a 0-1000 normalised coordinate system.
- Top-left is (0,0), bottom-right is (1000,1000).
- Place overlays in empty space near relevant content.
- NEVER place overlays on top of the student's existing work.
- If you can't clearly see a problem, say so honestly.

GROUNDING:
- Only discuss content you can actually see on the table.
- If asked about something not visible, ask the student to point to it or place it on the table.
- Do not guess or hallucinate problem content."""
```

The persona name "Lumi" (from "illuminate") reinforces the projection concept and gives the agent a distinct identity for the demo.

#### Session management for long study sessions

```python
config = types.LiveConnectConfig(
    # ...
    context_window_compression=types.ContextWindowCompressionConfig(
        sliding_window=types.SlidingWindow()
    ),
    session_resumption=types.SessionResumptionConfig(handle=session_handle),
)
```

Context compression extends sessions indefinitely. Session resumption handles (valid 24 hours) survive connection drops. The edge client implements a reconnection loop with the latest handle.

#### Success criteria

- Speak a question → receive verbal response + projected overlay in correct position.
- Three consecutive interactions without crash or calibration loss.
- Voice → projection latency under 5 seconds.

---

## 10. Tutor Persona & Pedagogical Design

This section matters for the "distinct persona/voice" aspect of the Innovation judging criterion.

### Voice selection

Gemini Live offers voices: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr. For a warm, patient tutor:
- **Kore** — clear, warm, and measured. Good for explanations.
- **Aoede** — slightly more expressive. Good for encouragement.

Test both and pick the one that feels most like a real tutor.

### Pedagogical approach

The system prompt instructs the tutor to:
- **Scaffold** — break complex problems into steps, ask the student to try each step before revealing the next.
- **Hint before answer** — "What do you think happens when x = 0?" before showing the intercept.
- **Use spatial reference** — "Look at the graph I just projected next to problem 3 — see how the curve crosses the x-axis?"
- **Check understanding** — "Does that make sense? Can you tell me what the slope means here?"
- **Be encouraging** — "That's exactly right!" / "Good instinct — you're close."

### Proactive mode (stretch goal)

With proactive audio enabled, the tutor could observe periodic video frames and notice when the student writes an incorrect step — then offer a gentle correction without being asked. This would feel remarkably like having a real tutor looking over your shoulder.

---

## 11. Project File Structure

```
tablelight/
├── README.md                            (spin-up instructions, architecture)
├── LICENSE
├── requirements.txt
├── .env.example                         (GOOGLE_API_KEY, PHONE_IP, BACKEND_URL)
│
├── backend/                             (Cloud Run — FastAPI + raw GenAI SDK)
│   ├── main.py                          (FastAPI + WebSocket + binary protocol)
│   ├── agent.py                         (system prompt + tool schema generation)
│   └── tools.py                         (project_overlay, refresh_view, show_scene)
│
├── client/                              (Local edge client)
│   ├── main.py                          (asyncio orchestrator)
│   ├── camera.py                        (IP Webcam capture + ArUco + homography)
│   ├── audio.py                         (mic capture + speaker playback)
│   ├── projector.py                     (fullscreen window + overlay compositing)
│   ├── renderer/
│   │   ├── graph.py                     (matplotlib graph rendering)
│   │   ├── annotation.py               (text/arrow annotations)
│   │   └── highlight.py                (region highlighting)
│   └── ws_client.py                     (WebSocket client to Cloud Run backend)
│
├── simulation/                          (hardware-free testing + benchmarks)
│   ├── fake_audio.py                    (synthetic PCM: silence, sine, TTS)
│   ├── fake_camera.py                   (synthetic JPEG test frames)
│   ├── sim_client.py                    (WS client that runs scenarios)
│   ├── scenarios.py                     (silence, question, tool call, interrupt)
│   └── latency_benchmark.py            (CLI runner with summary tables)
│
├── infra/                               (Cloud Run deployment)
│   ├── Dockerfile                       (slim Python 3.12, backend deps only)
│   ├── deploy.sh                        (gcloud run deploy script)
│   ├── .dockerignore
│   └── cloudbuild.yaml
│
├── calibration/
│   ├── calibration/generate_mat.py       ✅ exists
│   ├── calibration_mat.png              ✅ exists
│   ├── poc2_projector_calibrate.py      (to build)
│   └── poc2_projector_verify.py         (to build)
│
├── poc/
│   ├── poc1_rectify.py                  ✅ exists
│   ├── poc3_roundtrip.py                (to build)
│   ├── poc4_gemini_localize.ipynb       (to build)
│   └── poc5_function_calling.py         (to build)
│
├── infra/                               (IaC for bonus points)
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── cloudbuild.yaml
│
├── docs/
│   ├── architecture_diagram.png         (for submission)
│   ├── demo_video_script.md
│   └── blog_post.md                     (for bonus points)
│
└── tests/
    ├── test_homography.py
    ├── test_renderer.py
    └── test_tools.py
```

---

## 12. Demo Video Plan

The demo video must be under 4 minutes and show real software working. Here's a structured plan:

### Shot list

**[0:00–0:30] The problem (30s)**
- Show a student at a desk with a maths textbook, looking frustrated.
- Narrate: "When students study alone, they hit a wall — they can't visualise the graph, can't see the connection between the equation and the shape. They need a tutor, but tutors aren't always available."
- Cut to a phone showing a typical chatbot interface. "Current AI tutors are confined to a screen — but the student's work is on their desk."

**[0:30–1:00] The solution (30s)**
- Wide shot of the TableLight setup: mat on table, phone on stand, projector to the side.
- "TableLight brings the tutor to the desk — literally."
- Student places homework on the mat. Quick shot of the camera view and rectified view on the laptop.

**[1:00–2:30] The demo (90s)**
- Live interaction 1: Student says "Hey Lumi, can you help me with problem 3?" Tutor responds verbally, identifies the equation, a graph appears projected on the table next to the problem. Hold the shot long enough for the audience to see the spatial relationship.
- Live interaction 2: Student asks a follow-up: "What happens if I change the coefficient?" Tutor explains, projects a second graph overlaid or adjacent for comparison.
- Live interaction 3: Student interrupts mid-explanation — "Wait, go back." Tutor handles the interruption gracefully.
- Show both projector mode (physical projection on table) and screen overlay mode (for viewers who want to see the coordinates clearly).

**[2:30–3:15] Architecture & Cloud (45s)**
- Show the architecture diagram (cloud/edge split).
- Quick cut to Cloud Run console showing the backend service running (proof of GCP deployment).
- Highlight: "Gemini Live API handles voice + vision + function calling in a single streaming session. The backend runs on Cloud Run. The edge client handles the physics — camera calibration, projector mapping."

**[3:15–3:50] Technical highlights (35s)**
- Brief explanation of the homography pipeline with a visual.
- Show the 1000×1000 coordinate system alignment between Gemini's bounding boxes and the physical table.
- "The AI reasons about physical space — it knows where things are on the table and where to place its output."

**[3:50–4:00] Close (10s)**
- "TableLight: bringing AI out of the screen and onto the desk."
- GitHub repo URL.

### Production notes

- Film from a tripod looking down at the table at ~45° to show both the physical paper and the projected overlay clearly.
- Good lighting is essential — the projector output needs to be visible on camera.
- Use a lavalier mic on the student for clean voice capture.
- Record the Cloud Run console screencast separately.

---

## 13. Key Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Projector too dim in ambient light | High | Test early. Bright colours on black. Dim room if needed. Screen overlay mode as fallback. |
| Projector resolution too low | Medium | Focus on graphs/diagrams, not text. Thick lines, large labels. |
| ArUco markers occluded | Low | Homography caching. Markers at corners, work in centre. |
| Gemini bounding boxes inaccurate | Medium | 50 units / 1000 ≈ 15mm offset — acceptable for "near" placement. |
| Context window fills with video | High | Context compression enabled. Reduce to 0.5 FPS if needed. |
| Audio echo / feedback | Medium | Use headphones. Configure VAD sensitivity. |
| WebSocket connection drops | Low | Session resumption (handles valid 24hr) + auto-reconnect loop. |
| Cloud Run cold start | Low | `--min-instances=1` keeps one warm instance. |

---

## 14. Latency Optimizations Applied

Measured via simulation harness (`simulation/latency_benchmark.py`):

| Optimization | Before | After | Saving |
|---|---|---|---|
| ADK → raw GenAI SDK (separate audio/video streams) | ~5000ms | 2249ms | ~2750ms |
| silence_duration_ms 1000 → 300 | 2249ms | ~1937ms | ~312ms |
| prefix_padding_ms 200 → 50 | — | — | ~150ms |
| JSON+base64 → binary WS frames for audio/video | 1937ms | 1593ms | ~344ms |
| Audio chunks 800→320 samples (50ms→20ms) | — | — | faster VAD onset |
| JPEG quality 85→70 for Gemini input | — | — | ~30% smaller frames |
| Cached _has_content flag (replaces per-frame np.any scan) | — | — | ~1ms/frame |

**Final benchmark (simple_question): 1593ms round-trip** (3.1x faster than ADK).

### WebSocket Protocol

Binary frames with 1-byte type prefix for audio/video (zero serialization overhead):
- `0x01` + PCM → audio from client (16kHz, 16-bit mono, 20ms chunks)
- `0x02` + JPEG → video from client (1 FPS, quality 70)
- `0x03` + PCM → audio response from server (24kHz)

JSON text frames for everything else (small, infrequent):
- `{"type": "text", "text": "..."}` — text input
- `{"type": "transcript_in/out", "text": "..."}` — transcriptions
- `{"type": "tool_result", "name": "...", "result": {...}}` — tool calls
- `{"type": "interrupted"}` — barge-in

---

## 15. Future Directions

Beyond the hackathon:

- **On-device processing.** Android app with Firebase AI Logic for direct Gemini Live access, eliminating WiFi camera latency.
- **Proactive tutoring.** Proactive audio mode lets the tutor observe and intervene unprompted.
- **Multi-content overlays.** Step-by-step solution animations, chemistry structures, language flashcards, music notation.
- **Hand gesture interaction.** Detect pointing to select a problem without voice.
- **Classroom mode.** Multiple edge clients connecting to a single Cloud Run backend — one teacher's aide covering many desks.
- **Higher-quality projector.** 1080p short-throw enables readable text and detailed diagrams.

---

## Appendix A: Architecture Decisions

### Raw GenAI SDK (current approach)

We use `google-genai` with `client.aio.live.connect()` directly. This gives full control over the WebSocket session, tool dispatch, and event handling. We initially used ADK but switched to the raw SDK because **ADK's `LiveRequestQueue` serializes all input (audio, video, text) into a single FIFO queue**, causing audio to get stuck behind video frames and adding ~5 seconds of speech-to-transcription latency. The raw SDK natively supports separate `audio=`/`video=` streams, eliminating this bottleneck. Tool schemas are auto-generated from Python function signatures via `function_to_declaration()` in `backend/agent.py`, preserving ADK's convenience without the overhead.

### ADK (previous approach, abandoned)

ADK (`google-adk`) provided `Runner.run_live()` + `LiveRequestQueue` for session lifecycle. While convenient, the FIFO queue design was fundamentally incompatible with our real-time audio+video streaming use case. We had to monkey-patch ADK's internal `GeminiLlmConnection.send_realtime` to split audio/video, capture the raw Gemini session to bypass the queue, and intercept all tool calls in `before_tool_callback` due to flaky dispatch with the native-audio model. The workarounds added more code than the raw SDK approach would have required from scratch.

### Pipecat (by Daily)

Open-source framework with `GeminiLiveLLMService`. Handles asyncio, VAD, transcription, WebRTC. A viable alternative to ADK with a different architectural philosophy (pipeline of processors vs. agent + tools). Worth considering if we need WebRTC for browser-based clients in the future.

### LiveKit

WebRTC infrastructure with Gemini Live integration. More relevant for scaling beyond a single-laptop setup or adding browser/mobile clients.
