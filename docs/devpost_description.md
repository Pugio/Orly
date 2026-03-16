# TableLight — Devpost Submission

## Inspiration

Every AI tutor today lives behind a screen. The student looks at their homework, then looks at the chat window, then back at their homework — context-switching constantly between the physical and digital worlds. We asked: **what if the AI's output existed in the physical world instead?**

We were inspired by the idea that studying is fundamentally physical. Students write on paper, flip textbook pages, point at problems. AR headsets are too expensive and isolating for a child. Phone screens are too small and require holding a device. But a projector? A projector just adds light to what's already there.

When we saw the Gemini Live Agent Challenge call for vision-enabled tutors, we realized Gemini's spatial understanding — its ability to return bounding boxes in a normalized coordinate space — could map directly to physical table coordinates. The AI doesn't just *see* the homework. It knows *where* things are, and can place its responses *right next to* the relevant problem. That was the spark.

## What it does

TableLight is a projected augmented reality tutoring system. A student places their homework on a table. An overhead camera sees the work surface. Gemini's Live API simultaneously hears the student speak and sees the table through a continuous video stream. The agent — a warm, encouraging tutor named **Lumi** — speaks explanations aloud while physically projecting graphs, diagrams, annotations, and step-by-step solutions directly onto the desk surface using a mini projector.

**The core interaction:** A student points at an equation and says, *"Can you show me what this graph looks like?"* Lumi locates the equation on the page, generates a graph, and projects it onto the table right next to the problem — while simultaneously explaining the concept. The student's eyes never leave their desk.

**What Lumi can project:**
- 📈 **Mathematical graphs** — plot any function with labeled axes
- 📝 **Step-by-step solutions** — reveal one step at a time with advance controls
- 🔬 **Chemistry molecules** — atom-bond diagrams with element labels
- 📐 **Geometry constructions** — points, lines, circles, arcs on a coordinate grid
- 🔢 **Number lines** — with highlighted points and solution ranges
- 🖼️ **AI-generated images** — diagrams, illustrations, and collaborative story scenes (Gemini image generation)
- 📋 **Markdown explanations** — formatted text with headers, bold, math notation
- 🃏 **Flashcards** — front/back cards with flip interaction
- 🎵 **Background music** — AI-generated study music via Google's Lyria model
- ✏️ **Annotations & highlights** — labels and color regions on the work surface

Lumi adapts to any school subject — math, science, language, history — detecting the subject from what's on the table. It offers hints before answers, asks follow-up questions, celebrates progress, and can even proactively spot errors in the student's writing between frames.

The system also supports a **screen overlay mode** for development and situations without a projector, compositing overlays onto the rectified camera view on a laptop screen.

## How we built it

### Architecture: Cloud Brain, Local Body

The system is split into two components connected by WebSocket:

**Cloud Run backend** — A FastAPI server hosting the Gemini Live session. It establishes a bidirectional connection using `client.aio.live.connect()` from the raw `google-genai` SDK. Audio and video are sent as separate concurrent streams (not serialized through a queue). Tool calls from Gemini are executed in the receive loop, and results are sent back immediately. The backend handles session resumption and context window compression for long study sessions.

**Local edge client** — A Python asyncio application that captures the camera feed (via IP Webcam), runs ArUco marker detection and homography calibration (OpenCV), captures microphone audio, and sends rectified frames + PCM audio to the backend over WebSocket. It receives audio responses (played through speakers) and tool results (rendered via matplotlib, warped to projector coordinates via homography, displayed on the projector).

### The Coordinate System Trick

This is the key technical insight. Gemini's vision returns spatial information in a normalized 0–1000 coordinate space. Our camera-to-table homography also maps to a normalized space. By aligning these two systems, **Gemini's understanding of where things are on the table maps directly to where the projector should place overlays**. The AI reasons about physical space without any explicit spatial programming.

### Binary WebSocket Protocol

The client–backend connection uses a binary WebSocket protocol with typed message frames:
- `0x01` + PCM audio (client → backend)
- `0x02` + JPEG frame (client → backend)
- `0x03` + PCM audio (backend → client, Gemini speech)
- JSON frames for text, tool results, transcripts, and notifications

This binary protocol saved ~344ms per round-trip compared to base64-encoding audio and images in JSON.

### Tool Schema Auto-Generation

Rather than manually writing JSON schemas for Gemini's function calling, we built `function_to_declaration()` — a utility that auto-generates Gemini-compatible tool schemas from Python function signatures, type hints, and Google-style docstrings. Same developer experience as ADK, zero dependency.

### Renderer Registry

Each overlay type (graph, geometry, chemistry, flashcard, etc.) is a self-contained Python module in `client/renderer/`. A registry auto-discovers renderers at startup, making it trivial to add new visualization types — write one file, it appears as a tool option.

### Key Technologies
- **Model:** `gemini-2.5-flash-native-audio` via the Live API for simultaneous voice + vision
- **Image generation:** Gemini image generation with enhance mode for incorporating student drawings
- **Music generation:** Google Lyria model for AI-generated background study music
- **Calibration:** ArUco fiducial markers for camera and projector homography
- **Rendering:** matplotlib on black backgrounds (projectors add light — black = transparent)

## Challenges we ran into

### The ADK Latency Wall

Our first implementation used the Google ADK (Agent Development Kit) with its `LiveRequestQueue`. We quickly discovered that ADK serializes all input — audio, video, and text — into a single FIFO queue. Video frames are large and slow to process, so audio packets would queue behind them, adding **~5 seconds of speech-to-transcription latency**. The tutor couldn't respond to "wait, go back" for five seconds. That's not a conversation — it's a voicemail.

We migrated to the raw `google-genai` SDK, which natively supports separate `audio=` and `video=` parameters in `session.send_realtime_input()`. Audio and video flow as independent concurrent streams. Latency dropped to **~1.6 seconds** end-to-end. The backend went from 437 lines of workarounds to ~250 lines of clean code.

### Projector Perspective Warp

A projector mounted at an angle distorts its output — rectangles become trapezoids. We needed a second homography (in addition to the camera's) to pre-warp overlay images so they appear undistorted on the table surface. Getting the axis order right (the classic y-before-x vs. x-before-y confusion in OpenCV) took several debugging sessions.

### Keeping Gemini Grounded

When the camera view is partially occluded or blurry, Gemini can hallucinate problem content. We addressed this with explicit system prompt instructions ("say 'I can't see that clearly' rather than guess"), a fresh-view query tool so the agent can request an updated camera frame, and cached clean frames (without overlay projections) to prevent the AI from seeing its own output and creating feedback loops.

### Marker Occlusion

Students place hands, books, and papers over the calibration markers. We implemented homography caching — the last-good-detection is preserved so brief occlusion doesn't break the coordinate system. A grace period prevents flicker, and error overlays provide clear feedback when calibration is truly lost.

## Accomplishments that we're proud of

- **The "wow" moment works.** A student says something, and a graph physically appears on the table next to the equation. It genuinely feels like magic.
- **1,352 tests** covering the full stack — renderers, coordinate transforms, overlay state management, WebSocket protocol, session persistence, tool validation, and more.
- **1,593ms end-to-end latency** from student speech to projected overlay — fast enough for natural conversation.
- **12+ overlay types** from a single unified tool interface — the agent just calls `overlay` with different `content_type` values.
- **Multi-subject support** — math, science, language, history — with subject-appropriate visualizations.
- **Session persistence** — overlays auto-save and restore, so a student can take a break and come back.
- **No screen required.** The entire interaction happens through voice, vision, and physical projection. The student's eyes never leave their work.

## What we learned

- **Separate streams matter more than model speed.** The biggest latency improvement wasn't from a faster model — it was from not serializing audio behind video in a FIFO queue. Architecture > raw performance.
- **Binary protocols pay for themselves.** Switching from base64-in-JSON to typed binary WebSocket frames saved ~344ms per round-trip. For a real-time system, that's the difference between responsive and sluggish.
- **Gemini's spatial understanding is underrated.** The 0–1000 bounding box system maps cleanly to physical coordinates through homography. We didn't need custom spatial reasoning — Gemini already knows where things are.
- **Black backgrounds are projector-transparent.** This simple insight — projectors add light, they can't subtract it — drove our entire rendering pipeline. Every overlay renders bright content on black, and the black regions "disappear" on the table.
- **Tool design shapes agent behavior.** Consolidating 15+ tool functions into 3 well-structured tools (overlay, query, music) with clear action parameters dramatically improved Gemini's tool-calling accuracy. Fewer tools with richer schemas > many simple tools.
- **Kids don't need instructions.** In early testing, children immediately started talking to the table and pointing at things. The physical interface is intuitive in a way that screen-based chat never is.

## What's next

- **Multi-student support** — multiple cameras and projectors for classroom deployment
- **Handwriting recognition** — real-time OCR to understand what the student is writing, not just what's printed
- **Curriculum integration** — connect to textbook databases so Lumi knows which chapter the student is on
- **Portable kit** — miniaturize to a single unit (camera + projector + compute) that clips to any desk lamp
- **Accessibility** — audio-only mode for visually impaired students, high-contrast mode for low-vision

## Built with

- Python
- Google Gemini 2.5 Flash (native audio, Live API)
- Google GenAI SDK (`google-genai`)
- Google Cloud Run
- Vertex AI
- Google Lyria (music generation)
- FastAPI
- OpenCV
- NumPy
- matplotlib
- asyncio
- WebSockets
- PyAudio
