# Orly — Devpost Submission

## Inspiration

Every AI today lives behind a screen. You talk to it in a text box, it replies in a text box. Even the "multimodal" ones — you point your camera at something and then look at your phone for the answer. The digital and physical worlds stay separate.

We asked: **what if the AI's output existed in the physical world instead?**

A child doing homework looks at their paper, not a screen. A family exploring a topic together gathers around a table. A maker working on a project has their hands full. In all these cases, the natural workspace is a physical surface — and the AI should meet you there.

When we saw the Gemini Live Agent Challenge call for agents that "see, hear, and speak," we realized Gemini's spatial understanding — its ability to return bounding boxes in a normalized coordinate space — could map directly to physical table coordinates through calibrated homography. The AI doesn't just *see* what's on the table. It knows *where* things are, and can place its responses *right next to* the relevant content. That was the spark.

**Orly** (from **O**ve**RL**a**Y**) is what happened next.

## What it does

Orly is a real-time AI agent that lives on your desk. A camera sees the table surface. Gemini's Live API simultaneously hears you speak and sees what's in front of you through a continuous video stream. Orly speaks back naturally while projecting images, diagrams, annotations, stories, and more directly onto the physical surface using a mini projector.

It's not just a tutoring system — it's a seamless blend of digital and material.

**Help with homework.** Point at an equation and say, *"Can you show me what this graph looks like?"* Orly locates the equation on the page, generates a graph, and projects it onto the table right next to the problem — while explaining the concept aloud.

**Create together.** Say, *"Let's make a story about a dragon."* Orly generates an illustrated scene and projects it onto the table. *"Now add a castle in the background."* The scene evolves, projected in front of you, page by page.

**Explore the world.** Place a leaf on the table: *"What kind of tree is this from?"* Orly identifies it and projects a diagram of the tree, its habitat, and fun facts — all visible without picking up a phone.

**Set the mood.** *"Play some calm music while I work."* AI-generated background music starts playing through the speakers (Google Lyria).

**What Orly can project:**
- 📈 **Mathematical graphs** — plot any function with labeled axes
- 📝 **Step-by-step solutions** — reveal one step at a time with advance controls
- 🖼️ **AI-generated images** — illustrations, scenes, diagrams (Gemini image generation)
- 🎬 **AI-generated videos** — short clips projected onto the surface (Google Veo)
- 🔬 **Chemistry molecules** — atom-bond diagrams with element labels
- 📐 **Geometry constructions** — points, lines, circles, arcs on a coordinate grid
- 🔢 **Number lines** — with highlighted points and solution ranges
- 📋 **Markdown explanations** — formatted text with headers, bold, math notation
- 🃏 **Flashcards** — front/back cards with flip interaction
- 🎵 **Background music** — AI-generated via Google Lyria
- ✏️ **Annotations & highlights** — labels and color regions on the work surface

Orly adapts to any subject and any context — math, science, language, history, creative projects, or just curiosity. No screen required. The entire interaction happens through voice, vision, and physical projection.

The system also supports a **screen overlay mode** for development and situations without a projector, compositing overlays onto the rectified camera view on a laptop screen.

## How we built it

### Architecture: Cloud Brain, Local Body

The system is split into two components connected by WebSocket:

**Cloud Run backend** — A FastAPI server hosting the Gemini Live session. It establishes a bidirectional connection using `client.aio.live.connect()` from the raw `google-genai` SDK. Audio and video are sent as separate concurrent streams (not serialized through a queue). Tool calls from Gemini are executed in the receive loop, and results are sent back immediately. The backend handles session resumption and context window compression for long sessions.

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
- **Image generation:** Gemini image generation with enhance mode for incorporating user drawings
- **Music generation:** Google Lyria model for AI-generated background music
- **Video generation:** Google Veo model for AI-generated video clips
- **Calibration:** ArUco fiducial markers for camera and projector homography
- **Rendering:** matplotlib on black backgrounds (projectors add light — black = transparent)

## Challenges we ran into

### The ADK Latency Wall

Our first implementation used the Google ADK (Agent Development Kit) with its `LiveRequestQueue`. We quickly discovered that ADK serializes all input — audio, video, and text — into a single FIFO queue. Video frames are large and slow to process, so audio packets would queue behind them, adding **~5 seconds of speech-to-transcription latency**. You couldn't say "wait, go back" and get a response for five seconds. That's not a conversation — it's a voicemail.

We migrated to the raw `google-genai` SDK, which natively supports separate `audio=` and `video=` parameters in `session.send_realtime_input()`. Audio and video flow as independent concurrent streams. Latency dropped to **~1.6 seconds** end-to-end. The backend went from 437 lines of workarounds to ~250 lines of clean code.

### Projector Perspective Warp

A projector mounted at an angle distorts its output — rectangles become trapezoids. We needed a second homography (in addition to the camera's) to pre-warp overlay images so they appear undistorted on the table surface. Getting the axis order right (the classic y-before-x vs. x-before-y confusion in OpenCV) took several debugging sessions.

### Keeping Gemini Grounded

When the camera view is partially occluded or blurry, Gemini can hallucinate content. We addressed this with explicit system prompt instructions ("say 'I can't see that clearly' rather than guess"), a fresh-view query tool so the agent can request an updated camera frame, and cached clean frames (without overlay projections) to prevent the AI from seeing its own output and creating feedback loops.

### Marker Occlusion

People place hands, books, and objects over the calibration markers. We implemented homography caching — the last-good-detection is preserved so brief occlusion doesn't break the coordinate system. A grace period prevents flicker, and error overlays provide clear feedback when calibration is truly lost.

## Accomplishments that we're proud of

- **The "wow" moment works.** You say something, and a graph — or an image, or a story illustration — physically appears on the table. It genuinely feels like magic.
- **1,352 tests** covering the full stack — renderers, coordinate transforms, overlay state management, WebSocket protocol, session persistence, tool validation, and more.
- **1,593ms end-to-end latency** from speech to projected overlay — fast enough for natural conversation.
- **12+ overlay types** from a single unified tool interface — the agent just calls `overlay` with different `content_type` values.
- **Multi-modal creation** — images, videos, music, and code generation all projected onto one surface.
- **Session persistence** — overlays auto-save and restore, so you can take a break and come back.
- **No screen required.** The entire interaction happens through voice, vision, and physical projection. Your eyes never leave the table.

## What we learned

- **Separate streams matter more than model speed.** The biggest latency improvement wasn't from a faster model — it was from not serializing audio behind video in a FIFO queue. Architecture > raw performance.
- **Binary protocols pay for themselves.** Switching from base64-in-JSON to typed binary WebSocket frames saved ~344ms per round-trip. For a real-time system, that's the difference between responsive and sluggish.
- **Gemini's spatial understanding is underrated.** The 0–1000 bounding box system maps cleanly to physical coordinates through homography. We didn't need custom spatial reasoning — Gemini already knows where things are.
- **Black backgrounds are projector-transparent.** This simple insight — projectors add light, they can't subtract it — drove our entire rendering pipeline. Every overlay renders bright content on black, and the black regions "disappear" on the table.
- **Tool design shapes agent behavior.** Consolidating 15+ tool functions into 3 well-structured tools (overlay, query, music) with clear action parameters dramatically improved Gemini's tool-calling accuracy. Fewer tools with richer schemas > many simple tools.
- **Physical interfaces are intuitive.** In early testing, children immediately started talking to the table and pointing at things. No instructions needed. The physical interface is natural in a way that screen-based chat never is.

## What's next

- **Multi-user support** — multiple cameras and projectors for shared spaces and classrooms
- **Real-time handwriting recognition** — OCR to understand what the user is writing, not just what's printed
- **Interactive mini-programs** — agent-authored Python programs that react to camera input in real time (object tracking, zone triggers)
- **Portable kit** — miniaturize to a single unit (camera + projector + compute) that clips to any desk lamp
- **Accessibility** — audio-only mode for visually impaired users, high-contrast mode for low-vision

## Built with

- Python
- Google Gemini 2.5 Flash (native audio, Live API)
- Google GenAI SDK (`google-genai`)
- Google Cloud Run
- Vertex AI
- Google Lyria (music generation)
- Google Veo (video generation)
- FastAPI
- OpenCV
- NumPy
- matplotlib
- asyncio
- WebSockets
- PyAudio
