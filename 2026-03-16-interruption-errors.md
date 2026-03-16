# Session Interruption & Reconnect Errors — March 16, 2026

## Summary

After migrating from ADK to the raw `google-genai` SDK, the Gemini Live session repeatedly drops and reconnects. The model starts speaking, gets interrupted, and the session dies — causing a reconnect loop. Overlays flash on screen then vanish. The system is unusable for sustained tutoring.

---

## Timeline: How We Got Here

### Day 1 (March 15) — ADK-based architecture

The original backend used the **Google Agent Development Kit (ADK)**. The ADK provides `Runner` and `LiveRequestQueue` for managing Live API sessions. Audio, video, and text all flow through a single FIFO queue.

**Problem: ~5s speech-to-transcription latency.** The student would speak and the model wouldn't start responding for 5 seconds. Root cause: `LiveRequestQueue` serializes all input into one queue. A large JPEG video frame (100–200KB) would block the queue, and audio chunks behind it would wait. Since video frames arrive at 1fps and audio at 50 chunks/sec, audio was constantly stuck behind video.

**Workarounds attempted (all hacky):**
- Monkey-patched `LiveRequestQueue` to prioritize audio over video
- Tried separate queues for audio and video, feeding into the ADK runner
- Added `before_tool_callback` to intercept tool calls before ADK processed them
- Wrapped tool execution with a `pending_tool_calls` queue to forward results to the client

None of these were clean. The backend was 437 lines of workarounds and monkey-patches. The FIFO architecture was fundamental to ADK — we couldn't fix it without replacing it.

### Day 2 morning (March 16) — Migration to raw genai SDK

**Commit `845e934`: "Replace ADK with raw google-genai SDK for Gemini Live sessions"**

The raw `google-genai` SDK's `client.aio.live.connect()` natively supports separate audio and video streams via `session.send_realtime_input(audio=blob)` and `session.send_realtime_input(video=blob)`. No FIFO queue. Audio and video flow independently.

Key changes:
- `backend/main.py`: Direct `client.aio.live.connect()` instead of `Runner`/`LiveRequestQueue`
- `backend/agent.py`: ADK `Agent` class → pure config + `function_to_declaration()` for JSON tool schemas from Python signatures
- All monkey-patching removed
- Session resumption + context window compression configured natively (Vertex AI only)
- Auto-reconnect loop with up to 10 retries

**Result:** Backend went from 437 lines of workarounds to ~250 lines of clean code. Audio latency dropped significantly.

**Also switched to a local webcam** (previously used IP Webcam phone camera over WiFi). This eliminated network latency on the video capture path and frame staleness issues we'd been fighting.

### Day 2 follow-up commits:
- `dfc0a8c`: Binary WebSocket protocol (1-byte prefix instead of JSON+base64) + VAD tuning (prefix_padding 200→50ms, silence_duration 500→300ms) + smaller audio chunks (20ms)
- Various feature additions: music gen, video gen, code gen, 1099+ tests

### Day 2 evening — Session drop/reconnect loop appears

With the new SDK integration working, we started end-to-end testing. The session kept dying.

---

## Error Manifestations

### Error 1: 1008 Policy Violation during tool calls

**Backend log:**
```
INFO:backend.main:Tool call: project_overlay({...})
ERROR:backend.main:Error in send_from_client
websockets.exceptions.ConnectionClosedError: received 1008 (policy violation)
  Operation is not implemented, or supported, or enabled.
```

**Root cause:** The Gemini Live API enters a **blocking state** when it sends a `tool_call`. If the client continues streaming audio/video via `send_realtime_input()` while the tool call is pending (before `send_tool_response()` is sent), the server terminates the connection with 1008.

**Fix applied:** Added an `asyncio.Event` gate (`tool_call_pending`) in `backend/main.py`. The sender task waits on this gate before sending audio/video. The gate closes when a `tool_call` arrives and reopens after `send_tool_response()` completes.

### Error 2: Session ends after every interruption

**Backend log:**
```
INFO:backend.main:INTERRUPTED
INFO:backend.main:Gemini WS closed: code=None reason=None
INFO:backend.main:receive_from_gemini exited (client_done=False)
INFO:backend.main:Task finished: ['receiver'] (still running: ['sender'])
INFO:backend.main:Session ended cleanly (attempt 1) — reconnecting
```

**Client log:**
```
[Lumi] Wow, a rocket ship! That sounds amazing! I'll start by creating...
(Reconnecting to Gemini...)
```

**Root cause:** The SDK's `session.receive()` method is designed for **single-turn** interactions. It breaks out of its internal loop on `turn_complete`:

```python
# From google/genai/live.py
async def receive(self):
    while result := await self._receive():
        if result.server_content and result.server_content.turn_complete:
            yield result
            break  # ← exits after one turn!
        yield result
```

Our code used a single `async for msg in session.receive()` expecting it to be a persistent stream across the entire session. When the model finished a turn (or was interrupted, which also completes the turn), the iterator would exhaust, we'd treat it as the session closing, and reconnect.

**Fix applied:** Wrapped the `async for` in an outer `while not client_done.is_set()` loop so we keep calling `session.receive()` for each new turn. The outer loop only breaks if the underlying WebSocket actually closes (detected by checking `ws.close_code`).

### Error 3: Overlays cleared on every interruption

**Client log:**
```
[TableLight] Interrupted — overlays cleared.
```

An image would generate and display, then vanish immediately when an interruption occurred.

**Root cause:** The client's `on_interrupted` handler called `overlay_state.clear()`, wiping all overlays. Interruptions are normal Live API behavior (user spoke while model was talking) and should not affect the display.

**Fix applied:** Changed `on_interrupted` to a no-op. Overlays now persist until explicitly removed by a tool call.

### Error 4: Audio echo causing false interruptions

**Client log:**
```
[Student]  I can generate a beautiful picture for your story. It might take a minute.
```

The model's own output was transcribed as "student" input — the speaker audio was being picked up by the microphone.

**Root cause:** No echo cancellation. The mic captures everything including speaker playback. Gemini's VAD detects the speaker audio as "user speech," triggers an interruption, and the model hears garbled versions of its own output.

**Status:** Not yet fixed. Potential solutions:
1. **Software echo cancellation** — suppress mic input while audio is playing (half-duplex)
2. **Hardware** — use headphones or a directional mic
3. **OS-level** — some audio drivers provide echo cancellation
4. **Gate the mic** — mute mic input when `AudioPlayer` is actively writing to the stream

### Error 5: Model name deprecation

**Model used:** `gemini-2.5-flash-native-audio-latest`

The `-latest` alias may resolve to `gemini-2.5-flash-native-audio-preview-09-2025`, which is scheduled for removal on **March 19, 2026** (3 days away).

**Fix applied:** Changed to `gemini-2.5-flash-native-audio-preview-12-2025` (the current recommended model for Google AI Studio).

---

## Current State of Fixes

| Issue | Status | Fix |
|-------|--------|-----|
| 1008 during tool calls | ✅ Fixed | `asyncio.Event` gate blocks audio/video during tool execution |
| Session dies after turn | ✅ Fixed | Outer `while` loop around `session.receive()` |
| Overlays cleared on interrupt | ✅ Fixed | `on_interrupted` is now a no-op |
| Audio echo / false interrupts | ⚠️ Unfixed | Need echo cancellation or mic gating |
| Model name deprecation | ✅ Fixed | Updated to `preview-12-2025` |
| Better diagnostics | ✅ Added | Task name logging, WS close code/reason logging |

---

## Architecture Diff: ADK vs Raw SDK

### ADK (before)
```
Client → WebSocket → ADK Runner → LiveRequestQueue (FIFO) → Gemini
                                   ↑ audio, video, text all serialized
                                   ↑ video blocks audio (~5s latency)
```
- `LiveRequestQueue` serializes everything into one queue
- Monkey-patched to try to prioritize audio
- `before_tool_callback` + `pending_tool_calls` queue for tool forwarding
- 437 lines, fragile

### Raw SDK (after)
```
Client → WebSocket → Backend → session.send_realtime_input(audio=)  → Gemini
                              → session.send_realtime_input(video=)  →
                              → session.send_client_content(text=)    →
```
- Audio and video sent as independent concurrent streams
- Tool calls handled directly in the receive loop
- `function_to_declaration()` auto-generates schemas from Python functions
- ~280 lines, clean
- BUT: `session.receive()` is per-turn, not per-session (discovered the hard way)

---

## Remaining Risks

1. **Echo cancellation** is the biggest UX problem. Without it, the model frequently interrupts itself, especially during longer responses. This degrades the tutoring experience even if the session no longer dies.

2. **Session stability** on the Google AI (non-Vertex) API may still be fragile. The free API has undocumented rate limits and session duration caps. Vertex AI with session resumption would be more robust but requires GCP billing.

3. **`session.receive()` semantics** may change in future SDK versions. Our outer-loop workaround depends on the current behavior where `receive()` returns after `turn_complete`. If the SDK changes to be a persistent stream, the outer loop would still work (it would just never re-enter).
