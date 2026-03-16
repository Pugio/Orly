# Gemini Live API Stability Audit — Suggested Changes

Based on thorough research of GitHub issues (python-genai, adk-python, livekit/agents, live-api-web-console), Google AI forums, production codebases (LiveKit, project-livewire, official Google examples), and analysis of our `backend/main.py`.

---

## 1. CRITICAL: Model May Not Support All Features We're Using

**Current:** `gemini-2.5-flash-native-audio-preview-12-2025` (agent.py:170)

**Problem:** Native-audio-dialog models have known issues with function calling. Multiple reports confirm that `*-native-audio-*` models sometimes:
- Don't make function calls at all (google-gemini forum, python-genai #803)
- Speak tool metadata aloud ("tools_output", "tool_call") instead of executing (python-genai #789)
- Hang indefinitely when a query should trigger a function call

We have 19 tools. This model variant is optimized for natural conversation, not heavy tool use.

**Suggested change:** Test with `gemini-2.0-flash-live-001` or `gemini-2.5-flash-preview-native-audio-dialog` as fallbacks. The official docs state: "Audio inputs and audio outputs negatively impact the model's ability to use function calling." Consider whether a non-native-audio model with separate TTS would be more reliable for our tool-heavy use case.

---

## 2. CRITICAL: Google AI API Gets Vertex-Only Features on Reconnect

**Current:** Lines 262-278 — when reconnecting with a resumption handle, the code rebuilds `LiveConnectConfig` and copies `config.context_window_compression` from the original config. But on Google AI (non-Vertex), `config.context_window_compression` is `None` since it was never set.

**However:** The resumption config clone (line 277) doesn't include `proactivity`, so if we're on Vertex that field is lost on reconnect. More importantly:

**The real risk:** `session_resumption` is being set on reconnect even on Google AI (line 265 — `sr_kwargs` is built regardless of `is_vertex`, only `transparent` is conditional). If the Google AI server doesn't support session resumption, sending a handle could trigger the 1008 "Operation is not implemented" error.

**Suggested change:**
```python
# Only attempt session resumption on Vertex AI
if resumption_handle["handle"] and is_vertex:
    # ... build resume config
else:
    resume_cfg = config  # fresh session on Google AI
```

---

## 3. CRITICAL: No Context Window Compression on Google AI

**Current:** Compression is Vertex-only (lines 237-243). On Google AI, there's no compression.

**Problem:** Audio+video sessions have a **2-minute context window limit** without compression. Audio = ~25 tokens/sec, video = ~258 tokens/sec per frame. At 1 FPS + audio, that's 283 tokens/sec = 128k tokens in ~7.5 minutes. Without compression, the session will be forcibly terminated.

**This is likely our #1 source of crashes on Google AI.** The server runs out of context and drops the connection.

**Suggested changes:**
- Context window compression IS available on Google AI as of late 2025 (confirmed in the cookbook examples and Google AI docs). The comment on line 234-236 may be outdated. Test enabling it.
- If it truly doesn't work on Google AI, implement client-side mitigation: reduce video frame rate to 0.5 FPS or lower, stop sending video when the student is just talking (no visual changes on table).

---

## 4. HIGH: Tool Call Gate Has a Race Condition

**Current:** Lines 319-326 — the sender waits on `tool_call_pending`, yields with `await asyncio.sleep(0)`, then checks again.

**Problem:** There's a TOCTOU race. Between the `sleep(0)` yield and the `is_set()` check, the receiver could have already cleared the gate AND the sender could still slip through. The `asyncio.sleep(0)` helps but doesn't guarantee ordering. Under high audio frame rates, frames can still leak through.

**Suggested changes:**
- Use a more robust pattern: instead of checking the event twice, use a single `await tool_call_pending.wait()` and accept that one frame might slip through (which the server tolerates). OR:
- Queue audio/video frames instead of sending directly. The sender puts frames in an `asyncio.Queue`, and a separate dispatcher task reads from the queue while respecting the gate. This is what LiveKit does.
- Consider dropping ALL queued frames when a tool call arrives (not just gating new ones). Old frames in flight can still trigger 1008.

---

## 5. HIGH: No `audioStreamEnd` Signal on Silence

**Current:** We stream audio continuously from the client.

**Problem:** The Gemini Live API expects `audioStreamEnd` when the mic pauses for >1 second (with automatic VAD). Without it, audio gets stuck in a server-side buffer. This can cause:
- Delayed transcription
- VAD becoming overly sensitive after tool calls
- Accumulated stale audio triggering false speech detection

**Suggested change:** Implement client-side silence detection. When silence exceeds ~1 second, send `audioStreamEnd`. Resume with fresh audio when speech resumes. This also saves token budget.

---

## 6. HIGH: Retry Loop Doesn't Reset Attempt Counter

**Current:** Lines 618-656 — `for attempt in range(1, max_retries + 1)` never resets.

**Problem:** If the session runs successfully for 9 minutes, gets a `go_away`, reconnects, runs for another 9 minutes, gets another `go_away`... after 10 clean reconnects, we give up. The attempt counter should reset after a successful period.

**Suggested change:** Reset `attempt = 0` after receiving the first successful message from Gemini (or after N seconds of stable connection). LiveKit resets retry count on first successful receive.

---

## 7. HIGH: `go_away` Doesn't Use `time_left` Field

**Current:** Lines 410-413 — we see `go_away` and immediately return (triggering reconnect).

**Problem:** The `go_away` message includes a `time_left` field telling us how long until the connection is forcibly terminated. We should use this time to:
1. Finish any in-progress tool call
2. Ensure we have the latest resumption handle
3. Gracefully close the session

**Suggested change:**
```python
if hasattr(msg, "go_away") and msg.go_away:
    time_left = getattr(msg.go_away, "time_left", None)
    logger.info("Received go_away — time_left=%s, reconnecting", time_left)
    # If time_left is very short, return immediately
    # Otherwise, let current turn complete before reconnecting
    return
```

---

## 8. MEDIUM: Reconnect Rebuilds Config Incompletely

**Current:** Lines 268-278 — the resume config manually copies fields from the original config.

**Problem:** If new fields are added to the original config (like `proactivity`, `enable_affective_dialog`, or future fields), the resume config won't include them. This is fragile.

**Suggested change:** Use a pattern that preserves all original config fields:
```python
# Build resume config by overlaying session_resumption onto original
resume_kwargs = {
    k: v for k, v in config.to_dict().items() if v is not None
}
resume_kwargs["session_resumption"] = types.SessionResumptionConfig(**sr_kwargs)
resume_cfg = types.LiveConnectConfig(**resume_kwargs)
```

Or simpler: since the SDK docs say "you can change config parameters except the model when resuming," just set `session_resumption` on the existing config object if the SDK allows mutation.

---

## 9. MEDIUM: No Exponential Backoff

**Current:** Fixed 0.2s (transient) or 1.0s (unknown) sleep between retries.

**Problem:** If Gemini is having a regional outage, rapid retries at 0.2s intervals for 10 attempts will:
- Burn through rate limits
- Potentially trigger `OVERLOADED_TOO_MANY_RETRIES_PER_REQUEST` (a known server-side error)
- Not give the server time to recover

**Suggested change:** Exponential backoff with jitter:
```python
import random
base_delay = 0.2 if is_transient else 1.0
delay = min(base_delay * (2 ** (attempt - 1)), 30) + random.uniform(0, 0.5)
await asyncio.sleep(delay)
```

---

## 10. MEDIUM: No Error Classification for 500 Errors

**Current:** Only 1008 and 1011 are classified as transient (line 635).

**Problem:** Server-side 500 errors ("Internal error encountered") are a commonly reported transient failure. They should get the fast-retry treatment too. Similarly, 1006 (abnormal closure) is often transient.

**Suggested change:**
```python
is_transient = any(code in exc_str for code in ("1006", "1008", "1011", "500", "503"))
```

---

## 11. MEDIUM: Video Frame Rate Not Throttled

**Current:** We send every video frame the client sends us.

**Problem:** The Gemini Live API processes video at 1 FPS regardless of send rate. Extra frames are queued and consume bandwidth + tokens. At 258 tokens per frame, sending 5 FPS means 5x the token burn for no benefit.

**Suggested change:** Server-side frame rate limiting:
```python
last_video_sent = 0
MIN_VIDEO_INTERVAL = 1.0  # seconds

# In send_from_client video handling:
now = time.monotonic()
if now - last_video_sent < MIN_VIDEO_INTERVAL:
    continue  # drop frame
last_video_sent = now
```

Also consider adaptive frame rate: send more frames when the table content changes (homework placed), fewer when static.

---

## 12. MEDIUM: Session Resumption Handle May Be None Initially

**Current:** Lines 401-408 — we store handle when it arrives.

**Problem:** `SessionResumptionUpdate.new_handle` is often `None` for the first several messages. The handle only appears after the first model response is fully streamed. If we crash before getting a handle, we can't resume.

**This is expected behavior** but our code handles it correctly (the `if update.new_handle:` guard). Just noting it for awareness — early crashes (first 5-10 seconds) will always start fresh sessions.

**No change needed**, but consider logging when we get the first handle so we know the session is "resumable."

---

## 13. LOW: `send_client_content` and `send_realtime_input` Interleaving

**Current:** We use `send_realtime_input` for audio/video (lines 330, 339) and `send_client_content` for text messages and notifications (lines 348, 363).

**Problem:** The official docs warn: "Do not interleave `send_client_content` and `send_realtime_input`." While this may work in practice, it's technically unsupported and could be a source of instability.

**Suggested change:** Consider using `send_realtime_input(text=...)` for text messages instead of `send_client_content`, OR use `send_client_content` for everything (but this would lose the real-time streaming benefit for audio/video). The safest approach is to use `send_realtime_input` exclusively for all user input during a live session, and only use `send_client_content` for injecting historical context.

---

## 14. LOW: Single Gemini Client Instance

**Current:** Lines 160-177 — one cached `genai.Client` per process.

**Problem:** The SDK has known issues with aiohttp session reuse under concurrency (python-genai #1074, #1083). If multiple WebSocket sessions share one client, connector limits may be too low, causing `ServerDisconnectedError`.

**Suggested change:** For now this is fine (we likely only have one session at a time), but if we ever scale to multiple concurrent sessions, create a client per session or increase aiohttp connector limits.

---

## 15. LOW: No Keepalive / Heartbeat

**Current:** No periodic ping to Gemini during quiet periods.

**Problem:** If the student is silently reading for several minutes with no audio/video activity, the WebSocket may be closed by intermediate proxies or the server itself.

**Suggested change:** Send periodic silent audio frames or a minimal video frame every 30 seconds during idle periods. Alternatively, rely on the WebSocket library's built-in ping/pong, but verify the Gemini server responds to pings.

---

## Summary: Priority Order

| # | Priority | Issue | Likely Impact |
|---|----------|-------|---------------|
| 3 | CRITICAL | No context compression on Google AI | Sessions die after ~2-7 min |
| 1 | CRITICAL | Native audio model + heavy tool use | Tool calls fail/hang |
| 2 | CRITICAL | Session resumption sent on Google AI | 1008 on reconnect |
| 4 | HIGH | Tool call gate race condition | Sporadic 1008 errors |
| 5 | HIGH | No audioStreamEnd on silence | Stale audio, bad VAD |
| 6 | HIGH | Retry counter never resets | Session dies after 10 reconnects |
| 7 | HIGH | go_away doesn't use time_left | Abrupt reconnect |
| 8 | MEDIUM | Incomplete config rebuild on reconnect | Lost features after reconnect |
| 9 | MEDIUM | No exponential backoff | Rate limit triggers |
| 10 | MEDIUM | Missing error code classification | Slow retry for transient errors |
| 11 | MEDIUM | No video frame rate throttling | Token waste, faster context exhaustion |
| 13 | LOW | client_content + realtime_input mixing | Potential instability |
| 14 | LOW | Single shared client | Concurrency risk (future) |
| 15 | LOW | No keepalive during idle | Idle timeout disconnects |

---

## Quick Wins (< 30 min each)

1. **Test enabling context_window_compression on Google AI** — just remove the `if is_vertex` guard and see if it works. The cookbook examples don't gate it.
2. **Guard session resumption to Vertex-only** — add `and is_vertex` to the reconnect handle check.
3. **Reset retry counter** after successful message receive.
4. **Add 500/503/1006 to transient error codes.**
5. **Add exponential backoff with jitter.**
6. **Throttle video to 1 FPS server-side.**

## Sources

- [python-genai issues](https://github.com/googleapis/python-genai/issues) — #803, #789, #872, #1074, #1083, #1224, #1285, #1490, #1710, #1720, #1859, #1893, #2117
- [Google AI Forum: 1008 Error Thread (57+ replies)](https://discuss.ai.google.dev/t/gemini-live-api-websocket-error-1008/114644)
- [Google AI Forum: Hard-Won Patterns March 2026](https://discuss.ai.google.dev/t/hard-won-patterns-for-building-voice-apps-with-gemini-live-march-2026/128155)
- [LiveKit agents plugin](https://github.com/livekit/agents) — the only production-grade reconnection implementation
- [project-livewire](https://github.com/heiko-hotz/project-livewire) — best error categorization
- [Google AI: Live API Limits](https://firebase.google.com/docs/ai-logic/live-api/limits-and-specs)
- [Google AI: Session Management](https://ai.google.dev/gemini-api/docs/live-session)
- [Google AI: Best Practices](https://ai.google.dev/gemini-api/docs/live-api/best-practices)
