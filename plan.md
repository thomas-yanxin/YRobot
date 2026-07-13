# Reachy Mini — Live Video Chat (全双工实时视频对话 App)

> A full-duplex, low-latency, bilingual (中文 / English) voice+video conversational agent for
> **Reachy Mini**, running the `VAD → ASR → LLM → TTS` pipeline locally on Apple Silicon, with
> expressive motion, DOA-driven head orientation, and token-efficient vision.

## 1. Goals & hard requirements (from the user)

| # | Requirement | How we meet it |
|---|-------------|----------------|
| 1 | Full-duplex conversation (streaming in, auto endpointing, real-time reply, barge-in) | Concurrent stage threads + a conversation state machine; barge-in cancels TTS + motion |
| 2 | **End-to-end latency ≤ 1.5 s** (user's last word → first audio out) | Streaming ASR during speech, LLM clause-flush, Kokoro first-chunk TTS. Budget ≈ 0.5–1.1 s (§5) |
| 3 | Reachy-specific motion + **DOA** for emotional value | 100 Hz control loop, emotions library moves, DOA head look-at, state-driven idle/listen/think/speak |
| 4 | Motions must **match the language/content** (user or robot) | Bilingual intent→emotion mapping + LLM inline `<emo>` tags; nod on "yes/好的", tilt on questions, etc. |
| 5 | Motions within **safety range** — never exceed | `motion/safety.py` clamps every command to the documented joint limits (§4) |
| 6 | zh + en support | SenseVoice ASR (zh/en/code-switch), bilingual LLM prompt, Kokoro zh + en voices, language auto-detect |
| 7 | Pipeline = VAD-ASR-LLM-TTS, squeeze Mac compute (MLX / llama.cpp) | Local MLX for LLM/ASR/TTS; see stack (§3) |
| 8 | Video → high image demand, but **save tokens** | Vision gating: send a frame only on visual-question intent or scene change; downscale+JPEG; keep 1 keyframe; route to cloud VLM (§6) |

## 2. Hardware / platform facts (from Reachy Mini SDK, verified)

- App type: **Python app** (`ReachyMiniApp`), heavy on-robot/local compute — not a web-only Space.
- 6-DOF head (Stewart platform), rotating body, 2 antennas. Mic array with **DOA**, camera, speaker.
- Client/daemon split: daemon on the machine wired to the robot exposes REST + WS on `:8000`.
- Media (16 kHz float32 audio, BGR camera frames) via `mini.media.*`.
- Dev without hardware → `--sim` uses `reachy-mini[mujoco]` daemon, or our `FakeMini` shim.

### Key SDK calls we use
- Motion: `set_target(head=4x4, antennas=[r,l]rad, body_yaw=rad)` (100 Hz), `goto_target(...)` (blocking),
  `look_at_world(x,y,z,duration)`, `play_move(move, ...)`, `cancel_move()`, `enable_motors()`.
- Pose helper: `create_head_pose(x,y,z,roll,pitch,yaw, mm=, degrees=)`.
- Audio: `media.start_recording()/get_audio_sample()` (→(N,2) f32 16k), `media.push_audio_sample()`, `media.start_playing()`.
- DOA: `media.get_DoA() -> (angle_rad, is_speech)`. Convention: 0=left, π/2=front, π=right.
- Camera: `media.get_frame() -> (H,W,3) uint8 BGR`.
- Emotions: `RecordedMoves("pollen-robotics/reachy-mini-emotions-library").get(name)`.

## 3. Local model stack (Apple Silicon, validated)

| Stage | Model / repo | Library | Role |
|-------|--------------|---------|------|
| VAD | `snakers4/silero-vad` v5 (+ optional TEN-VAD) | `silero-vad` | 32 ms frame speech/silence |
| Turn detect | `livekit/turn-detector` (multilingual, optional) | onnxruntime | semantic endpoint to cut false cuts |
| ASR | `FunAudioLLM/SenseVoiceSmall` | `funasr` | zh/en/code-switch, RTF≈0.007 |
| LLM (text) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | `mlx-lm` server (OpenAI-compat) | fast streaming reply, strong zh |
| LLM (vision) | `Qwen-Ambassador/Qwen3.7-Plus` (cloud) or `MiniCPM-o-2.6` (local) | OpenAI client / ollama | only on keyframes → token-saving |
| TTS | `hexgrad/Kokoro-82M` (zh voice `zf_*`, en `af_*`) | `mlx-audio` | lowest time-to-first-audio, streams |
| AEC | WebRTC APM (optional) + gating | `webrtc-audio-processing` | stop TTS self-triggering VAD |

All heavy deps are **lazy-imported** inside each engine so the package imports and unit-tests run
without them; a `FakeMini` + stub engines let the pipeline run end-to-end in `--sim` for development.

## 4. Safety limits (clamped in `motion/safety.py`)

| Joint / axis | Range |
|---|---|
| Head pitch / roll | [-40°, +40°] |
| Head yaw | [-180°, +180°] |
| Body yaw | [-160°, +160°] |
| Head−body yaw delta | ≤ 65° |

`clamp_head_pose()` decomposes the 4×4 to rpy, clamps, recomposes. `clamp_body_yaw()` also enforces
the delta vs the current head yaw. The SDK clamps too — this is defense-in-depth and keeps our own
generated idle/DOA motion honest. Emotion-library moves are pre-authored within range; we still route
their target stream through the clamp when we drive `set_target` ourselves.

## 5. Latency budget (end-of-speech → first audio)

ASR runs *during* speech; LLM streams into TTS clause-by-clause; stages overlap. Critical path after
the user stops talking:

| Step | Budget |
|---|---|
| VAD endpoint + (semantic) turn confirm | 200–350 ms |
| ASR finalize last chunk (SenseVoice) | 50–150 ms |
| LLM time-to-first-token (Qwen3-4B 4bit, cached sys prompt) | 150–300 ms |
| First clause → TTS | overlaps |
| Kokoro time-to-first-audio | 100–300 ms |
| **Total critical path** | **~0.5–1.1 s** (headroom under 1.5 s) |

Levers if over budget: shorter VAD silence window + semantic turn; 4B not 8B; prompt-cache; flush TTS
on first clause; keep vision on cloud so local GPU stays free.

## 6. Token-efficient vision

1. **Gate on intent first** — only consider sending a frame when the user asks something visual
   (bilingual keyword/intent check: "看/这是什么/what is this/what am I holding" …).
2. **Scene-change filter** — 64×64 gray mean-abs-diff + `imagehash.phash` Hamming distance vs last
   kept frame; skip near-duplicates.
3. **Compress** — resize long edge → ~768 px, JPEG q75, base64.
4. **One keyframe in context** — keep a single "current visual" slot; drop the previous image (keep a
   short text description for continuity). Caps vision tokens at one image per visual turn.
5. Route vision turns to the **cloud** VLM; local box stays free for ASR/LLM/TTS latency.

## 7. Concurrency / architecture

```
 mic ─► AudioIn ─► VAD/Turn ─► ASR ──► Orchestrator ──► LLM(stream) ─► clause ─► TTS ─► AudioOut ─► speaker
                     │                    │  ▲                                   │
                  barge-in            state machine        Vision(gated) ────────┘ (keyframe)
                     │                    │
                     ▼                    ▼
                MotionController  ◄── intents (DOA look-at, listen/think/speak, emotion moves)
                 (single 100 Hz owner of set_target — control-loop skill rule)
```

- **One** thread owns `set_target` (per control-loop skill). Others enqueue motion *intents*.
- Conversation states: `IDLE → LISTENING → THINKING → SPEAKING` (+ `INTERRUPTED`). Each maps to a
  motion mood (idle breathing, attentive lean + DOA turn, thoughtful, subtle speak wobble).
- Barge-in: while `SPEAKING`, VAD still runs (echo-gated); confirmed human speech → `cancel_move()`
  + stop TTS → `LISTENING`.

## 8. Project layout

```
reachy_mini_live_chat/          package (entrypoint reachy_mini_apps → main:LiveChatApp)
  main.py            ReachyMiniApp subclass, wires Pipeline, optional web UI
  config.py          env-driven dataclass config
  bus.py             ConvState enum, thread-safe queues/events
  pipeline.py        full-duplex orchestrator + state machine
  audio/  io.py vad.py aec.py
  asr/    funasr_asr.py
  llm/    client.py router.py prompts.py
  tts/    kokoro_tts.py
  vision/ gating.py
  motion/ safety.py controller.py emotions.py doa.py
  sim/    fake_mini.py         mock ReachyMini + media for hardware-free dev
static/   index.html style.css main.js   web UI (transcript, video, settings)
tests/    pure-logic unit tests
scripts/  setup_mac.sh
pyproject.toml  README.md(+HF tag)  .env.example  index.html/style.css(HF landing)
```

## 9. Milestones

1. Scaffold (pyproject/README/env). 2. Core (config/bus/sim). 3. Audio (io/vad/aec).
4. Engines (asr/llm/tts). 5. Motion (safety/controller/emotions/doa). 6. Vision + orchestrator + entry
+ web UI. 7. Tests + 3.12 venv install + sim smoke test.

## 10. Open assumptions (proceeding with sensible defaults)

- **Text LLM local (Qwen3-4B MLX), vision cloud (Qwen3.7-Plus)** — best latency + token saving. Both
  swappable via `.env` (`LLM_BASE_URL`, `VISION_BASE_URL`, model ids).
- Reachy **Lite** (USB, full Mac compute) is the primary target; wireless works but vision→cloud.
- Dev happens in `--sim`/FakeMini; real-robot run needs the daemon + downloaded models on a 3.12 venv.
