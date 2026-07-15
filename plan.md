# Reachy Mini — Realtime Full-Duplex Audio-Visual Dialogue (MiniCPM-o 4.5 omni)

> Rebuilds the conversational core of `reachy_mini_live_chat` around a **remote end-to-end
> omni model** (MiniCPM-o 4.5, served by `llama.cpp-omni`'s `llama-omni-server`) over a
> **full-duplex WebSocket**, replacing the local `VAD→ASR→LLM→TTS` pipeline. Runs **on-robot
> (CM4, Wireless)**; all heavy inference is offloaded to the GPU server at
> `wss://10.0.16.187:8006`. Keeps the robot's DOA, expressive motion, and hard safety limits.

## 0. Decisions (confirmed with the user)

1. **Deployment**: app runs **on the robot's CM4** (Wireless). Direct mic-array/DOA, camera,
   speaker; cognition fully offloaded → CM4 only does I/O + WebSocket + 100 Hz motion. Light load.
2. **Scope**: **fully replace** the local `asr/`, `llm/`, `tts/`, `vision/gating` modules with the
   omni client. No local ML models remain. (Removes the mlx/funasr/kokoro/llama.cpp dependency.)
3. **Video**: **continuous ~1 fps** — one current camera frame attached to each 1 s audio chunk
   (matches the model's 1 Hz TDM design; LAN bandwidth is ample).

## 1. Hard requirements → how we meet them

| # | Requirement (from user) | How |
|---|---|---|
| 1 | 视听交互尽可能实时 (full-duplex, low latency) | End-to-end omni over WS in `full_duplex` mode: stream 1 s mic chunks + 1 frame; the model decides listen/speak at 1 Hz; audio deltas play as they arrive. No local ASR/LLM/TTS hops. |
| 2 | DOA 融入：说话时机器人依 DOA 转向说话人 | Kept **unchanged** — `MotionController` polls `mini.media.get_DoA()` at 10 Hz while a lightweight local VAD says a human is speaking, EMA-smoothed → head yaw (+ body yaw past ±45°), all clamped. |
| 3 | 必要时动作交互，且**务必保证物理安全** | Motion moods per conversation state + emotion gestures inferred from the omni **text** stream (`kind:"text"` deltas). **Every** command routed through `motion/safety.py` clamps (head pitch/roll ±40°, yaw ±180°, body ±160°, head−body Δ≤65°); SDK clamps again (defense-in-depth). Return-to-rest on interrupt/shutdown. |

## 2. Confirmed omni WS protocol (verified against `tc-mb/llama.cpp-omni` source)

Source of truth: `tools/server/ws_handler.cpp` + `protocol.cpp` (not just the cookbook).

- **Endpoint**: `wss://10.0.16.187:8006/backend` (auto-append `/backend` if absent). Self-signed
  cert → TLS verification disabled by default (`OMNI_TLS_INSECURE=1`). **One active session** at a
  time server-side.
- **Audio encoding**: base64 of **raw float32 little-endian PCM**, **mono, 16 kHz** (input WAV is
  hardcoded 16 k/mono/IEEE-float). Output audio deltas are float32 PCM too.
- **`session.init`** (client→server):
  ```json
  {"type":"session.init","payload":{
     "mode":"full_duplex", "use_tts":true,
     "system_prompt":"…",
     "voice":{"ref_audio":"<b64 wav>"},          // optional voice-clone
     "config":{"temperature":…, "top_p":…, "top_k":…,
               "listen_prob_scale":…, "force_listen_count":…,
               "max_new_speak_tokens_per_chunk":…, "tts_temperature":…}}}
  ```
  → server replies `session.created {session_id, mode, metrics}`. In full-duplex the server does an
  index-0 prefill (system prompt / ref audio) on init.
- **`input.append`** (full-duplex; client→server, ≈ once/second):
  ```json
  {"type":"input.append","input":{
     "audio":"<b64 float32 PCM 16k mono ~1s>",
     "video_frames":["<b64 jpeg>"],   // only the FIRST frame is used per append
     "force_listen": false }}          // optional: force this step to LISTEN
  ```
  Must **not** carry `messages` (that's turn_based only → `mode_mismatch` fail-fast).
- **Server → client events**:
  - `response.output.delta` + `kind`:
    - `"text"` → `{text}` (accumulate for UI + drive emotion gestures)
    - `"audio"` → `{audio:"<b64 float32 PCM>"}` (async, as TTS is produced → play)
    - `"listen"` → model chose to listen this step (robot stays quiet)
  - `response.done` → `{text, reason:"turn_end", audio:null|…}` (turn boundary)
  - `session.closed` → `{reason}` (reconnect)

## 3. Architecture (what changes, what stays)

```
 mic ─► AudioEngine ─┬─► energy VAD ──► user_speaking (DOA trigger + listen mood + barge-in)
                     └─► 1 s float32 16k chunk ─┐
 camera ─► VideoGrabber ─► latest JPEG b64 ─────┤
                                                ▼
                                        OmniClient  (asyncio thread, WebSocket /backend)
                                          send: input.append(audio[, frame])
                                          recv: text→emit; audio→bus.tts_audio; listen/done→state
                                                ▼
 speaker ◄── AudioEngine.playback ◄── bus.tts_audio (float32 16k, barge-in-cuttable)
                                                │
                    MotionController (100 Hz, single set_target owner)
                    ◄── DOA look-at · state moods · emotion gestures (from omni text)
                        all clamped by motion/safety.py
```

**Reused unchanged**: `bus.py` (state machine, queues, events, barge-in), `motion/*`
(controller, safety, doa, emotions), `audio/aec.py`, `audio/vad.py` (now only DOA/barge-in gate),
`audio/io.py` playback path, `sim/fake_mini.py`, `web.py` + `static/` UI, `text_utils.py`.

**New**:
- `omni/protocol.py` — pure codec: base64⇄float32 PCM, `build_session_init`, `build_input_append`,
  `parse_event`. No I/O → fully unit-testable.
- `omni/client.py` — `OmniClient`: runs asyncio (websockets) in its own thread; sender pulls 1 s
  chunks + latest frame; receiver dispatches events onto the `Bus`; init/reconnect w/ backoff;
  resamples omni output SR→16 k onto `bus.tts_audio`.
- `omni/video.py` — grab `mini.media.get_frame()` → downscale (long edge ≤ `OMNI_VIDEO_MAX_EDGE`)
  → JPEG (`OMNI_VIDEO_JPEG_QUALITY`) → base64, throttled to `OMNI_VIDEO_FPS`. (Salvaged from
  `vision/gating.py`'s encode helpers; gating removed since video is continuous.)
- `omni/fake_server.py` — tiny local asyncio omni WS server for `--stub`/tests: exercises the real
  client path offline (emits `session.created`, alternating `listen`/`text`+`audio`+`done`).

**Rewritten**:
- `pipeline.py` → thin **orchestrator**: wires AudioEngine + VideoGrabber + OmniClient +
  MotionController; maps omni events + local VAD → `ConvState` (LISTENING/SPEAKING/IDLE); barge-in;
  greeting; graceful shutdown → safe rest.
- `audio/io.py` → capture assembles 1 s 16 k mono chunks and hands them to a callback (+ keeps the
  VAD start/stop → `user_speaking`); playback loop kept.
- `config.py` → drop ASR/LLM/TTS/vision-model knobs; add `OMNI_*` (see §4).
- `main.py` → CLI/`LiveChatApp` unchanged in shape; `--stub` starts the fake omni server.

**Removed**: `asr/`, `llm/`, `tts/`, `vision/` packages; their deps in `pyproject.toml`.

## 4. Config (env-driven, `.env`)

```
OMNI_WS_URL=wss://10.0.16.187:8006        # /backend auto-appended
OMNI_MODE=full_duplex                       # full_duplex | turn_based
OMNI_USE_TTS=1
OMNI_TLS_INSECURE=1                         # self-signed cert
OMNI_SYSTEM_PROMPT="你是 Reachy Mini…（中英双语，简洁口语化）"
OMNI_VOICE_REF=                             # optional path to ref .wav for voice-clone
OMNI_OUT_SR=24000                           # omni TTS output rate (assumption; 16000 fallback)
OMNI_CHUNK_MS=1000                          # 1 s per input.append (model's TDM period)
OMNI_SEND_VIDEO=1
OMNI_VIDEO_FPS=1
OMNI_VIDEO_MAX_EDGE=448
OMNI_VIDEO_JPEG_QUALITY=70
# sampling passthrough → session.init.config
OMNI_TEMPERATURE / OMNI_TOP_P / OMNI_TOP_K / OMNI_LISTEN_PROB_SCALE /
OMNI_FORCE_LISTEN_COUNT / OMNI_MAX_SPEAK_TOKENS_PER_CHUNK / OMNI_TTS_TEMPERATURE
# kept: VAD (DOA/barge-in only), ENABLE_AEC, motion (ENABLE_MOTION/DOA, CONTROL_HZ), WEB_*
```

## 5. Real-time, barge-in & safety notes

- **Latency**: granularity is the 1 s TDM chunk (model design); first audio ≈ chunk boundary +
  LAN RTT + prefill + generate. Audio deltas play incrementally. `OMNI_CHUNK_MS` tunable.
- **Echo**: we keep streaming mic while the robot speaks (so the model can barge-in), so we must
  feed **echo-suppressed** mic (`audio/aec.py` capture processing) or the model hears itself; this
  is the main real-hardware tuning knob. Local energy-VAD barge-in also flushes `bus.tts_audio`.
- **Output SR assumption**: MiniCPM-o token2wav ≈ 24 kHz; the server callback drops the rate field,
  so `OMNI_OUT_SR` is configurable and flip-to-16000 is one env change if pitch sounds off.
- **Safety**: unchanged clamps on every `set_target`; emotion moves are pre-authored in-range and
  still clamped; interrupt/shutdown returns to a safe rest pose. Physical safety is not weakened.

## 6. Tests & tooling

- New pure-logic tests: `omni/protocol.py` (b64⇄float32 round-trip, init/append builders,
  event parsing incl. all `kind`s + `done`/`closed`), `omni/video.py` encode.
- Keep motion/safety/doa/bus tests. Remove ASR/LLM/TTS tests.
- `--sim --stub` smoke test drives the **real** OmniClient against `omni/fake_server.py` end-to-end
  (mic→chunk→WS→audio back→speaker→motion) with no GPU server and no hardware.
- `scripts/setup_cm4.sh` (+ README): lightweight venv (numpy, scipy, websockets, pillow, opencv or
  SDK frames, python-dotenv) — no mlx/funasr/kokoro. Update README + `.env.example`.

## 7. Milestones

1. `omni/protocol.py` + tests.  2. `omni/client.py` (asyncio thread, init/reconnect, event bridge).
3. `omni/video.py` grabber.  4. Rewrite `audio/io.py` capture chunking; keep playback.
5. Rewrite `pipeline.py` orchestrator + state mapping + barge-in.  6. `config.py`/`main.py`/`--stub`
   fake server.  7. Delete `asr`/`llm`/`tts`/`vision`; update `pyproject`, `.env.example`, README,
   `scripts/`.  8. Tests green + `ruff` clean + `--sim --stub` smoke run.
```
