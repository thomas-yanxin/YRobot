---
title: Reachy Mini Realtime Dialogue
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
tags:
  - reachy_mini_python_app
---

# Reachy Mini — Realtime Full-Duplex Audio-Visual Dialogue 🤖🎙️👁️

A **full-duplex, bilingual (中文 / English)** voice + video conversational app for
[Reachy Mini](https://github.com/pollen-robotics/reachy_mini). The brain is a **remote
end-to-end omni model** — **MiniCPM-o 4.5** served by
[`llama.cpp-omni`](https://github.com/tc-mb/llama.cpp-omni)'s `llama-omni-server` — reached
over a **full-duplex WebSocket**. The robot side (Wireless **CM4**) just streams 1-second mic
chunks + a camera frame up, plays the returned speech, turns its head toward whoever is speaking
via **DOA**, and reacts with expressive, **safety-clamped** motion.

Because ASR + reasoning + TTS all happen on the omni server, the on-robot workload is tiny — no
local ML models. Design & rationale: see [`plan.md`](plan.md).

## Highlights

- **End-to-end omni, full duplex** — the model itself decides *listen* vs *speak* at ~1 Hz;
  audio plays incrementally as it streams back. Talk over the robot and it can barge-in.
- **DOA** — while you speak, the head (and body, past ±45°) turns toward your direction, using the
  robot's mic-array `get_DoA()`, EMA-smoothed and clamped.
- **Expressive, safe motion** — conversation-state moods (idle / listening / speaking) plus emotion
  gestures inferred from the model's transcript (nod on "好的/yes", tilt on questions, …). **Every**
  command is clamped to the documented joint range in `motion/safety.py`.
- **Continuous vision** — one downscaled JPEG frame is attached to each audio chunk (~1 fps), so the
  model always has current visual context.

## Protocol (verified against the source)

The client speaks llama.cpp-omni's `/backend` WebSocket protocol
(`tools/server/ws_handler.cpp` + `protocol.cpp`):

- `session.init` `{payload:{mode:"full_duplex", use_tts, system_prompt, voice, config}}` → `session.created`.
- Full-duplex `input.append` carries base64 **float32-LE PCM, mono, 16 kHz** (~1 s) as `audio`, plus an
  optional `video_frames:[<b64 jpeg>]` (only the first frame is used). It must **not** carry `messages`.
- Server events: `response.output.delta` with `kind` ∈ `text` / `audio` (base64 float32 PCM) / `listen`;
  `response.done` (turn boundary); `session.closed`.

**Two ways to reach it** (set `OMNI_ENDPOINT`):

- `gateway` (default) — the MiniCPM-o-Demo `gateway.py` public entry (e.g. `:8006`). The client connects
  to `/v1/realtime?mode=<video|audio|chat>`; the gateway queues the session (emits `session.queued`, model
  cold-start 10–60 s) then passes the protocol above through to the backend unchanged. `video` = audiovisual.
- `backend` — a raw `llama-omni-server` (e.g. `:28099`), connecting directly to `/backend`.

Either way the on-the-wire `session.init`/`input.append` protocol is identical; only the URL path differs.

## Install

Primary target — **on the robot's CM4** (Wireless, ARM64 Linux):

```bash
./scripts/setup_cm4.sh            # 3.12 venv, thin client — no local ML models
# then edit .env → set OMNI_WS_URL=wss://<your-omni-server>:8006
```

The robot is only a client, so the hard dependencies are just **`numpy` + `websockets`**
(`reachy-mini` is already in the CM4 image). There is **no torch / CUDA / scipy** — the omni
server does all inference. Optional extras each improve one thing and each has a
pure-numpy/stdlib fallback, so you add only what you want:

| Extra | Adds | Fallback if absent |
|-------|------|--------------------|
| `vision` | `pillow` (downscale frames) | the SDK's full-res `get_frame_jpeg()` |
| `hifi` | `scipy` polyphase resampling | numpy linear-interp resampling |
| `web` | `fastapi`/`uvicorn` control UI | headless (state via logs) |

Dev on a laptop (bridged to the robot):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[laptop,dev]"
cp .env.example .env
```

## Run

```bash
# on the robot (daemon running on the CM4):
reachy-mini-live-chat                # connects to the daemon, opens web UI on :8042
#    or let the Reachy Mini app launcher discover it (entry point: reachy_mini_apps).
```

Open the web UI at <http://localhost:8042> for the live transcript, camera view, and state.

## Configuration (`.env`)

| Var | Meaning | Default |
|-----|---------|---------|
| `OMNI_WS_URL` | base WS url (scheme://host:port); path derived from `OMNI_ENDPOINT` | `wss://10.0.16.187:8006` |
| `OMNI_ENDPOINT` | `gateway` (MiniCPM-o-Demo `:8006`) \| `backend` (raw `llama-omni-server`) | `gateway` |
| `OMNI_GATEWAY_MODE` | gateway only: `video` (audiovisual) \| `audio` \| `chat` | `video` |
| `OMNI_MODE` | `session.init` protocol mode: `full_duplex` \| `turn_based` | `full_duplex` |
| `OMNI_USE_TTS` / `OMNI_TLS_INSECURE` | model speaks / skip TLS verify (self-signed) | `1` / `1` |
| `OMNI_SESSION_READY_S` | wait for `session.created` (gateway queue + cold start) | `60` |
| `OMNI_SYSTEM_PROMPT` / `OMNI_VOICE_REF` | persona / voice-clone ref .wav | built-in / — |
| `OMNI_OUT_SR` | server TTS output rate (set `16000` if voice sounds sped up) | `24000` |
| `OMNI_CHUNK_MS` | mic audio per `input.append` | `1000` |
| `OMNI_SEND_VIDEO` / `OMNI_VIDEO_*` | attach a frame per chunk; size/quality | `1` / 448px q70 |
| `VAD_SILENCE_MS` | silence marking end of human speech (DOA/barge-in) | `320` |
| `ENABLE_MOTION` / `ENABLE_DOA` / `ENABLE_AEC` | feature toggles | `1` / `1` / `0` |
| `LANG` | `auto` \| `zh` \| `en` | `auto` |

## Architecture

Voice activity comes from the XVF3800 firmware's post-AEC voice flag (polled over USB at ~20 Hz;
it drives `user_speaking` → DOA + listen mood + barge-in); the mic feeds a continuous chunker. `OmniClient` runs the full-duplex WebSocket on its own asyncio
thread: a sender streams `{audio, frame}` up; a receiver dispatches `text`/`audio`/`listen`/`done`
events. A single 100 Hz control-loop thread owns `set_target` (per the SDK guidance); every other
thread only enqueues motion *intents*. See [`plan.md`](plan.md) for the diagram and safety table.

## Safety

Every motion command is clamped to Reachy Mini's documented limits (head pitch/roll ±40°, head yaw
±180°, body yaw ±160°, head−body yaw delta ≤65°) in `motion/safety.py` — defense-in-depth on top of
the SDK's own clamping. On barge-in/shutdown the robot returns to a safe rest pose. The app never
commands beyond that range.

## Tests

```bash
pytest            # omni protocol codec, video encode, safety clamp, emotion mapping, DOA math, wiring smoke
```

## License

Apache-2.0. Uses the Reachy Mini SDK and community model libraries under their respective licenses.
