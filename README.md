---
title: Reachy Mini Live Chat
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
tags:
  - reachy_mini_python_app
---

# Reachy Mini — Live Video Chat 🤖🎙️

A **full-duplex, low-latency, bilingual (中文 / English)** voice + video conversational app for
[Reachy Mini](https://github.com/pollen-robotics/reachy_mini). It runs the classic
`VAD → ASR → LLM → TTS` pipeline **locally on Apple Silicon** (MLX), reacts with Reachy's
expressive motions, turns its head toward whoever is speaking via **DOA**, and keeps vision-token
usage tiny by only sending a camera frame to a vision model when it actually matters.

> Design & rationale: see [`plan.md`](plan.md).

## Highlights

- **Full duplex** — streaming input, automatic endpointing, real-time reply, and **barge-in**
  (talk over the robot and it stops and listens).
- **≤ 1.5 s** end-to-end latency (last word → first audio). Typical critical path ≈ 0.5–1.1 s.
- **Local on Mac** — Silero VAD · FunASR *SenseVoice* · `mlx-lm` Qwen3 · Kokoro TTS (`mlx-audio`).
- **Expressive & context-aware motion** — nods on agreement, tilts on questions, plays emotions
  from `pollen-robotics/reachy-mini-emotions-library`, all **clamped to the safe joint range**.
- **DOA** — the head orients toward the speaker's direction.
- **Token-efficient vision** — a frame is sent to the vision LLM only on a visual question or a real
  scene change; it is downscaled, JPEG-compressed, and only **one keyframe** is kept in context.
- **Runs without a robot** — `--sim` uses a mock backend so you can develop the whole pipeline.

## Install (Apple Silicon, Python 3.12)

```bash
./scripts/setup_mac.sh            # creates a 3.12 uv venv and installs everything
# or manually:
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[mac,dev]"
cp .env.example .env              # then edit tokens / model ids
```

## Run

```bash
# 1) hardware-free development (mock robot, stub engines):
reachy-mini-live-chat --sim --stub

# 2) real pipeline, mock robot (needs models downloaded):
reachy-mini-live-chat --sim

# 3) on the robot: start the daemon, then launch the app
reachy-mini-daemon                       # Lite (USB)
reachy-mini-live-chat                     # connects to the daemon, opens web UI on :8042
```

Open the web UI at <http://localhost:8042> for the live transcript, camera view, and settings.

## Configuration (`.env`)

| Var | Meaning | Default |
|-----|---------|---------|
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | local text LLM (mlx-lm server, OpenAI-compatible) | `http://localhost:8080/v1` · `Qwen3-4B` |
| `VISION_BASE_URL` / `VISION_MODEL` / `VISION_API_KEY` | cloud/local vision LLM | ModelScope `Qwen-Ambassador/Qwen3.7-Plus` |
| `ASR_MODEL` | FunASR model id | `iic/SenseVoiceSmall` |
| `TTS_VOICE_ZH` / `TTS_VOICE_EN` | Kokoro voices | `zf_xiaobei` · `af_heart` |
| `VAD_SILENCE_MS` | silence to end a turn | `320` |
| `LANG` | `auto` \| `zh` \| `en` | `auto` |
| `ENABLE_VISION` / `ENABLE_MOTION` / `ENABLE_AEC` | feature toggles | `1` / `1` / `0` |

## Architecture

`VAD → ASR → LLM → TTS` stages run as concurrent threads coordinated by a conversation state machine
(`IDLE → LISTENING → THINKING → SPEAKING`). A single 100 Hz control-loop thread owns `set_target`
(per the SDK control-loop guidance); every other thread only enqueues motion *intents*. See
[`plan.md`](plan.md) for the diagram, latency budget, and safety table.

## Safety

Every motion command is clamped to Reachy Mini's documented limits (head pitch/roll ±40°, head yaw
±180°, body yaw ±160°, head−body yaw delta ≤65°) in `motion/safety.py` — defense-in-depth on top of
the SDK's own clamping. The app never commands beyond that range.

## Tests

```bash
pytest            # pure-logic tests: safety clamp, emotion mapping, DOA math, vision gating, clause split
```

## License

Apache-2.0. Uses the Reachy Mini SDK and community model libraries under their respective licenses.
