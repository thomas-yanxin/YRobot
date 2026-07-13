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
- **Local on Mac** — Silero VAD · FunASR *SenseVoice* · **MiniCPM-V-4.6 VLM via llama.cpp** · Kokoro
  TTS (`mlx-audio`). The VLM (SigLIP2-400M + Qwen3.5-0.8B) handles **both text and vision locally**.
- **Expressive & context-aware motion** — nods on agreement, tilts on questions, plays emotions
  from `pollen-robotics/reachy-mini-emotions-library`, all **clamped to the safe joint range**.
- **DOA** — the head orients toward the speaker's direction.
- **Efficient vision** — a camera frame is attached only on a visual question or a real scene change;
  it is downscaled, JPEG-compressed, and only **one keyframe** is kept in context. Since the VLM is
  local, this saves compute/latency (the image encoder is the costly path), not cloud tokens.
- **Runs without a robot** — `--sim` uses a mock backend so you can develop the whole pipeline.

## Install (Apple Silicon, Python 3.12)

```bash
./scripts/setup_mac.sh            # 3.12 venv + Python deps + llama.cpp + MiniCPM-V-4.6 weights
# or manually:
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[mac,dev]"
brew install llama.cpp            # serves the MiniCPM-V-4.6 VLM (weights auto-download on first run)
cp .env.example .env              # then edit model paths if needed
```

## Run

```bash
# 1) hardware-free development (mock robot, stub engines):
reachy-mini-live-chat --sim --stub

# 2) real pipeline, mock robot. First start the MiniCPM-V-4.6 server (first run
#    auto-downloads the model + vision projector, ~2-3 GB, from Hugging Face):
llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --reasoning off --host 0.0.0.0 --port 8080 -c 4096 &
# --reasoning off is important: MiniCPM-V-4.6 is a thinking model; leaving it on adds a
# long reasoning trace before the answer and blows the latency budget. Vision works either way.
# If Hugging Face is slow/blocked, prefix with a mirror:
#   HF_ENDPOINT=https://hf-mirror.com llama-server -hf openbmb/MiniCPM-V-4.6-gguf:Q4_K_M --reasoning off --port 8080 -c 4096 &
reachy-mini-live-chat --sim
# Note: the app's first non-sim/stub run also downloads the SenseVoice ASR model
# (~900 MB) once — let it finish; it's cached afterwards.

# 3) on the robot: start the daemon, then launch the app
reachy-mini-daemon                       # Lite (USB)
reachy-mini-live-chat                     # connects to the daemon, opens web UI on :8042
```

Open the web UI at <http://localhost:8042> for the live transcript, camera view, and settings.

## Configuration (`.env`)

| Var | Meaning | Default |
|-----|---------|---------|
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | MiniCPM-V-4.6 VLM served by llama.cpp (OpenAI-compatible) | `http://localhost:8080/v1` · `openbmb/MiniCPM-V-4.6-gguf` |
| `VISION_BASE_URL` / `VISION_MODEL` / `VISION_API_KEY` | vision turns (defaults to the same local VLM; can point to a cloud VLM) | same local server |
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
