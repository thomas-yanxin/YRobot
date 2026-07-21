---
title: YRobot
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
short_description: Full-duplex MiniCPM-o conversation for Reachy Mini Wireless
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# YRobot

YRobot gives Reachy Mini Wireless a real-time audiovisual conversation loop powered by
MiniCPM-o 4.5. The app runs as a thin client on the CM4 and connects directly to a remote
`llama-omni-server`.

## What it does

- Streams one-second 16 kHz microphone slices and current camera JPEGs to `/backend`.
- Plays streamed 24 kHz speech through a stateful 24→16 kHz resampler, preserving
  filter/phase continuity across TTS deltas instead of creating audible block edges.
- Uses a 320 ms first-audio prebuffer to stay beyond the player's 200 ms clock-reset window
  and absorb Wi-Fi and Token2Wav delivery jitter.
- Keeps the full-duplex microphone cadence while Reachy speaks, but uploads silence for
  far-end-only audio; locally confirmed near-end speech is passed through for voice barge-in.
- Supports voice barge-in: debounced DoA speech plus post-AEC microphone energy clears
  local/GStreamer playback and holds `force_listen` until the server acknowledges listening.
- Applies the official conversation app's XVF3800 echo/noise/gain tuning at startup.
- Turns toward a detected speaker with Reachy's DoA API.
- Keeps a slightly raised natural gaze; DoA changes yaw without accumulating downward pitch.
- Keeps the last speaker as an attention anchor instead of replacing it with permanent random poses.
- Adds small mixed-frequency breathing, occasional minimum-jerk glances, listening nods, and
  asymmetric antenna motion; antennas hold still while the user is speaking.
- Uses the SDK's playback-synchronised audio-reactive wobble for speech, layered over a smaller
  base motion so the voice remains expressive without looking mechanically repetitive.
- Uses the CM4-local media backend, avoiding a WebRTC encode/decode loop on the robot.
- Encodes camera JPEGs on a dedicated latest-frame worker so video cannot stall audio upload.
- Disables WebSocket compression for already-dense PCM/JPEG payloads and serializes uploads
  away from the receive loop, keeping streamed TTS packets responsive on the CM4.
- Sends all Phase-A motion through one phase-aligned 50 Hz `set_target` control loop with
  time-based speed limits, so loop jitter cannot change movement speed.
- Settles a late final TTS delta back to listening when real playback drains, avoiding a stuck
  speaking pose.
- Reconnects after network failures and returns to a neutral pose on shutdown.

The first motion phase is deliberately bounded. MiniCPM-o's raw full-duplex protocol emits
`listen`, `text`, and `audio`, but no tool calls. Named dances and emotions will be added later
through a small action interface instead of coupling them to the audio transport.

## Install on the CM4

```bash
git clone <your-yrobot-repository>
cd YRobot
./scripts/setup_cm4.sh
```

The setup script creates `.venv`, installs the app, and creates `.env` from `.env.example`.
The default endpoint is already:

```text
wss://10.0.16.187:28099/backend
```

The certificate is currently treated as self-signed (`OMNI_TLS_VERIFY=0`). Enable verification
after the server uses a certificate trusted by the CM4.

## Run

With the Reachy daemon already running, execute this on the Wireless CM4:

```bash
source .venv/bin/activate
yrobot
```

The Reachy dashboard can also discover the `yrobot` Python-app entry point.

Useful overrides:

```bash
yrobot --no-video
yrobot --url wss://another-server:28099/backend
yrobot --tls-verify
```

## Verify

```bash
reachy-mini-app-assistant check .
python scripts/probe_omni.py
```

The probe sends one second of silence with the protocol's `force_listen` hint. It checks TLS,
session initialization, audio prefill, and the response channel without making the model speak.
For local development checks, install `.[dev]` and run `python -m pytest` plus `ruff check .`.

The motion worker never runs on the microphone, WebSocket, camera, or playback workers. DoA is
sampled at 20 Hz inside the motion worker while pose interpolation runs at 50 Hz, so the added
animation does not delay the full-duplex conversation path.

At runtime, warnings named `TTS supply gap`, `Slow playback stage`,
`Slow Omni input cadence`, and `Omni switched to listen` identify whether a remaining pause
comes from the remote TTS, the CM4 audio path, upload backpressure, or a server-side turn
transition while local speech is still buffered. If speech is still interrupted, first run
`yrobot --no-video` as an A/B check and inspect whether `User barge-in` appears without an
actual interruption from the user.

Simulation can exercise lifecycle and motion code, but physical audio, camera, DoA, and speaker
behavior must be verified on the Wireless robot.

## Design

The runtime is intentionally four modules:

- `config.py`: validated environment configuration.
- `omni.py`: protocol codec and one reconnecting WebSocket session.
- `robot.py`: Reachy media and serialized phase-A motion.
- `main.py`: app and CLI lifecycle.

See [plan.md](plan.md) for protocol findings and acceptance criteria.

## Sources

- [MiniCPM-o 4.5 llama.cpp-omni deployment](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/minicpmo_4_5_llamacpp_omni_zh.md)
- [MiniCPM-o Realtime examples](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/main/examples/realtime)
- [Reachy Mini Python SDK](https://huggingface.co/docs/reachy_mini/SDK/python-sdk)
- [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app)
- [Conversation app single-owner movement loop](https://github.com/pollen-robotics/reachy_mini_conversation_app/blob/main/src/reachy_mini_conversation_app/moves.py)
- [Reachy Mini audio-reactive head wobbler](https://github.com/pollen-robotics/reachy_mini/blob/main/src/reachy_mini/motion/head_wobbler.py)
