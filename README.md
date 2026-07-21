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
- Starts recording and playback first, then applies the official conversation app's XVF3800
  echo/noise/gain tuning with the far-end reference path active.
- Continuously uploads the real post-XVF microphone signal while Reachy speaks. Audio is never
  replaced with silence, so MiniCPM-o can hear new user speech and make its native `listen/speak`
  decision on every full-duplex time slice.
- Selects XVF channel 0, matching the official conversation app instead of averaging both USB
  channels, and enables XVF's robust double-talk mode (`PP_DTSENSITIVE=10`).
- Uses MiniCPM's natural `listen/speak` decision first. As a reliability fallback, DoA can never
  interrupt alone: post-AEC speech must remain at least 6 dB above the learned far-end residual
  for a sustained 120 ms before playback is cleared and `force_listen` is sent.
- Sends `force_listen` as a silent control slice and temporarily holds the real microphone slice;
  after MiniCPM acknowledges `listen`, the user's interrupting words are forwarded normally instead
  of being consumed by the control decision. Waiting for that acknowledgement is bounded, and a new
  session clears any stale interruption state, so a lost acknowledgement can never mute the robot.
- Discards the whole interrupted turn: the backend streams a turn in bursts that run many seconds
  ahead of playback, so its audio is dropped until the next `listen` boundary (the model actually
  stopped speaking) instead of resuming mid-sentence after a timeout. The GStreamer flush runs on
  the playback worker, the only thread that may cycle the shared record+playback pipeline.
- Treats `response_id`/`response.done` as one-second time-slice bookkeeping, not sentence
  boundaries: one utterance spans several consecutive slices, so slice boundaries never restart
  playback, reset the resampler, or insert preroll waits, and transcript fragments are aggregated
  until the `listen` boundary before logging.
- Adapts the start-of-utterance preroll to observed TTS delivery gaps (a mid-utterance stall
  resumes immediately rather than re-buffering) and lets one slow Wi-Fi send degrade the next
  slice to audio-only, so video never costs speech fluency.
- Applies uplink AGC toward 0.12 rms (gain-up only, frozen while the robot speaks) because the omni
  model treats quiet near-end speech as background and never answers it.
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

XVF3800 is responsible for suppressing far-end echo, and MiniCPM-o receives every post-AEC
microphone slice while it is speaking. For reliable barge-in, the client learns the speaker-only
residual during the first 400 ms of playback and requires sustained double-talk evidence before
forcing `listen`. When MiniCPM confirms `listen`, the application and GStreamer queues are flushed.
For an ordinary quiet end-of-turn, buffered sentence audio is allowed to drain so the answer is
not cut off. Physical testing should still confirm the threshold against the robot's actual room,
speaker volume, and user distance.

YRobot sends `length_penalty=1.1` by default to reduce premature end-of-turn sampling. It can be
tuned with `OMNI_LENGTH_PENALTY` in the supported backend range of 0.1–5.0.

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

At runtime, warnings named `TTS supply gap`, `Slow playback stage`, and
`Slow Omni input cadence` identify whether a remaining pause comes from remote TTS, the CM4 audio
path, or upload backpressure. A real interruption logs `Post-AEC double talk detected` followed by
`MiniCPM listen confirmed user interruption`; DoA activity by itself cannot trigger either action.

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
- [MiniCPM-o native full-duplex protocol](https://github.com/OpenBMB/MiniCPM-o-Demo/blob/main/docs/zh/api/duplex.md)
- [Reachy Mini Python SDK](https://huggingface.co/docs/reachy_mini/SDK/python-sdk)
- [Reachy Mini conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app)
- [XMOS XVF3800 echo/double-talk tuning](https://www.xmos.com/documentation/XM-014888-PC/html/modules/fwk_xvf/doc/user_guide/04_tuning_the_application.html)
- [Conversation app single-owner movement loop](https://github.com/pollen-robotics/reachy_mini_conversation_app/blob/main/src/reachy_mini_conversation_app/moves.py)
- [Reachy Mini audio-reactive head wobbler](https://github.com/pollen-robotics/reachy_mini/blob/main/src/reachy_mini/motion/head_wobbler.py)
