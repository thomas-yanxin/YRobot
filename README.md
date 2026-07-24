---
title: YRobot
emoji: 🤖
colorFrom: indigo
colorTo: pink
sdk: static
pinned: false
short_description: Full-duplex MiniCPM-o 4.5 conversation for Reachy Mini Wireless
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# YRobot

Full-duplex, omni-modal conversation for **Reachy Mini Wireless**, powered by the
**MiniCPM-o 4.5 realtime API** ([docs](https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/)).

The robot listens, watches, speaks and moves at the same time: you can talk over it and it
shuts up within one audio tick, it turns to face whoever is speaking, and it breathes,
glances and dances its antennas while it talks. The whole application is ~900 lines across
six single-purpose modules — it is meant to be *read*, not just run.

```
you ─── voice ─► XVF3800 AEC mic ─► 20 ms VAD/control ─► bounded audio queue ─┐
      camera ─► independent latest-only JPEG worker ──────────────────────────┤
                                                                              ▼
                                wss://…/v1/realtime?mode={audio|video}
                                                                              │
◄── lifelike motion ◄─ 50 Hz choreographer ◄─ DoA compass                    │
◄── speech          ◄─ epoch-tagged speaker ◄─ 24 kHz audio deltas ◄─────────┘
```

## Why it feels responsive

Everything below is encoded in the source with the reasoning attached; this is the map.

| Problem | Mechanism | Where |
|---|---|---|
| Reply latency | Mic/VAD never waits for camera encoding or WebSocket sends. Audio enters a bounded realtime queue; video is latest-only. Adaptive 0.25–0.8 s playback preroll absorbs server jitter | `main.py`, `audio.py` |
| Barge-in | Turn-aware and client-owned: a qualified candidate immediately ducks playback, sends its current audio with `force_listen`, and suppresses every old output branch. `response_id` is deliberately not used as a turn boundary. The final user tail is flushed after 180 ms of silence; only a `listen` acknowledged after the last force plus 450 ms of user quiet admits the new answer | `turn.py`, `audio.py`, `main.py` |
| False triggers from its own echo/motors | The AEC residual tracks the far-end envelope within a syllable, so barge-in is two-staged: an `EchoGuard` predicts the residual from the exact audio we played and gates candidates; a candidate provisionally forces listening while *ducking* playback (un-played tail retained), then `DuckVerifier` re-checks with the speaker silent — sustained voice commits the flush, silence resumes the held tail and the guard learns the leakage. Motor noise is separately killed by an asymmetric RMS noise floor plus a 60 ms confirmation streak on top of WebRTC VAD | `audio.py`, `turn.py` |
| Wooden motion | One 50 Hz thread owns the pose. Breathing and posture cross-fade; gaze and idle saccades both use velocity-limited second-order trajectories, so a new random glance cannot step the head in one tick | `motion.py` |
| Deaf DoA | Samples are gated by post-echo confirmed user voice, transformed with the daemon's physical head pose, confidence-weighted with the XVF speech flag, circularly averaged, and dead-banded | `motion.py`, `main.py` |
| Context rot | Vision costs ~64 kv tokens/frame against an ~8 k budget: frames go up at 1 fps in conversation, 0.2 fps idle, never while only the robot is speaking; sessions rotate at the first quiet moment past the time/kv budget | `main.py` |

## Protocol in one paragraph

Connect to `wss://HOST/v1/realtime?mode=audio` for voice-only or `mode=video` for
camera input, wait for `session.queue_done`, send
`session.init` (the system prompt must start with the trained line `You are a helpful
assistant.` — a free-form persona drifts the model out of its duplex distribution), wait
~14 s for `session.created`, then stream `input.append` units: base64 float32 16 kHz mono
audio, optional `force_listen`, and—only in video mode—base64 JPEG `video_frames`. The server streams
`response.output.delta` events with `kind ∈ {listen, text, audio}` (audio is 24 kHz
float32); **only `listen` is an utterance boundary** — text and audio are independent
streams. See `realtime.py`.

## Run

On the robot (Python 3.12 venv on the CM4), or any machine that can reach the daemon:

```bash
pip install -e .
cp .env.example .env   # point YROBOT_REALTIME_URL at your gateway
yrobot
```

It also registers as a Reachy Mini app (`reachy_mini_apps` entry point `yrobot`), so the
dashboard can start and stop it.

Development without hardware:

```bash
pip install -e ".[dev]"
pytest && ruff check .
```

## Layout

```
yrobot/config.py     env → one frozen Settings dataclass; URL normalization
yrobot/realtime.py   gateway protocol client + <think>-leak filter
yrobot/turn.py       barge-in state machine (pure logic, fully unit-tested)
yrobot/audio.py      mic framing, VAD stack, 24→16 k resampler, epoch speaker
yrobot/motion.py     DoA sound compass + 50 Hz choreographer
yrobot/main.py       wiring, session rotation, ReachyMiniApp + CLI
```

Every module docstring states the non-obvious constraint it encodes (gateway behaviour
verified live, SDK threading rules, XVF3800 quirks). If you change a number, read the
docstring above it first.

## License

Apache-2.0
