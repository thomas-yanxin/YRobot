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
you ──── voice / camera ──────────► XVF3800 AEC mic ─► VAD ─► 500 ms chunks ─┐
                                       camera ──────► 1 fps JPEG (adaptive) ─┤
                                                                             ▼
                                              wss://…/v1/realtime?mode=audio (MiniCPM-o 4.5)
                                                                             │
◄── lifelike motion ◄─ 50 Hz choreographer ◄─ DoA compass                    │
◄── speech          ◄─ epoch-tagged speaker ◄─ 24 kHz audio deltas ◄─────────┘
```

## Why it feels responsive

Everything below is encoded in the source with the reasoning attached; this is the map.

| Problem | Mechanism | Where |
|---|---|---|
| Reply latency | 500 ms uplink chunks (halves perceived latency vs the 1 s browser cadence; 250 ms breaks the model's turn-taking), `mode=audio` (600 s sessions, frames still accepted), adaptive 0.25–0.8 s playback preroll with a hard 1.2 s device backlog cap | `config.py`, `audio.py` |
| Barge-in | Client-owned: the gateway's `force_listen` neither stops generation nor prevents the model *resuming* an interrupted monologue. On voice onset we flush the player and discard the whole stale turn; only two consecutive clean `listen` boundaries release the discard latch | `turn.py` |
| False triggers from its own motors | Hardware AEC removes the robot's voice; motor noise is killed by an asymmetric RMS noise floor (falls fast in quiet, creeps up under sustained sound) plus a 60 ms confirmation streak on top of WebRTC VAD | `audio.py` |
| Wooden motion | One 50 Hz thread owns the pose: phase-shifted breathing oscillators, held idle glances, cross-faded listening/speaking postures, a critically damped gaze spring. Speech articulation is the SDK's daemon-side `enable_wobbling()` (PTS-synced to the speaker), body rotation follows via `set_automatic_body_yaw(True)` | `motion.py` |
| Deaf DoA | The XVF3800 firmware speech flag fires **pre-AEC** — it hears the robot itself, which is why naive DoA feels broken. We sample `DOA_VALUE_RADIANS` only while *our* VAD hears the user, average on the circle over 1 s, and hand the choreographer a dead-banded gaze target | `motion.py` |
| Context rot | Vision costs ~64 kv tokens/frame against an ~8 k budget: frames go up at 1 fps in conversation, 0.2 fps idle, never while only the robot is speaking; sessions rotate at the first quiet moment past the time/kv budget | `main.py` |

## Protocol in one paragraph

Connect to `wss://HOST/v1/realtime?mode=audio`, wait for `session.queue_done`, send
`session.init` (the system prompt must start with the trained line `You are a helpful
assistant.` — a free-form persona drifts the model out of its duplex distribution), wait
~14 s for `session.created`, then stream `input.append` units: base64 float32 16 kHz mono
audio, optional base64 JPEG `video_frames`, optional `force_listen`. The server streams
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
