# YRobot

Full-duplex bilingual (中/英) voice + vision live chat for **Reachy Mini
(wireless)**, driven by a remote **MiniCPM-o 4.5** realtime gateway
([OpenBMB/MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo)).
The robot hears, sees, talks and moves at the same time; you can interrupt it
mid-sentence like a person.

```
mic (16 kHz AEC'd) ──500 ms chunks──►┐
camera ──JPEG, adaptive cadence─────►├─ wss /v1/realtime ──► MiniCPM-o 4.5
                                     │
speaker (flushable) ◄─24 kHz audio──┤◄─ listen / text / audio deltas
daemon head-wobble ◄─(same audio)   │
face-tracking + breath + DOA ◄──────┘  conversation state
```

## Design notes (measured against the live gateway)

- **Latency.** 500 ms uplink chunks give ≈1.1–1.3 s from end-of-speech to the
  first reply audio (vs ≈1.7 s at 1000 ms). Smaller chunks break the server's
  turn-taking — it answers before you finish. Downlink audio is pushed to the
  speaker the moment it arrives; no client-side buffering.
- **Barge-in is client-owned.** The server acks a `force_listen` within
  ~0.2 s but keeps streaming (and later *resumes*) the interrupted reply.
  So on a voice onset while the robot speaks, YRobot instantly flushes the
  GStreamer player, discards the whole stale turn, and keeps re-forcing
  listen until the model actually yields (`yrobot/turn.py`).
- **No self-interruption.** Voice detection is an adaptive energy gate on the
  XVF3800's *AEC'd* mic stream — the only signal free of the robot's own
  voice — with a stricter threshold while the robot speaks.
- **Session budget.** The backend's kv cache degrades past 8192 tokens and a
  new session costs ~14 s of server-side model reset. Vision is the main
  burner (~64 tokens/frame), so frame cadence adapts (1 fps active, 1/5 s
  idle) and sessions rotate kv/age-aware at quiet moments. `mode=audio`
  accepts video frames too and doubles the session cap to 600 s — it is the
  default.
- **Motion is layered, not scripted.** Daemon-side audio-reactive head
  wobble (synced to the actual speaker output) + daemon face tracking with a
  state-dependent blend weight + a 20 Hz puppeteer adding breath, antenna
  moods, DOA body turns and a visible "dozing" pose while reconnecting.

## Install (on the robot)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .
```

## Run

```bash
cp .env.example .env   # optional; defaults match the LAN gateway
yrobot                 # or: python -m yrobot.main
```

It also registers as a Reachy Mini app (`reachy_mini_apps` → `yrobot`) for
the dashboard/app manager.

## Test

```bash
pip install -e .[dev]
pytest
```

Unit tests cover the turn gate, voice gate, AGC, resampler and protocol —
no hardware needed. `scripts/probe_realtime.py` exercises the live gateway
(latency, chunk sizing, barge-in behaviour) from any machine.

## Configuration

Everything is a `YROBOT_*` environment variable (or `.env`); see
[.env.example](.env.example) for the full annotated list.
