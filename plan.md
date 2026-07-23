# YRobot blank-slate rebuild plan

Status: implemented and locally verified on 2026-07-23. Physical Wireless acceptance
remains intentionally separate because this workstation has no connected Reachy Mini.

## Product target

- Hardware: Reachy Mini Wireless, application running on its CM4.
- Model: MiniCPM-o 4.5 public Realtime API in `mode=video`.
- Experience: continuous audio/video input, concurrent speech output, immediate local
  barge-in, echo-resistant near-end detection, sound-oriented attention, and smooth motion.
- Constraint: do not derive the design from the previous YRobot implementation.

## Protocol contract

- Wait for `session.queue_done`, send one `session.init`, then wait for
  `session.created`.
- Send one real-time-paced `input.append` per second: 16 kHz mono little-endian
  float32 PCM plus at most one 640 px JPEG and `max_slice_nums=1`.
- Consume independent `response.output.delta` branches: `listen`, `text`, and 24 kHz
  mono float32 `audio`. Full-duplex mode does not use per-turn `response.done`.
- On barge-in, flush Reachy's player locally, invalidate all older playback epochs, drop
  further old output, and set `force_listen=true` on input units until `kind=listen`.
- Stop input before `session.close`; treat `session.closed`, WebSocket close, and `error`
  as terminal. Recreate video sessions before the 300 second hard limit.

## Runtime architecture

- Keep Reachy's LOCAL media backend so the XVF3800 hardware AEC remains in the path.
- Split microphone samples into 20 ms frames for local VAD while aggregating exactly one
  second for the model. Never wait for an utterance to end before uploading.
- Give capture, playback, camera, DoA, WebSocket, and motion independent bounded workers.
  A slow camera or USB read must never block the audio hot path.
- Use Reachy's `clear_player()` for physical barge-in; never use the deprecated no-op
  `clear_output_buffer()`.
- Use AEC output plus WebRTC VAD, adaptive energy gating, and recent playback correlation
  to distinguish near-end speech from the robot's own voice.
- Poll hardware DoA independently, circularly smooth it, and accept a bearing when either
  XVF3800 speech detection or the local near-end detector is active. The linear array's
  documented front/back ambiguity remains a hardware limitation.
- Keep one fixed-rate motion owner. It composes DoA attention, low-amplitude idle motion,
  speaking state, antenna motion, and rate/acceleration limiting before the only
  `set_target()` call. Reachy's daemon-side speech wobble remains an additive offset.
- Record monotonic latency and drop/interrupt counters without logging audio, images, or
  secrets.

## Acceptance

- Protocol payloads and lifecycle match the public MiniCPM-o 4.5 documentation.
- Stale audio cannot cross an interruption or session boundary.
- A synthetic echo does not trigger barge-in; independent near-end speech does.
- Uplink units are exactly 16,000 float32 samples and are sent at monotonic one-second
  cadence without burst replay.
- Motion has one writer, fixed cadence, bounded velocity/acceleration, and graceful stop.
- Ruff, compile checks, unit tests, the Reachy app checker, and no-hardware integration
  tests pass locally.
- Physical Wireless acceptance still covers audible stop latency, false interrupts,
  first-audio latency, DoA response, motion feel, and session rollover.

## Questions

None outstanding. The user supplied the target robot and model documentation; the local
configuration supplies the gateway and confirms on-robot execution. All remaining tuning
values are safe, documented environment settings.
