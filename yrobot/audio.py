"""Audio capture, voice detection and interruptible playback.

Reachy Mini Wireless routes the microphone through the XVF3800's hardware
acoustic echo canceller, but the AEC residual of the robot's own speech
still tracks the far-end envelope within a syllable — loud TTS passages
pierce any static VAD threshold (hardware logs 2026-07-22 and 2026-07-24).
Barge-in detection is therefore two-staged:

1. ``EchoGuard`` predicts the expected residual from the exact audio we
   played (we push every frame, so the far end is known) and only lets a
   candidate through when the mic beats prediction + margin;
2. a confirmed candidate only *ducks* playback (``Speaker.hold()`` — the
   device queue is flushed but the un-played tail is retained). With the
   speaker silent the echo path is dead, so the verify stage in
   ``turn.DuckVerifier`` can trust the microphone: sustained voice commits
   the destructive flush, silence resumes the held tail and the user hears
   a short dip instead of losing the turn.

Playback is epoch-tagged: a committed barge-in bumps the epoch, and
everything queued under an older epoch is dropped while the GStreamer
pipeline is flushed via ``clear_player()`` — always from the playback
thread, the only thread allowed to touch the pipeline.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque

import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)

FRAME_MS = 20
FRAME_SAMPLES = 16_000 * FRAME_MS // 1000  # 320
SILENT_DB = -120.0


class Microphone:
    """Yields fixed 20 ms mono float32 frames from the robot microphone."""

    def __init__(self, media) -> None:
        self._media = media
        self._buf = np.empty(0, np.float32)

    def read_frames(self) -> list[np.ndarray]:
        """Drain the device and return all complete 20 ms frames."""
        sample = self._media.get_audio_sample()
        if sample is None or len(sample) == 0:
            time.sleep(0.005)
            return []
        mono = sample[:, 0] if sample.ndim == 2 else sample
        self._buf = np.concatenate([self._buf, mono.astype(np.float32)])
        n = len(self._buf) // FRAME_SAMPLES
        frames = [self._buf[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES] for i in range(n)]
        self._buf = self._buf[n * FRAME_SAMPLES :]
        return frames


class VoiceDetector:
    """Motor-noise-resistant user speech detector over 20 ms frames.

    ``voiced``  — instantaneous, streak-confirmed; drives barge-in and the
                  turn gate (no hangover, so silence is detected promptly).
    ``active``  — voiced within the last 300 ms; gates DoA sampling and the
                  camera frame cadence.
    ``last_db`` — level of the most recent frame, for the echo guard.
    """

    CONFIRM_FRAMES = 3
    HANGOVER_S = 0.3
    ABS_RMS_MIN = 0.004
    FLOOR_FACTOR = 3.0

    def __init__(self, aggressiveness: int = 2, vad=None) -> None:
        self._vad = vad if vad is not None else webrtcvad.Vad(aggressiveness)
        self._floor = 0.002
        self._streak = 0
        self._last_voiced_at = -1e9
        self.last_db = SILENT_DB

    def process(self, frame: np.ndarray, now: float, floor_frozen: bool = False) -> bool:
        """Feed one 20 ms mono float32 frame; returns confirmed ``voiced``.

        ``floor_frozen`` must be True whenever the robot is audible: the
        echo of its own speech is handled by the EchoGuard and must not be
        learned as ambient noise — on hardware (2026-07-24) the floor
        climbed to the echo level during a long monologue and the user
        could no longer barge in at all.
        """
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
        self.last_db = 20.0 * math.log10(rms + 1e-9)
        # Asymmetric floor: falls fast in quiet, creeps up under sustained
        # sound — so steady motor noise stops counting as voice within a few
        # seconds, while real speech (which breathes every few hundred ms)
        # keeps pulling the floor back down.
        if not floor_frozen:
            if rms < self._floor:
                self._floor = 0.7 * self._floor + 0.3 * rms
            else:
                self._floor = min(self._floor * 1.01, rms)
            self._floor = max(self._floor, 5e-4)
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        raw = self._vad.is_speech(pcm16, 16_000) and rms > max(
            self.ABS_RMS_MIN, self._floor * self.FLOOR_FACTOR
        )
        if raw:
            self._streak += 1
        else:
            self._streak = 0
        voiced = self._streak >= self.CONFIRM_FRAMES
        if voiced:
            self._last_voiced_at = now
        return voiced

    def active(self, now: float) -> bool:
        return now - self._last_voiced_at < self.HANGOVER_S


class UplinkGain:
    """Automatic gain control for microphone audio sent to the model.

    XVF3800 capture runs far below full scale (user speech measures
    −33…−22 dB RMS on hardware, further suppressed during double-talk),
    and the duplex model's server-side listen decisions miss faint audio —
    after a barge-in it would resume its monologue as if the user had said
    nothing. Gain moves smoothly toward TARGET_RMS/rms, is only updated on
    speech-level chunks (never pumping up room noise), and is capped.
    """

    TARGET_RMS = 0.12
    MAX_GAIN = 8.0
    SPEECH_FLOOR_RMS = 0.006
    SMOOTHING = 0.3

    def __init__(self) -> None:
        self.gain = 1.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        if rms > self.SPEECH_FLOOR_RMS:
            target = min(self.MAX_GAIN, max(1.0, self.TARGET_RMS / rms))
            self.gain += (target - self.gain) * self.SMOOTHING
        return np.clip(chunk * self.gain, -1.0, 1.0).astype(np.float32)


class EchoGuard:
    """Rejects barge candidates that are the robot hearing its own speech.

    The far end is known exactly, so the expected post-AEC residual is
    ``playout_db + offset``: the offset (speaker → room → AEC leakage
    ratio) learns upward fast from echo-only frames, decays slowly, and is
    bumped after every false trigger the duck-verify stage catches — the
    same residual level then stops re-triggering. Values were tuned on
    hardware (2026-07-22 logs).
    """

    # Zero margin: XVF double-talk suppression pushes real interrupting
    # speech down to within a few dB of the residual prediction (hardware
    # log 2026-07-24: user at -27 dB vs a -22.9 dB threshold). A false duck
    # costs a 0.8 s dip; a missed user is deafness — bias accordingly.
    MARGIN_DB = 0.0
    # Hardware 2026-07-24: the steady-state residual sits well below -20 dB
    # (frame learning kept decaying the offset) and only TTS onset
    # transients spike to ~ -14 dB. Pinning the floor at -14 gated real
    # users out entirely — false-trigger bumps (+2.5 dB each, decaying at
    # 0.1 dB/s) are the intended defence against transients, because a
    # false duck now costs only a 0.8 s dip while a missed user is deafness.
    OFFSET_INIT_DB = -18.0
    OFFSET_RISE_DB = 0.5
    OFFSET_DECAY_DB = 0.002
    OFFSET_MIN_DB = -30.0
    FALSE_TRIGGER_BUMP_DB = 2.5

    def __init__(self) -> None:
        self.offset_db = self.OFFSET_INIT_DB

    def observe(self, mic_db: float, playout_db: float) -> bool:
        """Feed one mic frame taken while the robot is audible.

        Returns True when the level cannot be explained as our own echo.
        Frames below the prediction train the leakage offset.
        """
        if playout_db <= -90.0:
            return True  # nothing played recently: no echo to explain
        if mic_db >= playout_db + self.offset_db + self.MARGIN_DB:
            return True
        ratio = mic_db - playout_db
        if ratio > self.offset_db:
            self.offset_db = min(self.offset_db + self.OFFSET_RISE_DB, ratio)
        else:
            self.offset_db = max(self.offset_db - self.OFFSET_DECAY_DB, self.OFFSET_MIN_DB)
        return False

    def penalize(self) -> None:
        """A verified false trigger: raise the predicted leakage."""
        self.offset_db = min(self.offset_db + self.FALSE_TRIGGER_BUMP_DB, 0.0)


class StreamResampler:
    """Stateful linear resampler (24 kHz → 16 kHz) with phase continuity.

    Linear interpolation is transparent for 24→16 k speech and adds zero
    latency and zero dependencies — deliberately chosen over polyphase.
    """

    def __init__(self, rate_in: int = 24_000, rate_out: int = 16_000) -> None:
        self._step = rate_in / rate_out
        self.reset()

    def reset(self) -> None:
        self._tail = np.empty(0, np.float32)
        self._phase = 0.0

    def process(self, pcm: np.ndarray) -> np.ndarray:
        if len(pcm) == 0:
            return pcm.astype(np.float32)
        buf = np.concatenate([self._tail, pcm.astype(np.float32)])
        positions = np.arange(self._phase, len(buf) - 1, self._step)
        out = np.interp(positions, np.arange(len(buf)), buf).astype(np.float32)
        consumed = positions[-1] + self._step if len(positions) else self._phase
        keep = int(np.floor(consumed))
        self._phase = consumed - keep
        self._tail = buf[keep:]
        return out


class Speaker(threading.Thread):
    """Paced, interruptible, duckable playback of 24 kHz model audio.

    TTS units arrive roughly once per second with jitter, so streaming them
    straight to the device underruns audibly. An adaptive start-of-utterance
    preroll (0.25–0.8 s: +0.15 on underrun, ×0.9 on a clean turn) absorbs the
    jitter, while the device backlog is capped so flushes stay cheap.

    ``hold()`` silences the speaker *without* discarding the turn: the
    un-played device tail is stashed for a lossless ``release_hold()``.
    ``interrupt()`` is the destructive path (committed barge-in).

    Every pushed block is logged as ``(start, duration, dB)`` so the echo
    guard can query the playout envelope. The lookback must cover the full
    playout-to-capture latency — the shared GStreamer pipeline reports up
    to 1.256 s (hardware log 2026-07-22).
    """

    PREROLL_MIN, PREROLL_MAX = 0.25, 0.8
    BACKLOG_CAP_S = 1.2
    ENVELOPE_WINDOW_S = 1.6
    ENVELOPE_KEEP_S = 3.0
    TAIL_KEEP_SAMPLES = 2 * 16_000
    LOG_BLOCK_SAMPLES = 3_200  # 200 ms envelope resolution

    def __init__(self, media) -> None:
        super().__init__(name="yrobot-speaker", daemon=True)
        self._media = media
        self._q: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
        self._epoch = 0
        self._flush_to = 0
        self._resampler = StreamResampler()
        self._preroll = 0.3
        self._pending: deque[np.ndarray] = deque()
        self._buffered_s = 0.0
        self._pushed_until = 0.0  # monotonic time the device runs dry
        self._streaming = False  # preroll satisfied, turn is being played
        self._end_requested = False  # listen boundary seen for this turn
        self._turn_underrun = False
        self._holding = False
        self._hold_req = False
        self._resume_req = False
        self._held_tail: np.ndarray | None = None
        self._tail_ring = np.empty(0, np.float32)  # playback thread only
        self._push_log: deque[tuple[float, float, float]] = deque()
        self._log_lock = threading.Lock()
        self._halt = threading.Event()

    # -- called from other threads ----------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def holding(self) -> bool:
        return self._holding or self._hold_req

    def play(self, epoch: int, pcm_24k: np.ndarray) -> None:
        if epoch == self._epoch:
            self._q.put((epoch, pcm_24k))

    def interrupt(self) -> int:
        """Committed barge-in: advance the epoch and flush everything."""
        self._epoch += 1
        self._flush_to = self._epoch
        return self._epoch

    def hold(self) -> None:
        """Duck: silence the device now, keep the turn resumable."""
        self._hold_req = True

    def release_hold(self) -> None:
        """The duck was a false alarm: re-inject the held tail and go on."""
        self._resume_req = True

    def utterance_end(self) -> None:
        """Mark the turn boundary (a ``listen`` delta) — flush any short
        reply still waiting for preroll and adapt the preroll for next turn."""
        self._end_requested = True

    def audible(self, now: float | None = None) -> bool:
        """A turn is live somewhere between queue and device."""
        now = time.monotonic() if now is None else now
        return (
            self._pushed_until - now > 0.02
            or self._buffered_s > 0
            or self.holding
            or not self._q.empty()
        )

    def sounding(self, now: float | None = None) -> bool:
        """Sound is physically coming out (or held mid-duck) — the only
        states in which a barge candidate makes sense. During preroll the
        speaker is silent: there is no echo to guard against, and ducking
        would only delay the reply."""
        now = time.monotonic() if now is None else now
        return self._pushed_until - now > 0.02 or self.holding

    def playout_db(self, now: float) -> float:
        """Loudest block scheduled within the envelope lookback window."""
        loudest = SILENT_DB
        with self._log_lock:
            for start, duration, db in self._push_log:
                if start > now:
                    break
                if start + duration >= now - self.ENVELOPE_WINDOW_S:
                    loudest = max(loudest, db)
        return loudest

    def close(self) -> None:
        self._halt.set()

    # -- playback thread ----------------------------------------------------

    def run(self) -> None:
        flushed_at = 0
        while not self._halt.is_set():
            if self._flush_to > flushed_at:
                flushed_at = self._flush_to
                self._reset_turn()
                self._clear_device()
                continue
            if self._hold_req:
                self._hold_req = False
                if not self._holding:
                    self._duck()
            if self._resume_req:
                self._resume_req = False
                if self._holding:
                    self._holding = False
                    if self._held_tail is not None and len(self._held_tail):
                        self._push(self._held_tail)
                    self._held_tail = None

            try:
                epoch, pcm = self._q.get(timeout=0.05)
                if epoch == self._epoch:
                    self._pending.append(pcm)
                    self._buffered_s += len(pcm) / 24_000
            except queue.Empty:
                if not self._streaming and not self._pending:
                    # A boundary for a fully-discarded turn must not bypass
                    # the preroll of the next real turn.
                    self._end_requested = False

            if self._holding:
                continue  # the verify stage decides: resume or interrupt
            if not self._streaming and self._pending:
                if self._buffered_s >= self._preroll or self._end_requested:
                    self._streaming = True
            if self._streaming:
                self._stream_pending()
                self._check_turn_end()

    def _duck(self) -> None:
        self._holding = True
        now = time.monotonic()
        keep = min(len(self._tail_ring), int(max(0.0, self._pushed_until - now) * 16_000))
        self._held_tail = self._tail_ring[len(self._tail_ring) - keep :].copy() if keep else None
        self._clear_device()
        self._pushed_until = now
        self._truncate_log(now)

    def _stream_pending(self) -> None:
        while self._pending and self._flush_to <= self._epoch and not self._halt.is_set():
            now = time.monotonic()
            if self._pushed_until - now > self.BACKLOG_CAP_S:
                return  # revisit on the next loop tick; keep flushes cheap
            pcm = self._pending.popleft()
            self._buffered_s = max(self._buffered_s - len(pcm) / 24_000, 0.0)
            self._push(self._resampler.process(pcm))

    def _push(self, out: np.ndarray) -> None:
        """Push 16 kHz audio and log its envelope for the echo guard."""
        now = time.monotonic()
        self._media.push_audio_sample(out)
        start = max(self._pushed_until, now)
        with self._log_lock:
            for i in range(0, len(out), self.LOG_BLOCK_SAMPLES):
                block = out[i : i + self.LOG_BLOCK_SAMPLES]
                db = 20.0 * math.log10(float(np.sqrt(np.mean(np.square(block)))) + 1e-9)
                self._push_log.append((start + i / 16_000, len(block) / 16_000, db))
            horizon = now - self.ENVELOPE_KEEP_S
            while self._push_log and self._push_log[0][0] + self._push_log[0][1] < horizon:
                self._push_log.popleft()
        self._pushed_until = start + len(out) / 16_000
        self._tail_ring = np.concatenate([self._tail_ring, out])[-self.TAIL_KEEP_SAMPLES :]

    def _check_turn_end(self) -> None:
        if self._pending:
            return
        now = time.monotonic()
        if self._pushed_until > now:
            return
        # Device is dry and nothing is buffered: the turn either finished
        # cleanly (boundary seen) or genuinely underran mid-sentence.
        if self._end_requested:
            if not self._turn_underrun:
                self._preroll = max(self._preroll * 0.9, self.PREROLL_MIN)
            self._reset_turn(drain_queue=False)  # a next-turn chunk may already be queued
        else:
            if not self._turn_underrun:
                self._turn_underrun = True
                self._preroll = min(self._preroll + 0.15, self.PREROLL_MAX)
                logger.debug("underrun; preroll now %.2f s", self._preroll)
            self._streaming = False  # re-preroll the remainder of the turn

    def _clear_device(self) -> None:
        try:
            self._media.audio.clear_player()
        except Exception as exc:  # noqa: BLE001
            logger.warning("clear_player failed: %s", exc)

    def _truncate_log(self, now: float) -> None:
        """Drop scheduled-but-flushed blocks so the envelope stays physical."""
        with self._log_lock:
            kept = [
                (start, min(duration, now - start), db)
                for start, duration, db in self._push_log
                if start < now
            ]
            self._push_log.clear()
            self._push_log.extend(kept)

    def _reset_turn(self, drain_queue: bool = True) -> None:
        while drain_queue:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._pending.clear()
        self._buffered_s = 0.0
        self._pushed_until = 0.0
        self._streaming = False
        self._end_requested = False
        self._turn_underrun = False
        self._holding = False
        self._hold_req = False
        self._resume_req = False
        self._held_tail = None
        self._tail_ring = np.empty(0, np.float32)
        self._truncate_log(time.monotonic())
        self._resampler.reset()
