"""Audio capture, voice detection and interruptible playback.

Reachy Mini Wireless routes the microphone through the XVF3800's hardware
acoustic echo canceller, so ``get_audio_sample()`` already has the robot's
own speech removed — but motor noise is not echo and still reaches the VAD.
``VoiceDetector`` therefore stacks three guards: WebRTC VAD, an adaptive
RMS noise floor, and a 3-frame (60 ms) confirmation streak.

Playback is epoch-tagged: a barge-in bumps the epoch, and everything queued
under an older epoch is dropped while the GStreamer pipeline is flushed via
``clear_player()`` — always from the playback thread, which is the only
thread allowed to touch the pipeline.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)

FRAME_MS = 20
FRAME_SAMPLES = 16_000 * FRAME_MS // 1000  # 320


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

    def process(self, frame: np.ndarray, now: float) -> bool:
        """Feed one 20 ms mono float32 frame; returns confirmed ``voiced``."""
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
        # Asymmetric floor: falls fast in quiet, creeps up under sustained
        # sound — so steady motor noise stops counting as voice within a few
        # seconds, while real speech (which breathes every few hundred ms)
        # keeps pulling the floor back down.
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
    """Paced, interruptible playback of 24 kHz model audio.

    TTS units arrive roughly once per second with jitter, so streaming them
    straight to the device underruns audibly. An adaptive start-of-utterance
    preroll (0.25–0.8 s: +0.15 on underrun, ×0.9 on a clean turn) absorbs the
    jitter, while the device backlog is capped so a barge-in flush is cheap.
    """

    PREROLL_MIN, PREROLL_MAX = 0.25, 0.8
    BACKLOG_CAP_S = 1.2

    def __init__(self, media) -> None:
        super().__init__(name="yrobot-speaker", daemon=True)
        self._media = media
        self._q: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
        self._epoch = 0
        self._flush_to = 0
        self._resampler = StreamResampler()
        self._preroll = 0.4
        self._pending: deque[np.ndarray] = deque()
        self._buffered_s = 0.0
        self._pushed_until = 0.0  # monotonic time the device runs dry
        self._streaming = False  # preroll satisfied, turn is being played
        self._end_requested = False  # listen boundary seen for this turn
        self._turn_underrun = False
        self._halt = threading.Event()

    # -- called from other threads ----------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    def play(self, epoch: int, pcm_24k: np.ndarray) -> None:
        if epoch == self._epoch:
            self._q.put((epoch, pcm_24k))

    def interrupt(self) -> int:
        """Barge-in: advance the epoch and flush everything. Returns it."""
        self._epoch += 1
        self._flush_to = self._epoch
        return self._epoch

    def utterance_end(self) -> None:
        """Mark the turn boundary (a ``listen`` delta) — flush any short
        reply still waiting for preroll and adapt the preroll for next turn."""
        self._end_requested = True

    def audible(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self._pushed_until - now > 0.02 or self._buffered_s > 0 or not self._q.empty()

    def close(self) -> None:
        self._halt.set()

    # -- playback thread ----------------------------------------------------

    def run(self) -> None:
        flushed_at = 0
        while not self._halt.is_set():
            if self._flush_to > flushed_at:
                flushed_at = self._flush_to
                self._reset_turn()
                try:
                    self._media.audio.clear_player()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("clear_player failed: %s", exc)
                continue

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

            if not self._streaming and self._pending:
                if self._buffered_s >= self._preroll or self._end_requested:
                    self._streaming = True
            if self._streaming:
                self._stream_pending()
                self._check_turn_end()

    def _stream_pending(self) -> None:
        while self._pending and self._flush_to <= self._epoch and not self._halt.is_set():
            now = time.monotonic()
            backlog = self._pushed_until - now
            if backlog > self.BACKLOG_CAP_S:
                return  # revisit on the next loop tick; keep flushes cheap
            pcm = self._pending.popleft()
            out = self._resampler.process(pcm)
            self._buffered_s = max(self._buffered_s - len(pcm) / 24_000, 0.0)
            self._media.push_audio_sample(out)
            self._pushed_until = max(self._pushed_until, now) + len(out) / 16_000

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
        self._resampler.reset()
