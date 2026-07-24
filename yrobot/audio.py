"""Audio capture, voice detection and interruptible playback.

Reachy Mini Wireless routes the microphone through the XVF3800 hardware
front end.  The board defaults are not suitable for far-field double-talk,
so startup applies the same verified AGC/AEC/noise-suppression profile as
Pollen's conversation app before local VAD starts. Residual speech-shaped
echo is rejected against the exact scheduled playout waveform rather than a
static level threshold, preserving weak near-end double-talk.

Playback is epoch-tagged: a barge-in bumps the epoch, and everything queued
under an older epoch is dropped while the GStreamer pipeline is flushed via
``clear_player()`` — always from the playback thread, the only thread allowed
to touch the pipeline.  Remote ``force_listen`` is handled separately by the
turn controller; local silence never waits for a network acknowledgement.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import webrtcvad

logger = logging.getLogger(__name__)

FRAME_MS = 20
FRAME_SAMPLES = 16_000 * FRAME_MS // 1000  # 320
SILENT_DB = -120.0
WRITE_SETTLE_SECONDS = 0.1

# Pollen's Reachy Mini conversation app applies this exact profile before
# starting its realtime record/play loops.  In particular the hardware AGC
# and double-talk settings operate before our software VAD; outbound AGC is
# too late to recover a quiet interjection that VAD has already rejected.
AUDIO_STARTUP_CONFIG: tuple[tuple[str, tuple[float | int, ...]], ...] = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)


def apply_audio_startup_config(
    media,
    *,
    verify: bool = True,
    write_settle_seconds: float = WRITE_SETTLE_SECONDS,
) -> bool:
    """Best-effort application of the wireless XVF3800 duplex profile."""
    audio = getattr(media, "audio", None)
    apply_config = getattr(audio, "apply_audio_config", None)
    if not callable(apply_config):
        logger.warning("Reachy audio config API unavailable; continuing with board defaults")
        return False
    try:
        applied = bool(
            apply_config(
                AUDIO_STARTUP_CONFIG,
                verify=verify,
                write_settle_seconds=write_settle_seconds,
            )
        )
    except Exception as exc:  # noqa: BLE001 — audio startup remains best effort
        logger.warning("Reachy duplex audio config failed: %s", exc)
        return False
    if applied:
        logger.info("Applied Reachy XVF3800 duplex audio profile")
    else:
        logger.warning("Reachy XVF3800 duplex audio profile was not applied")
    return applied


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
    ``last_db`` — level of the most recent frame, used in interruption logs.
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

    @property
    def streak(self) -> int:
        """Consecutive raw-voiced frames — longer streaks reject impulsive
        servo knocks (the head motors sit centimetres from the mic array)."""
        return self._streak

    def process(self, frame: np.ndarray, now: float, floor_frozen: bool = False) -> bool:
        """Feed one 20 ms mono float32 frame; returns confirmed ``voiced``.

        ``floor_frozen`` must be True whenever the robot is audible. Its
        post-AEC residual is not ambient noise: learning it raised the floor
        during long replies until real double-talk could no longer pass.
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


@dataclass(frozen=True)
class EchoMatch:
    """How much a microphone window can be explained by recent playout."""

    similarity: float = 0.0
    unexplained_db: float = SILENT_DB
    lag_ms: float = 0.0


class PlaybackEchoMatcher:
    """Match a voiced microphone window against exact recent speaker PCM.

    XVF3800 AEC removes most far-end energy, but the remaining waveform still
    follows the exact audio sent to the speaker. A normalized correlation is
    searched over the whole capture/pipeline latency range instead of assuming
    one fixed delay. The best linear match also estimates how much microphone
    energy remains unexplained; genuine double-talk retains near-end energy
    even when some far-end correlation is present.

    Matching is decimated to 2 kHz after box filtering. This retains enough
    speech structure for echo identification while keeping the 20 ms capture
    loop bounded on the Reachy Mini CM4.
    """

    SAMPLE_RATE = 16_000
    HISTORY_S = 2.0
    DECIMATION = 8
    MIN_REFERENCE_RMS = 1e-4

    def __init__(self) -> None:
        self._blocks: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()

    def record(self, start: float, pcm_16k: np.ndarray) -> None:
        """Record one block at its scheduled physical playout time."""
        pcm = np.asarray(pcm_16k, dtype=np.float32).reshape(-1).copy()
        if len(pcm) == 0:
            return
        cutoff = start - self.HISTORY_S - 0.5
        with self._lock:
            self._blocks.append((start, pcm))
            while self._blocks:
                block_start, block = self._blocks[0]
                block_end = block_start + len(block) / self.SAMPLE_RATE
                if block_end >= cutoff:
                    break
                self._blocks.popleft()

    def match(self, mic_16k: np.ndarray, now: float) -> EchoMatch:
        """Return the strongest delayed far-end match for ``mic_16k``."""
        mic = np.asarray(mic_16k, dtype=np.float32).reshape(-1)
        if len(mic) < FRAME_SAMPLES:
            return EchoMatch()

        origin = now - self.HISTORY_S
        reference = np.zeros(round(self.HISTORY_S * self.SAMPLE_RATE), np.float32)
        with self._lock:
            blocks = tuple(self._blocks)
        for start, block in blocks:
            dst_start = round((start - origin) * self.SAMPLE_RATE)
            src_start = max(0, -dst_start)
            dst_start = max(0, dst_start)
            count = min(len(block) - src_start, len(reference) - dst_start)
            if count > 0:
                reference[dst_start : dst_start + count] = block[src_start : src_start + count]

        ref = self._prepare(reference)
        probe = self._prepare(mic)
        if len(probe) == 0 or len(ref) < len(probe):
            return EchoMatch()
        probe_energy = float(np.dot(probe, probe))
        if probe_energy <= 1e-12:
            return EchoMatch()

        # np.correlate is implemented in C; after 8x decimation this is a
        # small bounded search (~1M multiply-adds for a 200 ms candidate).
        correlation = np.correlate(ref, probe, mode="valid")
        squared = np.square(ref, dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(squared)))
        window_energy = cumulative[len(probe) :] - cumulative[: -len(probe)]
        minimum_energy = (self.MIN_REFERENCE_RMS**2) * len(probe)
        denominator = np.sqrt(np.maximum(window_energy, minimum_energy) * probe_energy)
        scores = np.divide(
            np.abs(correlation),
            denominator,
            out=np.zeros_like(correlation, dtype=np.float64),
            where=window_energy >= minimum_energy,
        )
        best_index = int(np.argmax(scores))
        similarity = min(1.0, float(scores[best_index]))

        mic_rms = float(np.sqrt(np.mean(np.square(mic, dtype=np.float64))))
        unexplained_rms = mic_rms * math.sqrt(max(0.0, 1.0 - similarity * similarity))
        unexplained_db = 20.0 * math.log10(unexplained_rms + 1e-9)
        matched_end = origin + ((best_index + len(probe)) * self.DECIMATION / self.SAMPLE_RATE)
        return EchoMatch(
            similarity=similarity,
            unexplained_db=unexplained_db,
            lag_ms=max(0.0, (now - matched_end) * 1000.0),
        )

    @classmethod
    def _prepare(cls, pcm: np.ndarray) -> np.ndarray:
        """Low-pass, decimate and high-pass one correlation vector."""
        usable = len(pcm) // cls.DECIMATION * cls.DECIMATION
        if usable == 0:
            return np.empty(0, np.float32)
        downsampled = pcm[:usable].reshape(-1, cls.DECIMATION).mean(axis=1)
        prepared = np.diff(downsampled, prepend=downsampled[0])
        prepared -= float(np.mean(prepared))
        return prepared.astype(np.float32, copy=False)


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
    PLAYBACK_RELEASE = 0.2

    def __init__(self) -> None:
        self.gain = 1.0

    def process(
        self,
        chunk: np.ndarray,
        *,
        playback_active: bool = False,
        confirmed_user_voice: bool = False,
    ) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        if playback_active and not confirmed_user_voice:
            # Do not preserve a large user-learned gain while only our own
            # AEC residual is present. It can otherwise make the model hear
            # and answer the robot itself.
            self.gain += (1.0 - self.gain) * self.PLAYBACK_RELEASE
        elif rms > self.SPEECH_FLOOR_RMS:
            target = min(self.MAX_GAIN, max(1.0, self.TARGET_RMS / rms))
            self.gain += (target - self.gain) * self.SMOOTHING
        return np.clip(chunk * self.gain, -1.0, 1.0).astype(np.float32)


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
    """Paced, epoch-interruptible playback of 24 kHz model audio.

    TTS units arrive roughly once per second with jitter, so streaming them
    straight to the device underruns audibly. An adaptive start-of-utterance
    preroll (0.25–0.8 s: +0.15 on underrun, ×0.9 on a clean turn) absorbs the
    jitter, while the device backlog is capped so flushes stay cheap.

    ``interrupt()`` advances the epoch before asking the playback thread to
    clear the SDK player, so both already-buffered audio and late deltas from
    the old turn are discarded. There is deliberately no resumable duck:
    once local VAD qualifies a user interruption, old speech must not return.
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
        self._preroll = 0.3
        self._pending: deque[np.ndarray] = deque()
        self._buffered_s = 0.0
        self._pushed_until = 0.0  # monotonic time the device runs dry
        self._streaming = False  # preroll satisfied, turn is being played
        self._end_requested = False  # listen boundary seen for this turn
        self._turn_underrun = False
        self._command_lock = threading.Lock()
        self._flush_event = threading.Event()
        self._flush_requested_at = 0.0
        self._flushed_epoch = 0
        self._flush_condition = threading.Condition()
        self._halt = threading.Event()
        self._echo_matcher = PlaybackEchoMatcher()

    # -- called from other threads ----------------------------------------

    @property
    def epoch(self) -> int:
        with self._command_lock:
            return self._epoch

    def play(self, epoch: int, pcm_24k: np.ndarray) -> None:
        with self._command_lock:
            if epoch == self._epoch:
                self._q.put((epoch, pcm_24k))

    def interrupt(self) -> int:
        """Committed barge-in: advance the epoch and flush everything."""
        with self._command_lock:
            self._epoch += 1
            self._flush_to = self._epoch
            epoch = self._epoch
            self._flush_requested_at = time.monotonic()
            self._flush_event.set()
            return epoch

    def wait_flushed(self, epoch: int, timeout: float | None = None) -> bool:
        """Wait until the playback thread has cleared through ``epoch``."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._flush_condition:
            while self._flushed_epoch < epoch:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._flush_condition.wait(remaining)
            return True

    def utterance_end(self) -> None:
        """Mark the turn boundary (a ``listen`` delta) — flush any short
        reply still waiting for preroll and adapt the preroll for next turn."""
        self._end_requested = True

    def audible(self, now: float | None = None) -> bool:
        """A turn is live somewhere between queue and device."""
        now = time.monotonic() if now is None else now
        return self._pushed_until - now > 0.02 or self._buffered_s > 0 or not self._q.empty()

    def sounding(self, now: float | None = None) -> bool:
        """Whether audio is physically scheduled on the SDK player."""
        now = time.monotonic() if now is None else now
        return self._pushed_until - now > 0.02

    def playing(self, now: float | None = None) -> bool:
        """Whether audio is physically scheduled on the SDK player."""
        now = time.monotonic() if now is None else now
        return self._pushed_until - now > 0.02

    def echo_match(self, mic_16k: np.ndarray, now: float) -> EchoMatch:
        """Compare a microphone candidate with recent scheduled playout."""
        return self._echo_matcher.match(mic_16k, now)

    def close(self) -> None:
        self._halt.set()
        self._flush_event.set()

    # -- playback thread ----------------------------------------------------

    def run(self) -> None:
        flushed_at = 0
        while not self._halt.is_set():
            if self._flush_event.is_set():
                with self._command_lock:
                    flushed_at = self._flush_to
                    requested_at = self._flush_requested_at
                    self._flush_event.clear()
                self._reset_turn()
                self._clear_device()
                if requested_at:
                    logger.info(
                        "playback epoch %d cleared locally in %.0f ms",
                        flushed_at,
                        (time.monotonic() - requested_at) * 1000,
                    )
                with self._flush_condition:
                    self._flushed_epoch = max(self._flushed_epoch, flushed_at)
                    self._flush_condition.notify_all()
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
        while self._pending and not self._flush_event.is_set() and not self._halt.is_set():
            now = time.monotonic()
            if self._pushed_until - now > self.BACKLOG_CAP_S:
                return  # revisit on the next loop tick; keep flushes cheap
            pcm = self._pending.popleft()
            self._buffered_s = max(self._buffered_s - len(pcm) / 24_000, 0.0)
            self._push(self._resampler.process(pcm))

    def _push(self, out: np.ndarray) -> None:
        """Push one 16 kHz block, recording its physical echo reference."""
        now = time.monotonic()
        self._media.push_audio_sample(out)
        start = max(self._pushed_until, now)
        self._echo_matcher.record(start, out)
        self._pushed_until = start + len(out) / 16_000

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
