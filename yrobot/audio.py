"""Robot audio I/O: continuous AEC'd mic uplink, flushable speaker downlink.

Hard-won invariants (verified on hardware):
- The mic streams continuously, even while the robot speaks — barge-in and the
  model's own turn-taking both need it. Echo suppression is the XVF3800's job;
  the stream we read is already AEC'd, so an energy gate on it is the only
  robot-voice-free speech signal (the firmware "speech detected" flag is
  pre-AEC and stays high while the speaker plays — never use it for voice).
- The capture loop must drain *every* queued appsink buffer per poll or a
  stall builds a permanent backlog and the model hears seconds late.
- `clear_player()` cycles the shared GStreamer pipeline; only the playback
  thread (the sole pusher) may call it, or the pipeline can wedge for good.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np

from yrobot.config import Config
from yrobot.state import Shared

logger = logging.getLogger(__name__)

IN_SR = 16000


class StreamResampler:
    """Streaming model-rate → speaker-rate resampler (soxr, scipy fallback)."""

    def __init__(self, in_rate: int, out_rate: int):
        self.in_rate, self.out_rate = in_rate, out_rate
        self._soxr = None
        try:
            import soxr

            self._make = lambda: soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32")
            self._soxr = self._make()
        except ImportError:
            from scipy.signal import resample_poly

            self._poly = resample_poly
            logger.warning("soxr not available; per-chunk polyphase resampling")

    def process(self, pcm: np.ndarray) -> np.ndarray:
        if self.in_rate == self.out_rate:
            return pcm
        if self._soxr is not None:
            return self._soxr.resample_chunk(pcm)
        from math import gcd

        g = gcd(self.out_rate, self.in_rate)
        return self._poly(pcm, self.out_rate // g, self.in_rate // g).astype(np.float32)

    def reset(self) -> None:
        if self._soxr is not None:
            self._soxr = self._make()


class VoiceGate:
    """Adaptive energy gate: dip-tolerant onset, slow noise-floor tracking."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._floor = 0.005
        self._active = False
        self._above_ms = 0.0
        self._below_ms = 0.0

    @property
    def threshold(self) -> float:
        return max(self._floor * self._cfg.gate_ratio, self._cfg.gate_min_rms)

    def process(self, rms: float, dur_ms: float, robot_speaking: bool) -> bool:
        cfg = self._cfg
        threshold = self.threshold
        if robot_speaking and not self._active:
            threshold *= cfg.gate_barge_mult
        onset_ms = cfg.barge_onset_ms if robot_speaking else cfg.onset_ms

        if rms > threshold:
            self._above_ms += dur_ms
            self._below_ms = 0.0
        else:
            self._below_ms += dur_ms
            if not self._active:  # decay partial onsets instead of hard reset
                self._above_ms = max(0.0, self._above_ms - 2 * dur_ms)
            # Track the floor on sub-threshold audio only — rising slowly so
            # borderline speech can't inflate it, falling fast so the gate
            # recovers quickly when a noisy period ends.
            tau = cfg.floor_tau_s * (4.0 if rms > self._floor else 0.25)
            self._floor += (rms - self._floor) * min(1.0, dur_ms / 1000.0 / tau)

        if not self._active and self._above_ms >= onset_ms:
            self._active = True
        elif self._active and self._below_ms >= cfg.release_ms:
            self._active = False
            self._above_ms = 0.0
        return self._active


class UplinkGain:
    """Gain-up-only AGC so quiet speakers aren't dismissed as background."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._speech_rms = cfg.agc_target_rms  # start at unity gain
        self.gain = 1.0

    def observe(self, rms: float, voice: bool, robot_speaking: bool) -> None:
        if voice and not robot_speaking and rms > 1e-4:
            self._speech_rms += (rms - self._speech_rms) * 0.05
            self.gain = float(np.clip(self._cfg.agc_target_rms / self._speech_rms,
                                      1.0, self._cfg.agc_max_gain))

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        return np.clip(chunk * self.gain, -1.0, 1.0) if self.gain > 1.001 else chunk


class AudioIO:
    """Capture and playback threads around the ReachyMini media manager."""

    def __init__(
        self,
        cfg: Config,
        media,
        state: Shared,
        on_chunk: Callable[[np.ndarray, bool], None],
        on_voice_edge: Callable[[bool], bool],
    ):
        """on_chunk(mono f32 @16 kHz, is_flush); on_voice_edge(active) -> flush partial now?"""
        self._cfg = cfg
        self._media = media
        self._state = state
        self._on_chunk = on_chunk
        self._on_voice_edge = on_voice_edge
        self._gate = VoiceGate(cfg)
        self._agc = UplinkGain(cfg)
        self._chunk_len = IN_SR * cfg.chunk_ms // 1000
        self._stop = threading.Event()
        self._playq: "queue.Queue[np.ndarray]" = queue.Queue()
        self._clear = threading.Event()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="yrobot-capture", daemon=True),
            threading.Thread(target=self._playback_loop, name="yrobot-playback", daemon=True),
        ]

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)

    # -- downlink ------------------------------------------------------------

    def play(self, pcm16k: np.ndarray) -> None:
        """Queue model speech (mono f32 @ speaker rate) for playback."""
        self._playq.put(pcm16k)

    def clear_playback(self) -> None:
        """Flush queued + in-pipeline audio; executed by the playback thread."""
        self._clear.set()
        self._state.play_head = 0.0  # silence the speaking flag immediately

    def _playback_loop(self) -> None:
        media = self._media
        while not self._stop.is_set():
            if self._clear.is_set():
                try:
                    media.audio.clear_player()
                except Exception:
                    logger.exception("clear_player failed")
                while True:
                    try:
                        self._playq.get_nowait()
                    except queue.Empty:
                        break
                self._clear.clear()
                continue
            try:
                pcm = self._playq.get(timeout=0.05)
            except queue.Empty:
                continue
            if self._clear.is_set():
                continue
            media.push_audio_sample(pcm)
            now = time.monotonic()
            self._state.play_head = max(self._state.play_head, now) + len(pcm) / IN_SR

    # -- uplink --------------------------------------------------------------

    def _capture_loop(self) -> None:
        cfg, state = self._cfg, self._state
        pending: list[np.ndarray] = []
        buffered = 0
        voice = False
        last_doa_poll = 0.0

        def flush(n: int, is_flush: bool) -> None:
            nonlocal pending, buffered
            data = np.concatenate(pending) if len(pending) > 1 else pending[0]
            chunk, rest = data[:n], data[n:]
            pending = [rest] if len(rest) else []
            buffered = len(rest)
            self._on_chunk(self._agc.apply(chunk), is_flush)

        while not self._stop.is_set():
            if cfg.doa and time.monotonic() - last_doa_poll > 0.2:
                last_doa_poll = time.monotonic()
                self._poll_doa()
            sample = self._media.get_audio_sample()
            if sample is None:
                time.sleep(0.005)
                continue

            mono = sample.mean(axis=1).astype(np.float32)
            dur_ms = len(mono) * 1000.0 / IN_SR
            rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
            speaking = state.robot_speaking()
            active = self._gate.process(rms, dur_ms, speaking)
            self._agc.observe(rms, active, speaking)

            pending.append(mono)
            buffered += len(mono)

            if active != voice:
                voice = active
                state.voice_active = active
                t = time.monotonic()
                if active:
                    state.last_voice_onset = t
                else:
                    state.last_voice_end = t
                # An interrupt ships whatever real speech we already hold, at
                # once, with force_listen — the model must hear the user now.
                if self._on_voice_edge(active) and buffered >= IN_SR // 10:
                    flush(buffered, True)

            while buffered >= self._chunk_len:
                flush(self._chunk_len, False)

    def _poll_doa(self) -> None:
        """Single owner of the DoA USB read; motion consumes the cached target."""
        try:
            res = self._media.get_DoA()
        except Exception:
            return
        if res is None:
            return
        angle, speech = res
        state = self._state
        # The firmware flag only validates the angle (it is pre-AEC): use it
        # while the user speaks and the robot is quiet.
        if not (speech and state.voice_active and not state.robot_speaking()):
            return
        yaw_err = np.pi / 2 - angle  # 0=left, π/2=front, π=right → signed error
        if abs(yaw_err) < self._cfg.doa_min_turn_rad:
            return
        target = float(np.clip(state.body_yaw + yaw_err, -2.7, 2.7))
        self._state.yaw_target = target
