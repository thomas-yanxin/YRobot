"""Audio transport: mic capture → 1 s chunks for the omni brain, and omni audio → speaker.

Owns two threads:
* **capture** — polls ``mini.media.get_audio_sample()``, downmixes to mono 16 kHz, runs
  echo suppression, then (a) feeds a lightweight VAD that only drives ``user_speaking``
  (→ DOA head-turn + attentive mood + barge-in), and (b) accumulates the cleaned signal
  into fixed ~1 s chunks handed to ``on_audio_chunk`` — the continuous full-duplex uplink.
* **playback** — drains ``bus.tts_audio`` (float32 @ 16 kHz) and pushes to
  ``mini.media.push_audio_sample()`` in 20 ms sub-chunks so a barge-in can cut the robot
  off mid-word. Stamps end-to-end latency on the first chunk after the user stops talking.

We stream the mic **continuously** (even while the robot speaks) so the model can hear a
barge-in — which is exactly why the capture path runs echo suppression first, so the model
doesn't hear the robot's own voice.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from ..bus import Bus, ConvState
from ..config import Config
from .aec import EchoController
from .vad import Endpointer, build_vad

log = logging.getLogger("live_chat.audio")

TARGET_SR = 16000
SUBCHUNK = int(TARGET_SR * 0.02)  # 20 ms


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.mean(axis=1).astype(np.float32)
    return x.astype(np.float32)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n = int(round(len(x) * sr_out / sr_in))
        idx = np.linspace(0, len(x) - 1, n).astype(np.int64)
        return x[idx]


class AudioEngine:
    def __init__(
        self,
        mini,
        cfg: Config,
        bus: Bus,
        on_audio_chunk: Callable[[np.ndarray], None],
    ) -> None:
        self.mini = mini
        self.cfg = cfg
        self.bus = bus
        self._on_audio_chunk = on_audio_chunk

        self.echo = EchoController(cfg.enable_aec, cfg.barge_in_energy)
        vad = build_vad(stub=cfg.stub, backend=cfg.vad_backend)
        self.endpointer = Endpointer(
            vad,
            threshold=cfg.vad_threshold,
            silence_ms=cfg.vad_silence_ms,
            min_speech_ms=cfg.vad_min_speech_ms,
            on_speech_start=self._on_speech_start,
            on_utterance=self._on_speech_end,
        )

        self._in_sr = TARGET_SR
        self._out_sr = TARGET_SR
        # tts_audio carries raw omni output (float32 at the model's TTS rate); we resample
        # to the device rate here on the playback thread — once, and off the WS read path.
        self._tts_sr = int(cfg.omni_out_sr or TARGET_SR)
        self._recent_loud = False
        self._chunk_samples = max(1, int(TARGET_SR * cfg.omni_chunk_ms / 1000))
        self._chunk_buf = np.zeros(0, dtype=np.float32)
        self._threads: list[threading.Thread] = []

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        try:
            self._in_sr = int(self.mini.media.get_input_audio_samplerate() or TARGET_SR)
            self._out_sr = int(self.mini.media.get_output_audio_samplerate() or TARGET_SR)
        except Exception:
            pass
        self.mini.media.start_recording()
        self.mini.media.start_playing()
        for target in (self._capture_loop, self._playback_loop):
            t = threading.Thread(target=target, name=target.__name__, daemon=True)
            t.start()
            self._threads.append(t)
        log.info("AudioEngine started (in=%dHz out=%dHz, chunk=%dms)",
                 self._in_sr, self._out_sr, self.cfg.omni_chunk_ms)

    def join(self) -> None:
        for t in self._threads:
            t.join(timeout=1.0)

    # -- capture ------------------------------------------------------------
    def _capture_loop(self) -> None:
        # Poll on a fixed ~10 ms cadence instead of spinning: the mic buffers audio, so
        # reading every 10 ms batches ~10 ms per call and frees a whole core on the CM4
        # (a hot poll loop was a big part of the CPU starvation → choppy playback).
        poll_dt = 0.01
        while not self.bus.stop_event.is_set():
            t0 = time.monotonic()
            try:
                sample = self.mini.media.get_audio_sample()
            except Exception as e:
                log.debug("get_audio_sample error: %s", e)
                sample = None
            if sample is not None and len(sample):
                mono = _to_mono(sample)
                mono = _resample(mono, self._in_sr, TARGET_SR)
                robot_speaking = self.bus.robot_speaking.is_set()
                # barge-in energy check on the *raw* mic (before ducking)
                self._recent_loud = self.echo.is_barge_in(mono, robot_speaking)
                clean = self.echo.process_capture(mono, robot_speaking)
                # (a) local VAD only sets user_speaking → DOA / mood / barge-in
                self.endpointer.process(clean)
                # (b) continuous 1 s chunks → the omni uplink
                self._accumulate(clean)
            time.sleep(max(0.0, poll_dt - (time.monotonic() - t0)))

    def _accumulate(self, clean: np.ndarray) -> None:
        self._chunk_buf = np.concatenate([self._chunk_buf, clean.astype(np.float32)])
        while len(self._chunk_buf) >= self._chunk_samples:
            chunk = self._chunk_buf[: self._chunk_samples]
            self._chunk_buf = self._chunk_buf[self._chunk_samples:]
            try:
                self._on_audio_chunk(chunk)
            except Exception as e:
                log.debug("on_audio_chunk error: %s", e)

    def _on_speech_start(self) -> None:
        if self.bus.robot_speaking.is_set():
            # Human started while the robot spoke → only a real barge-in if clearly
            # louder than the residual echo.
            if not self._recent_loud:
                return
            log.info("barge-in detected")
            self.bus.request_interrupt()
        self.bus.user_speaking.set()
        self.bus.set_state(ConvState.LISTENING)
        self.bus.emit("system", {"text": "listening"})

    def _on_speech_end(self, pcm: np.ndarray) -> None:
        # We don't consume the utterance PCM (the omni server does the ASR); this only
        # marks the end of human speech for DOA decay + the e2e latency clock.
        self.bus.user_speaking.clear()
        self.bus.pending_t_end = time.monotonic()
        self.bus.pending_measured = False
        # If we cut the robot off (barge-in), recover: allow future audio to play and
        # leave the INTERRUPTED state so motion stops the speaking wobble.
        if self.bus.interrupt_event.is_set() or self.bus.state == ConvState.INTERRUPTED:
            self.bus.clear_interrupt()
            if self.bus.state == ConvState.INTERRUPTED:
                self.bus.set_state(ConvState.IDLE)

    # -- playback -----------------------------------------------------------
    def _playback_loop(self) -> None:
        idle_since = time.monotonic()
        while not self.bus.stop_event.is_set():
            try:
                item = self.bus.tts_audio.get(timeout=0.1)
            except Exception:
                if self.bus.robot_speaking.is_set() and time.monotonic() - idle_since > 0.25:
                    self.bus.robot_speaking.clear()
                    self.bus.emit("system", {"text": "spoke"})
                continue
            if item is None:  # end-of-turn sentinel
                idle_since = time.monotonic()
                continue
            self._play_chunk(item)
            idle_since = time.monotonic()

    def _play_chunk(self, chunk: np.ndarray) -> None:
        if self.bus.interrupt_event.is_set():
            return  # dropped: belongs to an interrupted turn
        # first audio after the user stopped → stamp end-to-end latency
        if not self.bus.pending_measured and self.bus.pending_t_end is not None:
            ms = self.bus.measure_e2e(self.bus.pending_t_end)
            self.bus.pending_measured = True
            log.info("e2e latency (user stop → first audio): %.0f ms", ms)
        self.bus.robot_speaking.set()
        chunk = _resample(chunk.astype(np.float32), self._tts_sr, self._out_sr)
        for i in range(0, len(chunk), SUBCHUNK):
            if self.bus.interrupt_event.is_set():
                break
            sub = chunk[i:i + SUBCHUNK]
            self.echo.note_playback(sub)
            try:
                self.mini.media.push_audio_sample(sub.reshape(-1, 1))
            except Exception as e:
                log.debug("push_audio_sample error: %s", e)
            time.sleep(len(sub) / self._out_sr)
