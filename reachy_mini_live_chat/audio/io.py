"""Audio transport: mic capture -> VAD/endpointer, and TTS -> speaker.

Owns two threads:
* **capture** — polls ``mini.media.get_audio_sample()``, downmixes to mono 16 kHz,
  runs echo control, and drives the :class:`Endpointer`. Handles barge-in.
* **playback** — drains ``bus.tts_audio`` and pushes to ``mini.media.push_audio_sample()``
  in 20 ms sub-chunks so a barge-in can cut speech off mid-sentence. Stamps end-to-end
  latency on the first chunk of each answer.
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
        from scipy.signal import resample_poly

        from math import gcd

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
        on_utterance_pcm: Callable[[np.ndarray, float], None],
    ) -> None:
        self.mini = mini
        self.cfg = cfg
        self.bus = bus
        self._on_utterance_pcm = on_utterance_pcm

        self.echo = EchoController(cfg.enable_aec, cfg.barge_in_energy)
        vad = build_vad(stub=cfg.stub)
        self.endpointer = Endpointer(
            vad,
            threshold=cfg.vad_threshold,
            silence_ms=cfg.vad_silence_ms,
            min_speech_ms=cfg.vad_min_speech_ms,
            on_speech_start=self._on_speech_start,
            on_utterance=self._on_utterance,
        )

        self._in_sr = TARGET_SR
        self._out_sr = TARGET_SR
        self._recent_loud = False
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
        log.info("AudioEngine started (in=%dHz out=%dHz)", self._in_sr, self._out_sr)

    def join(self) -> None:
        for t in self._threads:
            t.join(timeout=1.0)

    # -- capture ------------------------------------------------------------
    def _capture_loop(self) -> None:
        while not self.bus.stop_event.is_set():
            try:
                sample = self.mini.media.get_audio_sample()
            except Exception as e:
                log.debug("get_audio_sample error: %s", e)
                sample = None
            if sample is None or len(sample) == 0:
                time.sleep(0.005)
                continue
            mono = _to_mono(sample)
            mono = _resample(mono, self._in_sr, TARGET_SR)
            robot_speaking = self.bus.robot_speaking.is_set()
            # barge-in energy check on the *raw* mic (before ducking)
            self._recent_loud = self.echo.is_barge_in(mono, robot_speaking)
            clean = self.echo.process_capture(mono, robot_speaking)
            self.endpointer.process(clean)

    def _on_speech_start(self) -> None:
        if self.bus.robot_speaking.is_set():
            # Human started talking while the robot was speaking. Only treat as a real
            # barge-in if they're clearly louder than the residual echo.
            if not self._recent_loud:
                return
            log.info("barge-in detected")
            self.bus.request_interrupt()
        self.bus.user_speaking.set()
        self.bus.set_state(ConvState.LISTENING)
        self.bus.emit("system", {"text": "listening"})

    def _on_utterance(self, pcm: np.ndarray) -> None:
        self.bus.user_speaking.clear()
        t_end = time.monotonic()
        # ignore utterances that are essentially silence
        if float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) < 1e-4:
            self.bus.set_state(ConvState.IDLE)
            return
        self._on_utterance_pcm(pcm, t_end)

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
            if item is None:  # end-of-utterance sentinel
                idle_since = time.monotonic()
                continue
            self._play_chunk(item)
            idle_since = time.monotonic()

    def _play_chunk(self, chunk: np.ndarray) -> None:
        if self.bus.interrupt_event.is_set():
            return  # dropped: this belongs to an interrupted turn
        # first audio of the answer -> stamp end-to-end latency
        if not self.bus.pending_measured and self.bus.pending_t_end is not None:
            ms = self.bus.measure_e2e(self.bus.pending_t_end)
            self.bus.pending_measured = True
            log.info("e2e latency: %.0f ms", ms)
        self.bus.robot_speaking.set()
        chunk = _resample(chunk.astype(np.float32), TARGET_SR, self._out_sr)
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
