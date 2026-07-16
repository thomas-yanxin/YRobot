"""Audio transport: mic capture → 1 s chunks for the omni brain, and omni audio → speaker.

Owns two threads:
* **capture** — polls ``mini.media.get_audio_sample()``, downmixes to mono 16 kHz, then
  (a) refreshes the XVF3800 firmware's post-AEC voice flag (~20 Hz USB poll) that drives
  ``user_speaking`` (→ DOA head-turn + attentive mood + barge-in), and (b) accumulates the
  signal into fixed chunks handed to ``on_audio_chunk`` — the continuous full-duplex uplink.
* **playback** — drains ``bus.tts_audio`` (float32 @ 16 kHz) and pushes to
  ``mini.media.push_audio_sample()`` in 20 ms sub-chunks so a barge-in can cut the robot
  off mid-word. Stamps end-to-end latency on the first chunk after the user stops talking.

Echo cancellation is done in hardware by the ReSpeaker XVF3800 (see ``respeaker_config.py``),
so the mic is already echo-free — we stream it **continuously** (even while the robot speaks)
so the model can hear a barge-in, exactly like the official Reachy Mini app.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable

import numpy as np

from ..bus import Bus, ConvState
from ..config import Config
from .respeaker_config import apply_startup_config
from .vad import EnergyVad, Endpointer, UplinkAgc

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

        # Voice activity = adaptive energy gate over the AEC'd mic stream — the only
        # signal free of the robot's own voice, so it works during playback (barge-in).
        # The firmware's own speech flag is pre-AEC (hears the robot itself) and is
        # NOT used for voice decisions — only its DOA angle is (see _poll_hw_doa).
        self.endpointer = Endpointer(
            EnergyVad(),
            threshold=cfg.vad_threshold,
            silence_ms=cfg.vad_silence_ms,
            min_speech_ms=cfg.vad_min_speech_ms,
            on_speech_start=self._on_speech_start,
            on_utterance=self._on_speech_end,
        )
        self._hw_last = 0.0       # time of the last get_DoA USB read (throttle)
        self._mic_gain = float(getattr(cfg, "omni_mic_gain", 1.0) or 1.0)
        # Uplink-only software AGC (fixes "too quiet → model never answers"). The VAD
        # always sees the un-AGC'd signal so its adaptive floor never chases our gain.
        self._agc = (
            UplinkAgc(cfg.omni_mic_agc_target, cfg.omni_mic_agc_max_gain)
            if getattr(cfg, "omni_mic_agc", True) else None
        )
        self._in_sr = TARGET_SR
        self._out_sr = TARGET_SR
        # playback pacing (set once out_sr is known in start())
        self._play_sub_n = SUBCHUNK
        self._play_cushion = max(0.0, cfg.omni_playback_cushion_ms / 1000.0)
        self._turn_start = None       # wall-clock start of the current spoken turn
        self._turn_played = 0.0       # seconds of audio pushed this turn
        # tts_audio carries raw omni output (float32 at the model's TTS rate); we resample
        # to the device rate here on the playback thread — once, and off the WS read path.
        self._tts_sr = int(cfg.omni_out_sr or TARGET_SR)
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
        self._play_sub_n = max(1, int(self._out_sr * self.cfg.omni_playback_chunk_ms / 1000))
        self.mini.media.start_recording()
        self.mini.media.start_playing()
        # Tune the ReSpeaker XVF3800 (hardware AEC/NS/AGC) AFTER the media pipelines are
        # up — the official app writes it ~1 s after pipeline start, because (re)opening
        # the audio device can reset the chip to defaults. Writing before start_recording
        # risks the tuning being clobbered, leaving echo/AGC at chip defaults. Done here
        # synchronously (before the capture thread's DoA polling touches USB).
        if getattr(self.cfg, "omni_respeaker_config", True):
            time.sleep(0.5)  # let the pipelines settle, mirroring the official app
            apply_startup_config(self.mini)
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
            # Drain EVERYTHING the appsink has queued, not one buffer per poll.
            # get_audio_sample() returns a single buffer; if we ever fall behind
            # (CPU spike, pipeline pause), a one-per-poll reader can never catch
            # up by more than ~2x, so the backlog — and with it a permanent mic
            # delay that breaks barge-in — persists forever.
            parts = []
            try:
                while True:
                    sample = self.mini.media.get_audio_sample()
                    if sample is None or not len(sample):
                        break
                    parts.append(sample)
            except Exception as e:
                log.debug("get_audio_sample error: %s", e)
            if parts:
                sample = parts[0] if len(parts) == 1 else np.concatenate(parts)
                mono = _to_mono(sample)
                mic = _resample(mono, self._in_sr, TARGET_SR)
                if self._mic_gain != 1.0:
                    mic = np.clip(mic * self._mic_gain, -1.0, 1.0)
                self._poll_hw_doa(t0)
                # Barge-in fast path: shorten the onset gate while the robot talks.
                # The energy source is safe for this — the AEC'd mic doesn't carry
                # the robot's own voice, so sustained energy here is a human.
                self.endpointer.min_speech_ms = (
                    self.cfg.vad_barge_min_speech_ms
                    if self.bus.robot_speaking.is_set()
                    else self.cfg.vad_min_speech_ms
                )
                # The XVF3800 already did AEC/NS/AGC in hardware, so the mic is
                # echo-free: stream the uplink continuously — same as the official app.
                self.endpointer.process(mic)
                self._maybe_barge()
                if self._agc is not None:
                    self._agc.update(mic, self.endpointer.in_speech)
                    mic = self._agc.apply(mic)
                self._accumulate(mic)
            time.sleep(max(0.0, poll_dt - (time.monotonic() - t0)))

    def _poll_hw_doa(self, now: float) -> None:
        """Refresh the cached DOA angle from the XVF3800, at most every 50 ms.

        Single owner of get_DoA() USB reads — the motion controller consumes the
        cached bus.doa_angle, so no two threads ever hit the USB device at once.
        The firmware's speech flag only gates the *angle* update (an angle without
        sound activity is stale); it is NOT a voice source — it runs pre-AEC and
        stays high while the robot's own speaker plays.
        """
        if now - self._hw_last < 0.05:
            return
        self._hw_last = now
        try:
            res = self.mini.media.get_DoA()
        except Exception:
            res = None
        if res is not None and bool(res[1]):
            self.bus.doa_angle = float(res[0])

    def _accumulate(self, clean: np.ndarray) -> None:
        self._chunk_buf = np.concatenate([self._chunk_buf, clean.astype(np.float32)])
        while len(self._chunk_buf) >= self._chunk_samples:
            chunk = self._chunk_buf[: self._chunk_samples]
            self._chunk_buf = self._chunk_buf[self._chunk_samples:]
            try:
                self._on_audio_chunk(chunk)
            except Exception as e:
                log.debug("on_audio_chunk error: %s", e)

    def _maybe_barge(self) -> None:
        """Cut the robot off when a human talks over it.

        Level-triggered each capture poll. The voice source is the energy gate
        over the AEC'd mic — the robot's own voice is hardware-cancelled out of
        that stream, so sustained energy during playback is a human. Only
        bus.request_interrupt is signalled; the device-buffer flush runs on the
        playback thread (see _playback_loop), never from this thread.
        """
        if not (self.bus.robot_speaking.is_set() and self.endpointer.in_speech):
            return
        if self.bus.interrupt_event.is_set():
            return  # already tearing this reply down
        log.info("barge-in detected")
        self.bus.request_interrupt()
        # Tell the SERVER as fast as we tell the speaker: ship the partial capture
        # buffer right now instead of waiting for the 1 s chunk boundary. The sender
        # stamps force_listen on it (interrupt_event is set), so the model stops
        # generating up to ~1 s sooner. Anything shorter than ~120 ms isn't worth a
        # message of its own — the next full chunk is imminent.
        if getattr(self.cfg, "omni_barge_flush", True):
            self._flush_uplink(min_ms=120)

    def _flush_uplink(self, min_ms: int = 120) -> None:
        """Send the partially-accumulated uplink chunk immediately (barge-in path)."""
        n = len(self._chunk_buf)
        if n < int(TARGET_SR * min_ms / 1000):
            return
        chunk, self._chunk_buf = self._chunk_buf, np.zeros(0, dtype=np.float32)
        try:
            self._on_audio_chunk(chunk)
        except Exception as e:
            log.debug("on_audio_chunk error (barge flush): %s", e)

    def _on_speech_start(self) -> None:
        # NOTE: barge-in is NOT decided here — this edge callback also fires for
        # noise that fools the energy VAD. See _maybe_barge for the confirmed cut.
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

    def _flush_device_playback(self) -> None:
        # Audio already handed to push_audio_sample sits in the device's GStreamer
        # playback pipeline (~cushion + pipeline latency, up to >1 s) and would keep
        # playing after we stop pushing. clear_player() flushes the appsrc so the
        # robot goes quiet at once. MediaManager doesn't forward it, so reach for
        # media.audio; best-effort — absent on older SDKs.
        media = self.mini.media
        fn = getattr(media, "clear_player", None) or getattr(
            getattr(media, "audio", None), "clear_player", None
        )
        if fn is None:
            log.warning("barge-in: clear_player unavailable on this media backend — "
                        "device-buffered audio (~0.3-1 s) will play out")
            return
        try:
            fn()
            log.info("barge-in: flushed device playback buffer")
        except Exception as e:
            log.warning("barge-in: clear_player failed: %s", e)

    # -- playback -----------------------------------------------------------
    def _playback_loop(self) -> None:
        idle_since = time.monotonic()
        interrupt_flushed = False
        while not self.bus.stop_event.is_set():
            # Barge-in: this thread owns every push into the playback appsrc, so
            # it is the only place clear_player() can run without racing a
            # concurrent push_buffer (which can wedge the shared pipeline).
            if self.bus.interrupt_event.is_set():
                if not interrupt_flushed and self.bus.robot_speaking.is_set():
                    self._flush_device_playback()
                    interrupt_flushed = True
            else:
                interrupt_flushed = False
            try:
                item = self.bus.tts_audio.get(timeout=0.1)
            except Exception:
                if self.bus.robot_speaking.is_set() and time.monotonic() - idle_since > 0.25:
                    self.bus.robot_speaking.clear()
                    self.bus.speech_env.clear()
                    self.bus.speech_level = 0.0
                    self.bus.emit("system", {"text": "spoke"})
                    self._turn_start = None  # reset pacing for the next spoken turn
                # Safety net for a STALE interrupt only. The interrupt must stay up for
                # the user's whole barge utterance (it gates force_listen + downlink
                # discard); clearing it just because playback drained — as this net
                # once did — re-opened the door for the old reply ~0.35 s after the
                # cut ("pauses, then keeps talking"). Normal clearing is the VAD's
                # speech-end hook; this only catches a wedged VAD via a 10 s cap.
                if self.bus.interrupt_event.is_set() and not self.bus.robot_speaking.is_set():
                    if not self.bus.user_speaking.is_set():
                        self.bus.clear_interrupt()
                    elif time.monotonic() - self.bus.interrupt_since > 10.0:
                        log.warning("barge-in interrupt active >10 s (VAD wedged?) — clearing")
                        self.bus.clear_interrupt()
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
        if self._turn_start is None:
            self._turn_start = time.monotonic()
            self._turn_played = 0.0
        n = self._play_sub_n
        for i in range(0, len(chunk), n):
            if self.bus.interrupt_event.is_set():
                self.bus.speech_env.clear()
                self.bus.speech_level = 0.0
                break
            sub = chunk[i:i + n]
            # Voice envelope for the speech-synced wobble, stamped with the time this
            # sub-chunk will actually PLAY (we push up to a cushion ahead of real time,
            # so publishing at push time would move the head before the sound). Loudness
            # mapping follows the official head-wobbler DSP: dBFS normalized over
            # [-46, -18] dB with gamma 0.9 — perceptually much closer than linear RMS.
            due = self._turn_start + self._turn_played
            if len(sub):
                rms = float(np.sqrt(np.mean(sub.astype(np.float64) ** 2)))
                dbfs = 20.0 * math.log10(rms + 1e-9)
                lvl = min(1.0, max(0.0, (dbfs + 46.0) / 28.0)) ** 0.9
                self.bus.speech_env.append((due, lvl))
            try:
                self.mini.media.push_audio_sample(sub.reshape(-1, 1))
            except Exception as e:
                log.debug("push_audio_sample error: %s", e)
            # Pace to stay ~cushion ahead of real time: push freely to build the cushion,
            # only sleep once we're further ahead than the cushion (bounds latency). If the
            # server falls behind, `ahead` goes negative and we never sleep (catch up).
            self._turn_played += len(sub) / self._out_sr
            ahead = self._turn_played - (time.monotonic() - self._turn_start)
            if ahead > self._play_cushion:
                time.sleep(ahead - self._play_cushion)
