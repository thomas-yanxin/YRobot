"""Voice-activity detection + turn endpointing on the AEC'd mic stream.

The voice source is an adaptive RMS gate over the *echo-cancelled* mic signal —
the one signal in the system where the robot's own voice is absent (the XVF3800
removes it in hardware before the samples reach us). That property is what makes
barge-in possible: while the robot speaks, energy on this stream is a human.

The XVF3800 firmware also reports its own "speech detected" flag (with the DOA
angle), but that detector runs on the RAW mic array — pre-AEC — so during
playback it hears the robot's own speaker and servos and stays high (verified on
hardware: barge-ins fired seconds into every reply at mic rms ~0.01). It is
therefore only used for the DOA head-turn, never for voice/barge decisions.

The remote omni model does the real turn-taking — this VAD only drives DOA,
the listen mood, barge-in, and the e2e latency clock.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

log = logging.getLogger("live_chat.vad")

FRAME = 512  # 32 ms @ 16 kHz — the endpointer's debounce granularity


class EnergyVad:
    """Adaptive RMS gate with a fast-drop / slow-rise noise-floor tracker."""

    def __init__(self) -> None:
        # Start near a typical ambient level. The XVF3800's hardware AGC raises the mic's
        # noise floor (rms ~0.05 even when quiet), so a tiny init would look like permanent
        # speech until it adapts.
        self._floor = 1e-2

    def speech_prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-9)
        # Track the noise floor both ways: drop quickly toward quiet frames, rise slowly
        # during sustained sound. This adapts to the real ambient level (including one
        # raised by hardware AGC) instead of sticking at init and reporting speech forever.
        if rms < self._floor:
            self._floor = 0.9 * self._floor + 0.1 * rms
        else:
            self._floor = 0.995 * self._floor + 0.005 * rms
        snr = rms / (self._floor + 1e-9)
        # map SNR -> pseudo-probability
        return float(np.clip((snr - 2.5) / 6.0, 0.0, 1.0))


class UplinkAgc:
    """Software auto-gain for the model uplink: hold *speech* near a target RMS.

    Tracks an EMA of the RMS of chunks the VAD marks as speech and derives one
    slowly-moving gain ``target / speech_rms``. The gain only ever boosts
    (>= 1.0 — the hardware AGC already prevents clipping-loud speech) and is
    capped so a whisper across the room can't become full-scale noise.

    Why: the omni model does its own turn-taking from the audio; speech that
    arrives too quiet is treated as background and the model just keeps
    listening — the classic "I talk but it never answers". Normalizing the
    uplink makes that failure mode impossible regardless of mic distance.
    """

    def __init__(self, target_rms: float = 0.12, max_gain: float = 8.0) -> None:
        self.target = float(target_rms)
        self.max_gain = max(1.0, float(max_gain))
        self._speech_rms: Optional[float] = None
        self._gain = 1.0

    @property
    def gain(self) -> float:
        return self._gain

    def update(self, chunk: np.ndarray, voiced: bool) -> float:
        """Feed one mic chunk + the VAD's current speech verdict; returns the gain."""
        if voiced and len(chunk):
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            if rms > 1e-4:  # ignore silence mislabelled as speech
                if self._speech_rms is None:
                    self._speech_rms = rms
                else:
                    # fast-ish EMA: adapts within ~2 s of speech at 10 Hz updates
                    self._speech_rms = 0.9 * self._speech_rms + 0.1 * rms
                raw = self.target / max(self._speech_rms, 1e-4)
                target_gain = float(np.clip(raw, 1.0, self.max_gain))
                # smooth the applied gain so the uplink level never steps audibly
                self._gain += 0.2 * (target_gain - self._gain)
        return self._gain

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        if self._gain <= 1.0 + 1e-3:
            return chunk
        return np.clip(chunk * self._gain, -1.0, 1.0).astype(np.float32)


class Endpointer:
    """Stream 16 kHz mono chunks in; get utterance callbacks out."""

    def __init__(
        self,
        vad,
        *,
        threshold: float = 0.5,
        silence_ms: int = 320,
        min_speech_ms: int = 200,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_utterance: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self._vad = vad
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.on_speech_start = on_speech_start
        self.on_utterance = on_utterance

        self._buf = np.zeros(0, dtype=np.float32)
        self._speech: List[np.ndarray] = []
        self._in_speech = False
        self._silence_ms_run = 0.0
        self._speech_ms_run = 0.0
        self._frame_ms = FRAME / 16000 * 1000.0

    @property
    def in_speech(self) -> bool:
        """True while the detector currently believes a human is talking."""
        return self._in_speech

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._speech.clear()
        self._in_speech = False
        self._silence_ms_run = 0.0
        self._speech_ms_run = 0.0
        if hasattr(self._vad, "reset"):
            self._vad.reset()

    def process(self, chunk: np.ndarray) -> None:
        """Feed a mono float32 chunk (any length). Emits callbacks on transitions."""
        self._buf = np.concatenate([self._buf, chunk.astype(np.float32)])
        while len(self._buf) >= FRAME:
            frame, self._buf = self._buf[:FRAME], self._buf[FRAME:]
            self._step(frame)

    def _step(self, frame: np.ndarray) -> None:
        prob = self._vad.speech_prob(frame)
        voiced = prob >= self.threshold
        if voiced:
            self._silence_ms_run = 0.0
            self._speech_ms_run += self._frame_ms
            if not self._in_speech and self._speech_ms_run >= self.min_speech_ms:
                self._in_speech = True
                if self.on_speech_start:
                    self.on_speech_start()
            if self._in_speech:
                self._speech.append(frame)
        else:
            if self._in_speech:
                self._speech.append(frame)  # keep trailing context
                self._silence_ms_run += self._frame_ms
                if self._silence_ms_run >= self.silence_ms:
                    self._emit()
            else:
                # Decay (not reset) the onset run: the firmware flag can dip for
                # a beat mid-word, and a hard reset made the min_speech_ms gate
                # take far longer than min_speech_ms — slow barge-in detection.
                self._speech_ms_run = max(0.0, self._speech_ms_run - 2.0 * self._frame_ms)

    def _emit(self) -> None:
        if self._speech and self.on_utterance:
            pcm = np.concatenate(self._speech)
            self.on_utterance(pcm)
        self._speech.clear()
        self._in_speech = False
        self._silence_ms_run = 0.0
        self._speech_ms_run = 0.0
