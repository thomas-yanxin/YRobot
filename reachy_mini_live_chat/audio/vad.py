"""Voice-activity detection + turn endpointing, driven by the ReSpeaker XVF3800.

The mic board's firmware runs its own post-AEC voice detection (exposed together
with the DOA angle via a USB control read). That flag is the *only* voice source
here: unlike an energy gate it does not confuse residual echo or head-servo noise
with a human, which is exactly what barge-in needs. The remote omni model still
does the real turn-taking — this VAD only drives DOA head-turns, the listen mood,
barge-in, and the e2e latency clock.

:class:`HwVoiceFlag` adapts the cached firmware flag to the frame-based
:class:`Endpointer`, which turns it into debounced speech-start / utterance-end
callbacks via a min-speech / silence-window state machine.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

log = logging.getLogger("live_chat.vad")

FRAME = 512  # 32 ms @ 16 kHz — the endpointer's debounce granularity


class HwVoiceFlag:
    """Adapt a cached XVF3800 voice flag to the Endpointer's speech_prob API.

    ``get_flag`` returns the most recent firmware reading (refreshed by the
    audio capture loop's ~20 Hz USB poll); audio frames are ignored — the
    firmware already looked at the signal, post-AEC.
    """

    def __init__(self, get_flag: Callable[[], bool]) -> None:
        self._get_flag = get_flag

    def speech_prob(self, frame: np.ndarray) -> float:
        return 1.0 if self._get_flag() else 0.0


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
