"""Microphone front-end: WebRTC noise-suppression + auto-gain (optional).

Wraps `webrtc-noise-gain` (module ``webrtc_noise_gain``). It does **noise suppression
and automatic gain control only — NOT echo cancellation** (it never sees the played
audio, so it cannot remove the robot's own voice; that is handled separately by muting
the uplink while the robot speaks — see ``audio/io.py``).

Why use it: the robot mic is quiet and noisy, and a plain fixed gain (``OMNI_MIC_GAIN``)
amplifies background noise and hard-clips loud speech — both make the omni model mishear.
AGC pulls speech to a target level without clipping and NS removes steady background noise,
so the model gets cleaner input.

The WebRTC processor works on fixed **10 ms / 160-sample** frames at 16 kHz, so we buffer
the variable-length capture batches and emit whatever whole frames are ready. If the
package isn't installed we fall back to a plain fixed gain (pass-through at gain 1.0), so a
minimal on-robot install still runs.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("live_chat.denoise")

_FRAME = 160  # 10 ms @ 16 kHz (what webrtc-noise-gain requires)


class MicPreprocessor:
    """NS + AGC mic cleaner with a graceful fixed-gain fallback."""

    def __init__(self, agc_dbfs: int = 15, ns_level: int = 2, fallback_gain: float = 1.0) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._proc = None
        self._fallback_gain = float(fallback_gain or 1.0)
        agc_dbfs = max(0, min(31, int(agc_dbfs)))
        ns_level = max(0, min(4, int(ns_level)))
        if agc_dbfs or ns_level:
            try:
                from webrtc_noise_gain import AudioProcessor

                self._proc = AudioProcessor(agc_dbfs, ns_level)
                log.info("mic: WebRTC noise-suppression + auto-gain enabled (agc=%d dBFS, ns=%d)",
                         agc_dbfs, ns_level)
            except Exception as e:
                log.warning("mic: webrtc-noise-gain unavailable (%s); using plain gain x%.1f "
                            "(pip install webrtc-noise-gain to enable NS+AGC)",
                            e, self._fallback_gain)
                self._proc = None

    @property
    def active(self) -> bool:
        """True when WebRTC NS+AGC is doing the work (else fixed-gain fallback)."""
        return self._proc is not None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Clean a mono float32 16 kHz frame. Returns only whole 10 ms frames that are
        ready (buffers the remainder), so the output length can differ from the input."""
        if self._proc is None:
            if self._fallback_gain != 1.0:
                return np.clip(frame * self._fallback_gain, -1.0, 1.0)
            return frame.astype(np.float32)

        self._buf = np.concatenate([self._buf, frame.astype(np.float32)])
        out = []
        while len(self._buf) >= _FRAME:
            f = self._buf[:_FRAME]
            self._buf = self._buf[_FRAME:]
            pcm16 = (np.clip(f, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            try:
                res = self._proc.Process10ms(pcm16)
                cleaned = np.frombuffer(res.audio, dtype="<i2").astype(np.float32) / 32767.0
            except Exception as e:  # pragma: no cover - defensive: never kill capture
                log.debug("mic NS/AGC error: %s", e)
                cleaned = f
            out.append(cleaned)
        if out:
            return np.concatenate(out)
        return np.zeros(0, dtype=np.float32)
