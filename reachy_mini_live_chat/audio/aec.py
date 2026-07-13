"""Echo control for full-duplex barge-in.

The robot's own TTS leaks into its microphone and would re-trigger the VAD. Two
strategies, selectable via config:

* ``ENABLE_AEC=1`` — true acoustic echo cancellation via WebRTC APM. We feed it the
  mic capture plus the TTS render (reference) signal; it subtracts the echo, so the
  VAD only sees the human and real barge-in works.
* ``ENABLE_AEC=0`` (default) — **duck & gate**: while the robot speaks we attenuate the
  mic and raise the effective VAD threshold, only allowing an interrupt when the human
  is clearly louder than the residual echo (``barge_in_energy``). Ships without extra deps.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

log = logging.getLogger("live_chat.aec")


class EchoController:
    def __init__(self, enable_aec: bool = False, barge_in_energy: float = 0.02) -> None:
        self.enable_aec = enable_aec
        self.barge_in_energy = barge_in_energy
        self._apm = None
        self._ref: deque[np.ndarray] = deque(maxlen=50)  # recent TTS reference frames
        if enable_aec:
            self._init_apm()

    def _init_apm(self) -> None:
        try:
            from webrtc_audio_processing import AudioProcessingModule  # type: ignore

            self._apm = AudioProcessingModule(aec_type=2, enable_ns=True, agc_type=0)
            self._apm.set_stream_format(16000, 1)
            self._apm.set_reverse_stream_format(16000, 1)
            log.info("AEC: WebRTC APM enabled")
        except Exception as e:
            log.warning("AEC requested but unavailable (%s); using duck & gate", e)
            self._apm = None
            self.enable_aec = False

    def note_playback(self, frame: np.ndarray) -> None:
        """Register a TTS output frame as the echo reference."""
        self._ref.append(frame.astype(np.float32).copy())
        if self._apm is not None:
            try:
                self._apm.process_reverse_stream(_to_i16(frame))
            except Exception:
                pass

    def process_capture(self, frame: np.ndarray, robot_speaking: bool) -> np.ndarray:
        """Clean a mic frame for the VAD."""
        if self._apm is not None:
            try:
                return _from_i16(self._apm.process_stream(_to_i16(frame)))
            except Exception:
                return frame
        if robot_speaking and not self.enable_aec:
            # duck the mic so residual echo doesn't cross the VAD threshold
            return frame * 0.35
        return frame

    def is_barge_in(self, frame: np.ndarray, robot_speaking: bool) -> bool:
        """Is the human clearly talking over the robot?"""
        if not robot_speaking:
            return False
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-9)
        return rms >= self.barge_in_energy


def _to_i16(frame: np.ndarray) -> bytes:
    return (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()


def _from_i16(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
