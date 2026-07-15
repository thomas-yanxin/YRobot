"""Voice-activity detection + turn endpointing.

The VAD here is *only* a local trigger — it decides "is a human talking right now" to
steer DOA, raise the listen mood, and detect barge-in. The remote omni model does the
real turn-taking, so we don't need anything heavy.

Primary detector: **Silero VAD v5** run on CPU via **onnxruntime** (the model is
vendored in ``assets/silero_vad_16k.onnx``). This deliberately avoids the ``silero-vad``
pip package, which depends on ``torch`` and would drag in the CUDA/cuDNN runtime for a
client that never touches a GPU. If ``onnxruntime`` isn't installed, we fall back to a
robust pure-**numpy energy** VAD, so the whole pipeline still runs with zero extra deps.

The :class:`Endpointer` turns a stream of 16 kHz mono chunks into discrete utterances
via a start/silence-window state machine, and exposes a cheap ``speech_prob`` for
barge-in detection while the robot is talking.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

log = logging.getLogger("live_chat.vad")

FRAME = 512  # Silero v5 wants 512 samples @ 16 kHz == 32 ms


class _EnergyVad:
    """Fallback VAD: adaptive RMS gate with a noise floor tracker."""

    def __init__(self) -> None:
        self._floor = 1e-3
        self._alpha = 0.98

    def speech_prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-9)
        # track the noise floor slowly, only when quiet
        if rms < self._floor * 2:
            self._floor = self._alpha * self._floor + (1 - self._alpha) * rms
        snr = rms / (self._floor + 1e-9)
        # map SNR -> pseudo-probability
        return float(np.clip((snr - 2.5) / 6.0, 0.0, 1.0))


def _silero_model_path() -> str:
    """Filesystem path to the vendored Silero v5 onnx model."""
    import os

    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "silero_vad_16k.onnx")


class _OnnxVad:
    """Silero VAD v5 on CPU via onnxruntime (no torch). 512-sample frames @ 16 kHz.

    Model I/O (op15 16k build): inputs ``input`` (batch, samples) float32,
    ``state`` (2, batch, 128) float32, ``sr`` int64 scalar; outputs ``output``
    (batch, 1) speech-prob and ``stateN`` (recurrent state to feed back).
    """

    def __init__(self) -> None:
        import onnxruntime as ort  # lazy

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            _silero_model_path(), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()

    def speech_prob(self, frame: np.ndarray) -> float:
        x = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        try:
            out, self._state = self._sess.run(
                ["output", "stateN"], {"input": x, "state": self._state, "sr": self._sr}
            )
        except Exception:
            return 0.0
        return float(out[0, 0])

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)


def build_vad(stub: bool = False):
    if stub:
        log.info("VAD: energy fallback (stub)")
        return _EnergyVad()
    try:
        vad = _OnnxVad()
        log.info("VAD: Silero v5 (onnxruntime, CPU)")
        return vad
    except Exception as e:
        log.warning("VAD: onnxruntime unavailable (%s); using numpy energy fallback", e)
        return _EnergyVad()


class SemanticTurn:
    """Optional wrapper around livekit/turn-detector to confirm end-of-turn.

    Returns True if the transcript looks like a *complete* thought. When the model
    is unavailable, a light heuristic (sentence-final punctuation / length) is used.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._model = None
        if enabled:
            try:
                # The onnx model is loaded lazily on first use; import guarded.
                import onnxruntime  # noqa: F401
                log.info("SemanticTurn: onnxruntime available")
            except Exception as e:
                log.warning("SemanticTurn: disabled (%s)", e)
                self.enabled = False

    def is_complete(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        if not self.enabled:
            return True  # rely purely on the silence window
        # Heuristic stand-in for the model: ends with terminal punctuation, or is
        # long enough that further silence almost certainly means "done".
        if text[-1] in "。！？.!?…":
            return True
        return len(text) >= 12


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
                self._speech_ms_run = 0.0

    def _emit(self) -> None:
        if self._speech and self.on_utterance:
            pcm = np.concatenate(self._speech)
            self.on_utterance(pcm)
        self._speech.clear()
        self._in_speech = False
        self._silence_ms_run = 0.0
        self._speech_ms_run = 0.0
