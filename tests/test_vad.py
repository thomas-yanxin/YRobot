"""Tests for the VAD: numpy energy fallback + the vendored Silero onnx model."""
import numpy as np
import pytest

from reachy_mini_live_chat.audio import vad

FRAME = vad.FRAME  # 512 samples @ 16 kHz


def test_energy_vad_prob_range():
    v = vad._EnergyVad()
    silence = np.zeros(FRAME, dtype=np.float32)
    loud = (np.random.default_rng(0).standard_normal(FRAME) * 0.3).astype(np.float32)
    assert 0.0 <= v.speech_prob(silence) <= 1.0
    # a burst of energy should read higher than silence
    assert v.speech_prob(loud) >= v.speech_prob(silence)


def test_build_vad_stub_is_energy():
    assert isinstance(vad.build_vad(stub=True), vad._EnergyVad)


def test_build_vad_defaults_to_energy():
    # cheap energy VAD is the default (onnx is opt-in) so the CM4 stays responsive
    assert isinstance(vad.build_vad(stub=False), vad._EnergyVad)
    assert isinstance(vad.build_vad(stub=False, backend="energy"), vad._EnergyVad)


def test_onnx_vad_loads_and_scores():
    ort = pytest.importorskip("onnxruntime")  # noqa: F841
    v = vad._OnnxVad()
    # silence → low speech probability
    p_sil = v.speech_prob(np.zeros(FRAME, dtype=np.float32))
    assert 0.0 <= p_sil <= 1.0
    assert p_sil < 0.5
    # a 220 Hz tone (voiced-ish) should not error and stays in range
    t = np.arange(FRAME, dtype=np.float32) / 16000.0
    tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = v.speech_prob(tone)
    assert 0.0 <= p <= 1.0
    v.reset()  # state resets cleanly
    assert v.speech_prob(np.zeros(FRAME, dtype=np.float32)) < 0.5


def test_build_vad_onnx_when_requested():
    pytest.importorskip("onnxruntime")
    assert isinstance(vad.build_vad(stub=False, backend="onnx"), vad._OnnxVad)
    assert isinstance(vad.build_vad(stub=False, backend="auto"), vad._OnnxVad)


class _ScriptedVad:
    """speech_prob returns the scripted values in order (1.0 = voiced, 0.0 = not)."""

    def __init__(self, probs):
        self._probs = list(probs)

    def speech_prob(self, frame):
        return self._probs.pop(0) if self._probs else 0.0


def test_endpointer_onset_survives_brief_dip():
    # 3 voiced frames (~96 ms), 1 dip, then more voiced: the dip must only decay the
    # onset run, not reset it — with min_speech_ms=200 the start should fire on the
    # 8th voiced frame (~224 ms of accumulated speech), not restart from zero.
    started = []
    probs = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    ep = vad.Endpointer(
        _ScriptedVad(probs),
        threshold=0.5,
        silence_ms=320,
        min_speech_ms=200,
        on_speech_start=lambda: started.append(True),
    )
    frame = np.zeros(FRAME, dtype=np.float32)
    for _ in probs:
        ep._step(frame)
    assert started, "a single sub-threshold frame must not restart onset detection"
