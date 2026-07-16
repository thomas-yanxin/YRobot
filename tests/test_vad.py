"""Tests for the endpointer state machine over the XVF3800 voice-flag source."""
import numpy as np

from reachy_mini_live_chat.audio import vad

FRAME = vad.FRAME  # 512 samples @ 16 kHz == 32 ms


class _ScriptedVad:
    """speech_prob returns the scripted values in order (1.0 = voiced, 0.0 = not)."""

    def __init__(self, probs):
        self._probs = list(probs)

    def speech_prob(self, frame):
        return self._probs.pop(0) if self._probs else 0.0


def _endpointer(probs, **kw):
    events = {"started": 0, "utterances": []}
    ep = vad.Endpointer(
        _ScriptedVad(probs),
        on_speech_start=lambda: events.__setitem__("started", events["started"] + 1),
        on_utterance=lambda pcm: events["utterances"].append(pcm),
        **kw,
    )
    return ep, events


def test_energy_vad_prob_range_and_ordering():
    v = vad.EnergyVad()
    silence = np.zeros(FRAME, dtype=np.float32)
    loud = (np.random.default_rng(0).standard_normal(FRAME) * 0.3).astype(np.float32)
    assert 0.0 <= v.speech_prob(silence) <= 1.0
    assert v.speech_prob(loud) >= v.speech_prob(silence)


def test_energy_vad_adapts_floor_back_down():
    # Sustained sound raises the floor slowly; silence drops it fast — the gate must
    # recover so it doesn't report speech forever after a loud stretch.
    v = vad.EnergyVad()
    loud = (np.random.default_rng(1).standard_normal(FRAME) * 0.3).astype(np.float32)
    for _ in range(50):
        v.speech_prob(loud)
    silence = np.zeros(FRAME, dtype=np.float32)
    for _ in range(50):
        v.speech_prob(silence)
    assert v.speech_prob(silence) < 0.5


def test_endpointer_start_and_end():
    # 8 voiced frames (~256 ms) then silence past the 320 ms window → one utterance.
    probs = [1] * 8 + [0] * 12
    ep, events = _endpointer(probs, min_speech_ms=200, silence_ms=320)
    frame = np.zeros(FRAME, dtype=np.float32)
    for _ in probs:
        ep._step(frame)
    assert events["started"] == 1
    assert len(events["utterances"]) == 1
    assert not ep.in_speech


def test_endpointer_onset_survives_brief_dip():
    # 3 voiced frames (~96 ms), 1 dip, then more voiced: the dip must only decay the
    # onset run, not reset it — with min_speech_ms=200 the start should fire on the
    # 8th voiced frame (~224 ms of accumulated speech), not restart from zero.
    probs = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    ep, events = _endpointer(probs, min_speech_ms=200, silence_ms=320)
    frame = np.zeros(FRAME, dtype=np.float32)
    for _ in probs:
        ep._step(frame)
    assert events["started"] == 1, "a single sub-threshold frame must not restart onset detection"


def test_endpointer_ignores_short_blips():
    # 2 voiced frames (~64 ms) << min_speech_ms → never starts.
    probs = [1, 1] + [0] * 10
    ep, events = _endpointer(probs, min_speech_ms=200, silence_ms=320)
    frame = np.zeros(FRAME, dtype=np.float32)
    for _ in probs:
        ep._step(frame)
    assert events["started"] == 0
    assert events["utterances"] == []
