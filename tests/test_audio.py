"""VoiceGate timing, AGC behaviour, resampler ratio, preroll adaptation."""

import numpy as np

from yrobot.audio import PrerollPolicy, StreamResampler, UplinkGain, VoiceGate
from yrobot.config import Config

CFG = Config()


def test_preroll_grows_on_underruns_and_decays_when_clean():
    p = PrerollPolicy(CFG)  # min 0.25, max 0.8, starts at 0.4
    assert p.on_utterance_start(None) < 0.4  # first-ever start decays
    for _ in range(10):
        p.on_utterance_start(0.3)  # supply resumed after a short dry gap
    assert p.value == CFG.preroll_max_s
    for _ in range(50):
        p.on_utterance_start(10.0)  # clean starts after long silence
    assert p.value == CFG.preroll_min_s


def feed(gate, rms, ms, robot_speaking=False, step=20.0):
    active = False
    for _ in range(int(ms / step)):
        active = gate.process(rms, step, robot_speaking)
    return active


def test_gate_onset_needs_sustained_speech():
    g = VoiceGate(CFG)
    assert not feed(g, 0.002, 1000)  # settle floor
    assert not feed(g, 0.1, 60)  # 60 ms blip < onset_ms
    assert not feed(g, 0.002, 200)
    assert feed(g, 0.1, 200)  # sustained speech


def test_gate_release_tolerates_dips():
    g = VoiceGate(CFG)
    feed(g, 0.002, 1000)
    assert feed(g, 0.1, 200)
    assert feed(g, 0.001, 200)  # 200 ms dip < release_ms → still active
    assert not feed(g, 0.001, 400)  # sustained silence releases


def test_gate_is_stricter_while_robot_speaks():
    g = VoiceGate(CFG)
    feed(g, 0.002, 1000)
    # just above the normal threshold but below the barge threshold
    lvl = g.threshold * 1.2
    assert not feed(g, lvl, 400, robot_speaking=True)
    assert feed(g, g.threshold * CFG.gate_barge_mult * 1.2, 400, robot_speaking=True)


def test_agc_boosts_quiet_speech_only():
    a = UplinkGain(CFG)
    for _ in range(200):
        a.observe(0.02, voice=True, robot_speaking=False)
    assert a.gain > 3.0
    out = a.apply(np.full(100, 0.02, np.float32))
    assert out.max() > 0.06
    b = UplinkGain(CFG)
    for _ in range(200):
        b.observe(0.02, voice=True, robot_speaking=True)  # robot talking → no adapt
    assert b.gain == 1.0


def test_resampler_24k_to_16k():
    r = StreamResampler(24000, 16000)
    total_out = sum(len(r.process(np.zeros(2400, np.float32))) for _ in range(10))
    assert 15200 < total_out <= 16000  # 1 s in → 1 s out minus filter delay


def test_resampler_passthrough():
    r = StreamResampler(16000, 16000)
    x = np.ones(160, np.float32)
    assert r.process(x) is x
