"""Uplink AGC: quiet speech is boosted toward the target; loud speech untouched."""
import numpy as np

from reachy_mini_live_chat.audio.io import AudioEngine
from reachy_mini_live_chat.audio.vad import UplinkAgc
from reachy_mini_live_chat.bus import Bus
from reachy_mini_live_chat.config import Config


def _speech(rms: float, n: int = 1600) -> np.ndarray:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n).astype(np.float32)
    return x * (rms / (np.sqrt(np.mean(x ** 2)) + 1e-9))


def test_quiet_speech_is_boosted_toward_target():
    agc = UplinkAgc(target_rms=0.12, max_gain=8.0)
    chunk = _speech(0.02)  # far below the level the model treats as foreground
    for _ in range(200):   # past the 8 s warmup, then let the smoothed gain converge
        agc.update(chunk, voiced=True)
    assert agc.gain > 3.0
    out = agc.apply(chunk)
    out_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert 0.08 <= out_rms <= 0.20  # lands near the target


def test_loud_speech_never_attenuated():
    agc = UplinkAgc(target_rms=0.12, max_gain=8.0)
    chunk = _speech(0.3)
    for _ in range(200):
        agc.update(chunk, voiced=True)
    assert agc.gain == 1.0
    assert np.array_equal(agc.apply(chunk), chunk)  # identity fast path


def test_gain_capped_at_max():
    agc = UplinkAgc(target_rms=0.12, max_gain=4.0)
    chunk = _speech(0.001)
    for _ in range(300):
        agc.update(chunk, voiced=True)
    assert agc.gain <= 4.0 + 1e-6


def test_unvoiced_frames_do_not_move_the_gain():
    agc = UplinkAgc()
    noise = _speech(0.001)
    for _ in range(100):
        agc.update(noise, voiced=False)
    assert agc.gain == 1.0


def test_output_clipped_to_unit_range():
    agc = UplinkAgc(target_rms=0.5, max_gain=8.0)
    chunk = _speech(0.2)
    for _ in range(200):
        agc.update(chunk, voiced=True)
    out = agc.apply(chunk)
    assert float(np.max(np.abs(out))) <= 1.0


class _Media:
    def get_DoA(self):
        return None


class _Mini:
    def __init__(self):
        self.media = _Media()


def test_engine_agc_disabled_by_config():
    cfg = Config()
    cfg.omni_mic_agc = False
    eng = AudioEngine(_Mini(), cfg, Bus(), on_audio_chunk=lambda c: None)
    assert eng._agc is None


def test_engine_agc_disabled_by_default():
    # The XVF3800 already does AGC in hardware; software AGC is opt-in only.
    eng = AudioEngine(_Mini(), Config(), Bus(), on_audio_chunk=lambda c: None)
    assert eng._agc is None


def test_engine_agc_opt_in():
    cfg = Config()
    cfg.omni_mic_agc = True
    eng = AudioEngine(_Mini(), cfg, Bus(), on_audio_chunk=lambda c: None)
    assert eng._agc is not None


def test_agc_ignores_warmup_period():
    """Cold-start ambient mislabelled as speech must not move the gain."""
    agc = UplinkAgc(target_rms=0.12, max_gain=8.0)
    ambient = _speech(0.03, n=16000)  # 1 s per update
    for _ in range(7):                # 7 s: inside the warmup window
        agc.update(ambient, voiced=True)
    assert agc.gain == 1.0
    for _ in range(60):               # past warmup: adaptation resumes
        agc.update(ambient, voiced=True)
    assert agc.gain > 1.5
