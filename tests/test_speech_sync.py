"""Speech-synced motion: the talking wobble follows the played voice envelope."""
import numpy as np

from reachy_mini_live_chat.audio.io import AudioEngine
from reachy_mini_live_chat.bus import Bus
from reachy_mini_live_chat.config import Config
from reachy_mini_live_chat.motion.controller import MotionController


class _Media:
    def __init__(self):
        self.pushed = []

    def push_audio_sample(self, x):
        self.pushed.append(x)

    def get_DoA(self):
        return None


class _Mini:
    def __init__(self):
        self.media = _Media()


def test_playback_publishes_timestamped_envelope():
    cfg = Config()
    bus = Bus()
    eng = AudioEngine(_Mini(), cfg, bus, on_audio_chunk=lambda c: None)
    eng._play_cushion = 10.0  # no pacing sleeps in the test
    loud = (np.random.default_rng(0).standard_normal(4800) * 0.2).astype(np.float32)
    eng._play_chunk(loud)
    assert bus.robot_speaking.is_set()
    assert len(bus.speech_env) > 0
    dues = [d for d, _ in bus.speech_env]
    lvls = [lv for _, lv in bus.speech_env]
    assert dues == sorted(dues)          # due times are monotonic (playback order)
    assert max(lvls) > 0.5               # -14 dBFS speech is a strong wobble


def test_quiet_audio_gives_low_envelope():
    cfg = Config()
    bus = Bus()
    eng = AudioEngine(_Mini(), cfg, bus, on_audio_chunk=lambda c: None)
    eng._play_cushion = 10.0
    quiet = (np.random.default_rng(0).standard_normal(4800) * 0.002).astype(np.float32)
    eng._play_chunk(quiet)
    assert max(lv for _, lv in bus.speech_env) < 0.2


def test_interrupt_clears_envelope_and_drops_audio():
    cfg = Config()
    bus = Bus()
    eng = AudioEngine(_Mini(), cfg, bus, on_audio_chunk=lambda c: None)
    eng._play_cushion = 10.0
    bus.speech_level = 0.9
    bus.speech_env.append((0.0, 0.9))
    bus.interrupt_event.set()
    eng._play_chunk(np.ones(4800, dtype=np.float32) * 0.2)
    assert eng.mini.media.pushed == []   # dropped, belongs to the interrupted turn


def test_request_interrupt_stills_the_wobble():
    bus = Bus()
    bus.speech_env.append((0.0, 0.9))
    bus.speech_level = 0.9
    bus.request_interrupt()
    assert len(bus.speech_env) == 0
    assert bus.speech_level == 0.0


def _max_pitch(ctl, lvl):
    ctl._speech_lvl = lvl
    return max(abs(ctl._speaking(t=i * 0.02)[0][1]) for i in range(200))  # sweep 4 s


def test_speaking_wobble_scales_with_envelope():
    cfg = Config()
    bus = Bus()
    ctl = MotionController(_Mini(), cfg, bus)
    assert _max_pitch(ctl, 1.0) > _max_pitch(ctl, 0.0) * 3  # voice makes the nod


def test_doa_yaw_held_while_robot_speaks():
    """The head keeps facing the user for the whole reply (official-app behavior)."""
    cfg = Config()
    cfg.enable_doa = False
    bus = Bus()
    ctl = MotionController(_Mini(), cfg, bus)
    ctl._doa_yaw = 0.5
    bus.robot_speaking.set()
    for _ in range(50):
        ctl._compute(now=0.0)
    assert ctl._doa_yaw == 0.5           # no decay mid-reply
    bus.robot_speaking.clear()
    for _ in range(50):
        ctl._compute(now=0.0)
    assert ctl._doa_yaw < 0.1            # drifts back once the conversation is quiet
