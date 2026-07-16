"""Barge-in decision + the XVF3800 voice/DOA poll that feeds it."""
import numpy as np

from reachy_mini_live_chat.audio.io import AudioEngine
from reachy_mini_live_chat.bus import Bus
from reachy_mini_live_chat.config import Config


class _Media:
    def __init__(self, hw):
        self._hw = hw  # None (read failure) or (angle, speech)
        self.doa_reads = 0

    def get_DoA(self):
        self.doa_reads += 1
        return self._hw


class _Mini:
    def __init__(self, hw):
        self.media = _Media(hw)


def _engine(hw):
    cfg = Config()
    bus = Bus()
    eng = AudioEngine(_Mini(hw), cfg, bus, on_audio_chunk=lambda c: None)
    return eng, bus


def test_hw_poll_updates_voice_flag_and_doa():
    eng, bus = _engine((1.57, True))
    eng._poll_hw_voice(now=100.0)
    assert eng._hw_speech is True
    assert bus.doa_angle == 1.57
    # throttled: a read 10 ms later must not hit USB again
    eng._poll_hw_voice(now=100.01)
    assert eng.mini.media.doa_reads == 1


def test_hw_poll_keeps_flag_on_read_failure():
    eng, bus = _engine((0.5, True))
    eng._poll_hw_voice(now=100.0)
    assert eng._hw_speech is True
    eng.mini.media._hw = None  # transient USB failure
    eng._poll_hw_voice(now=101.0)
    assert eng._hw_speech is True  # previous flag kept


def test_no_voice_does_not_update_doa_angle():
    eng, bus = _engine((2.0, False))
    eng._poll_hw_voice(now=100.0)
    assert eng._hw_speech is False
    assert bus.doa_angle is None  # angle without voice is noise — don't steer


def test_barge_fires_while_robot_speaks():
    eng, bus = _engine((1.57, True))
    bus.robot_speaking.set()
    eng.endpointer._in_speech = True  # endpointer debounced the hw flag
    eng._maybe_barge()
    assert bus.interrupt_event.is_set()


def test_no_barge_when_robot_silent():
    eng, bus = _engine((1.57, True))
    eng.endpointer._in_speech = True
    eng._maybe_barge()
    assert not bus.interrupt_event.is_set()


def test_no_barge_without_speech():
    eng, bus = _engine((1.57, False))
    bus.robot_speaking.set()
    eng._maybe_barge()  # endpointer never entered speech
    assert not bus.interrupt_event.is_set()


def test_endpointer_is_driven_by_hw_flag():
    eng, bus = _engine((1.57, True))
    eng._hw_speech = True
    frame = np.zeros(4800, dtype=np.float32)  # ~300 ms of frames, flag held high
    eng.endpointer.process(frame)
    assert eng.endpointer.in_speech
