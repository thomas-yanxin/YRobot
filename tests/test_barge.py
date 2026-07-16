"""Barge-in decision: energy-VAD candidate + XVF3800 voice-flag confirmation."""
from reachy_mini_live_chat.audio.io import AudioEngine
from reachy_mini_live_chat.bus import Bus
from reachy_mini_live_chat.config import Config


class _Media:
    def __init__(self, hw_speech):
        self._hw_speech = hw_speech
        self.doa_reads = 0

    def get_DoA(self):
        self.doa_reads += 1
        if self._hw_speech is None:
            return None  # no ReSpeaker (sim)
        return (1.57, self._hw_speech)


class _Mini:
    def __init__(self, hw_speech):
        self.media = _Media(hw_speech)


def _engine(hw_speech, hw_confirm=True):
    cfg = Config()
    cfg.stub = True
    cfg.vad_barge_hw_confirm = hw_confirm
    bus = Bus()
    eng = AudioEngine(_Mini(hw_speech), cfg, bus, on_audio_chunk=lambda c: None)
    bus.robot_speaking.set()
    eng.endpointer._in_speech = True  # energy VAD says "voice candidate"
    return eng, bus


def test_barge_rejected_when_hw_says_no_voice():
    # Residual echo / servo noise fools the energy VAD, but the XVF3800 post-AEC
    # voice flag says no human → the reply must NOT be cut.
    eng, bus = _engine(hw_speech=False)
    eng._maybe_barge()
    assert not bus.interrupt_event.is_set()
    assert eng.mini.media.doa_reads == 1


def test_barge_fires_when_hw_confirms_voice():
    eng, bus = _engine(hw_speech=True)
    eng._maybe_barge()
    assert bus.interrupt_event.is_set()


def test_barge_falls_back_to_energy_without_respeaker():
    # get_DoA() -> None (sim / board absent): the energy decision stands alone.
    eng, bus = _engine(hw_speech=None)
    eng._maybe_barge()
    assert bus.interrupt_event.is_set()


def test_barge_skips_hw_read_when_disabled():
    eng, bus = _engine(hw_speech=False, hw_confirm=False)
    eng._maybe_barge()
    assert bus.interrupt_event.is_set()
    assert eng.mini.media.doa_reads == 0


def test_no_barge_when_robot_silent():
    eng, bus = _engine(hw_speech=True)
    bus.robot_speaking.clear()
    eng._maybe_barge()
    assert not bus.interrupt_event.is_set()
