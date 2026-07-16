"""Barge-in decision + the XVF3800 DOA poll (angle only — never a voice source)."""
import numpy as np

from reachy_mini_live_chat.audio.io import AudioEngine
from reachy_mini_live_chat.bus import Bus
from reachy_mini_live_chat.config import Config


class _Media:
    def __init__(self, hw):
        self._hw = hw  # None (read failure) or (angle, speech_flag)
        self.doa_reads = 0

    def get_DoA(self):
        self.doa_reads += 1
        return self._hw


class _Mini:
    def __init__(self, hw):
        self.media = _Media(hw)


def _engine(hw=(1.57, True)):
    cfg = Config()
    bus = Bus()
    eng = AudioEngine(_Mini(hw), cfg, bus, on_audio_chunk=lambda c: None)
    return eng, bus


def test_hw_poll_caches_doa_angle_and_throttles():
    eng, bus = _engine((1.57, True))
    eng._poll_hw_doa(now=100.0)
    assert bus.doa_angle == 1.57
    # throttled: a read 10 ms later must not hit USB again
    eng._poll_hw_doa(now=100.01)
    assert eng.mini.media.doa_reads == 1


def test_hw_poll_ignores_angle_without_sound_activity():
    eng, bus = _engine((2.0, False))
    eng._poll_hw_doa(now=100.0)
    assert bus.doa_angle is None  # angle without activity is stale — don't steer


def test_hw_poll_survives_read_failure():
    eng, bus = _engine(None)
    eng._poll_hw_doa(now=100.0)
    assert bus.doa_angle is None


def test_barge_fires_while_robot_speaks():
    # The firmware flag plays no role: the energy endpointer (AEC'd mic) decides.
    eng, bus = _engine()
    bus.robot_speaking.set()
    eng.endpointer._in_speech = True  # energy gate debounced sustained voice
    eng._maybe_barge()
    assert bus.interrupt_event.is_set()


def test_no_barge_when_robot_silent():
    eng, bus = _engine()
    eng.endpointer._in_speech = True
    eng._maybe_barge()
    assert not bus.interrupt_event.is_set()


def test_no_barge_without_speech():
    eng, bus = _engine()
    bus.robot_speaking.set()
    eng._maybe_barge()  # endpointer never entered speech
    assert not bus.interrupt_event.is_set()


def test_barge_is_latched_per_interrupt():
    eng, bus = _engine()
    bus.robot_speaking.set()
    eng.endpointer._in_speech = True
    eng._maybe_barge()
    assert bus.interrupt_event.is_set()
    bus.emit_count = None  # no-op; ensure a second call doesn't re-request
    eng._maybe_barge()  # already interrupting → no double signal path
    assert bus.interrupt_event.is_set()


def test_energy_endpointer_reaches_in_speech():
    eng, bus = _engine()
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(1600) * 0.005).astype(np.float32)
    loud = (rng.standard_normal(16000) * 0.5).astype(np.float32)  # 1 s of strong voice energy
    for _ in range(20):  # let the floor settle on ambient
        eng.endpointer.process(quiet)
    eng.endpointer.process(loud)
    assert eng.endpointer.in_speech


def test_barge_flushes_partial_uplink_chunk():
    """On barge-in the partial capture buffer ships immediately (with force_listen
    coming from interrupt_event) instead of waiting for the 1 s boundary."""
    sent = []
    cfg = Config()
    bus = Bus()
    eng = AudioEngine(_Mini((1.57, True)), cfg, bus, on_audio_chunk=sent.append)
    bus.robot_speaking.set()
    eng.endpointer._in_speech = True
    eng._chunk_buf = np.zeros(int(0.5 * 16000), dtype=np.float32)  # 500 ms pending
    eng._maybe_barge()
    assert bus.interrupt_event.is_set()
    assert len(sent) == 1 and len(sent[0]) == int(0.5 * 16000)
    assert len(eng._chunk_buf) == 0


def test_barge_flush_skips_tiny_buffer():
    sent = []
    cfg = Config()
    bus = Bus()
    eng = AudioEngine(_Mini((1.57, True)), cfg, bus, on_audio_chunk=sent.append)
    bus.robot_speaking.set()
    eng.endpointer._in_speech = True
    eng._chunk_buf = np.zeros(int(0.05 * 16000), dtype=np.float32)  # 50 ms < gate
    eng._maybe_barge()
    assert bus.interrupt_event.is_set()
    assert sent == []  # not worth its own message; next chunk is imminent


def test_barge_flush_disabled_by_config():
    sent = []
    cfg = Config()
    cfg.omni_barge_flush = False
    bus = Bus()
    eng = AudioEngine(_Mini((1.57, True)), cfg, bus, on_audio_chunk=sent.append)
    bus.robot_speaking.set()
    eng.endpointer._in_speech = True
    eng._chunk_buf = np.zeros(int(0.5 * 16000), dtype=np.float32)
    eng._maybe_barge()
    assert sent == []
