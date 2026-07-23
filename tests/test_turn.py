"""TurnGate: barge-in latch, discard, re-force, unlatch semantics."""

from yrobot.config import Config
from yrobot.turn import TurnGate

CFG = Config()  # quiet_s=0.7, hold_max_s=12, reforce_s=1


def test_normal_turn_plays():
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=False)
    g.on_voice(False, 2.0, robot_speaking=False)
    assert g.on_model_audio(3.0)
    assert not g.take_force_listen()


def test_barge_in_latches_and_discards():
    g = TurnGate(CFG)
    assert g.on_voice(True, 10.0, robot_speaking=True)  # interrupt
    assert g.take_force_listen()  # user still talking → force listen
    assert not g.on_model_audio(10.2)  # stale burst discarded


def test_onset_while_idle_is_not_interrupt():
    g = TurnGate(CFG)
    assert not g.on_voice(True, 1.0, robot_speaking=False)


def test_unlatch_requires_quiet_listen():
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=True)
    g.on_listen(0.1)  # user still speaking → keep latched
    assert not g.on_model_audio(0.2)
    g.on_voice(False, 1.0, robot_speaking=False)
    g.on_listen(1.2)  # only 0.2 s of quiet → keep latched
    assert not g.on_model_audio(1.3)
    g.on_listen(2.0)  # 1 s of quiet → unlatch
    assert g.on_model_audio(2.1)


def test_resumed_monologue_triggers_rate_limited_reforce():
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=True)
    g.on_voice(False, 1.0, robot_speaking=False)
    assert not g.on_model_audio(2.0)  # resumption while user quiet
    assert g.take_force_listen()  # → re-force
    assert not g.on_model_audio(2.5)
    assert not g.take_force_listen()  # rate-limited
    assert not g.on_model_audio(3.1)
    assert g.take_force_listen()  # window elapsed


def test_hold_cap_reopens_playback():
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=True)
    g.on_voice(False, 1.0, robot_speaking=False)
    assert g.on_model_audio(13.0)  # > hold_max_s after latch


def test_second_onset_during_latch_re_interrupts():
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=True)
    g.on_voice(False, 1.0, robot_speaking=False)
    assert g.on_voice(True, 1.5, robot_speaking=False)  # still latched
