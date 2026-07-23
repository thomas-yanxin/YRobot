"""TurnGate: barge-in latch, discard, re-force, streak-based unlatch."""

from yrobot.config import Config
from yrobot.turn import TurnGate

# quiet_s=0.7, unlatch_listens=2, reforce_ack_s=1.2, hold_max_s=12, reforce_s=1
CFG = Config()


def test_normal_turn_plays():
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=False)
    g.on_voice(False, 2.0, robot_speaking=False)
    assert g.on_model_audio(3.0)
    assert not g.take_force_listen(3.0)


def test_barge_in_latches_and_discards():
    g = TurnGate(CFG)
    assert g.on_voice(True, 10.0, robot_speaking=True)  # interrupt
    assert g.take_force_listen(10.1)  # user still talking → force listen
    assert not g.on_model_audio(10.2)  # stale burst discarded


def test_onset_while_idle_is_not_interrupt():
    g = TurnGate(CFG)
    assert not g.on_voice(True, 1.0, robot_speaking=False)


def test_unlatch_needs_two_clean_listens():
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=True)
    g.on_listen(0.1)  # user still speaking → streak reset
    assert not g.on_model_audio(0.2)
    g.on_voice(False, 1.0, robot_speaking=False)
    g.on_listen(1.2)  # only 0.2 s of quiet → no count
    assert not g.on_model_audio(1.3)  # audio also resets the streak
    g.on_listen(3.0)  # clean listen #1 — still latched
    assert not g.on_model_audio(3.1)  # ...so resumption here is still caught
    g = _latched_quiet_gate()
    g.on_listen(3.0)
    g.on_listen(4.0)  # clean listen #2 → unlatch
    assert g.on_model_audio(4.1)


def _latched_quiet_gate() -> TurnGate:
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=True)
    g.on_voice(False, 1.0, robot_speaking=False)
    return g


def test_forced_ack_listens_do_not_count():
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=True)
    g.take_force_listen(0.9)  # last force sent as the user stops
    g.on_voice(False, 1.0, robot_speaking=False)
    g.on_listen(2.0)  # 1.1 s after our force → just an ack, no count
    g.on_listen(3.0)  # clean #1
    assert not g.on_model_audio(3.1)
    g = TurnGate(CFG)
    g.on_voice(True, 0.0, robot_speaking=True)
    g.take_force_listen(0.9)
    g.on_voice(False, 1.0, robot_speaking=False)
    g.on_listen(2.5)  # clean #1 (past the ack window)
    g.on_listen(3.5)  # clean #2 → unlatch
    assert g.on_model_audio(3.6)


def test_resumed_monologue_triggers_rate_limited_reforce():
    g = _latched_quiet_gate()
    assert not g.on_model_audio(2.0)  # resumption while user quiet
    assert g.take_force_listen(2.0)  # → re-force
    assert not g.on_model_audio(2.5)
    assert not g.take_force_listen(2.5)  # rate-limited
    assert not g.on_model_audio(3.1)
    assert g.take_force_listen(3.1)  # window elapsed


def test_audio_resets_streak_and_stays_suppressed():
    g = _latched_quiet_gate()
    g.on_listen(3.0)  # clean #1
    assert not g.on_model_audio(3.5)  # resumption → streak reset
    g.on_listen(5.0)  # clean #1 again (past ack window of the 3.5 re-force? no)
    g.on_listen(6.0)
    g.on_listen(7.0)  # two clean listens past the re-force ack window → unlatch
    assert g.on_model_audio(7.1)


def test_hold_cap_reopens_playback():
    g = _latched_quiet_gate()
    assert g.on_model_audio(13.0)  # > hold_max_s after latch


def test_second_onset_during_latch_re_interrupts():
    g = _latched_quiet_gate()
    assert g.on_voice(True, 1.5, robot_speaking=False)  # still latched
