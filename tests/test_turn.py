"""Unit tests for the barge-in turn gate and duck verifier — the reliability core."""

from yrobot.turn import (
    CLEAN_LISTENS,
    FORCE_ACK_S,
    LATCH_CAP_S,
    QUIET_S,
    DuckVerifier,
    TurnGate,
)


def barge(gate: TurnGate, t: float) -> bool:
    return gate.user_frame(voiced=True, robot_audible=True, now=t)


def test_barge_latches_and_requests_flush():
    gate = TurnGate()
    assert barge(gate, 1.0) is True
    assert gate.latched
    assert barge(gate, 1.02) is False  # already latched: no second flush


def test_no_barge_when_robot_silent():
    gate = TurnGate()
    assert gate.user_frame(voiced=True, robot_audible=False, now=1.0) is False
    assert not gate.latched


def test_stale_audio_discarded_while_latched():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.model_audio(1.5) is False
    assert gate.model_audio(2.0) is False


def test_force_listen_rides_chunks_while_user_talks():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.chunk_force_listen(1.1) is True
    assert gate.chunk_force_listen(1.2) is False  # consumed until re-armed
    barge(gate, 1.4)
    assert gate.chunk_force_listen(1.5) is True


def test_reforce_when_stale_audio_arrives_after_quiet():
    gate = TurnGate()
    barge(gate, 1.0)
    gate.chunk_force_listen(1.1)
    gate.model_audio(1.1 + QUIET_S + 1.1)  # quiet + past re-force spacing
    assert gate.chunk_force_listen(1.1 + QUIET_S + 1.2) is True


def test_unlatch_needs_two_clean_listens():
    gate = TurnGate()
    barge(gate, 1.0)
    t = 1.0 + FORCE_ACK_S + QUIET_S + 0.1
    gate.model_listen(t)
    assert gate.latched  # one clean listen is the resume signature's prefix
    gate.model_listen(t + 1.0)
    assert not gate.latched


def test_model_audio_resets_clean_streak():
    gate = TurnGate()
    barge(gate, 1.0)
    t = 1.0 + FORCE_ACK_S + QUIET_S + 0.1
    gate.model_listen(t)
    gate.model_audio(t + 0.2)  # the resumed monologue arrives
    gate.model_listen(t + 1.0)
    assert gate.latched  # streak restarted
    gate.model_listen(t + 2.0)
    assert not gate.latched
    assert CLEAN_LISTENS == 2


def test_listen_near_our_force_is_only_an_ack():
    gate = TurnGate()
    barge(gate, 1.0)
    gate.chunk_force_listen(5.0)
    gate.model_listen(5.0 + FORCE_ACK_S - 0.1)  # ack, not genuine
    gate.model_listen(5.0 + FORCE_ACK_S - 0.05)
    assert gate.latched


def test_latch_expires_at_cap():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.model_audio(1.0 + LATCH_CAP_S + 0.1) is True  # cap released it
    assert not gate.latched


def test_verifier_ignores_in_flight_echo_during_settle():
    v = DuckVerifier()
    v.start(0.0)
    t = 0.0
    while t < DuckVerifier.SETTLE_S:  # muted speech still in the air
        assert v.frame(True, t) is None
        t += 0.02
    # echo dies with the speaker silent: quiet until the window closes
    while t < DuckVerifier.WINDOW_S:
        assert v.frame(False, t) is None
        t += 0.02
    assert v.frame(False, DuckVerifier.WINDOW_S) == "resume"
    assert not v.active


def test_verifier_commits_on_sustained_post_settle_voice():
    v = DuckVerifier()
    v.start(0.0)
    t = DuckVerifier.SETTLE_S + 0.01
    assert v.frame(True, t) is None
    assert v.frame(True, t + 0.02) is None
    assert v.frame(True, t + 0.04) == "commit"
    assert not v.active


def test_verifier_streak_must_be_consecutive():
    v = DuckVerifier()
    v.start(0.0)
    t = DuckVerifier.SETTLE_S + 0.01
    v.frame(True, t)
    v.frame(True, t + 0.02)
    v.frame(False, t + 0.04)  # gap resets the evidence
    assert v.frame(True, t + 0.06) is None
    assert v.frame(True, t + 0.08) is None
    assert v.frame(True, t + 0.10) == "commit"


def test_verifier_inactive_returns_none():
    assert DuckVerifier().frame(True, 5.0) is None
