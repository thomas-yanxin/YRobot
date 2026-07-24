"""Unit tests for the barge-in turn gate — the interruption reliability core."""

from yrobot.turn import LATCH_CAP_S, QUIET_S, TurnGate


def barge(gate: TurnGate, t: float) -> bool:
    return gate.user_frame(voiced=True, robot_audible=True, now=t)


def send_force(gate: TurnGate, input_id: str, t: float) -> None:
    assert gate.chunk_force_listen(t) is True
    gate.force_sent(input_id, t)


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


def test_different_response_ids_do_not_release_stale_monologue():
    gate = TurnGate()
    assert barge(gate, 1.0) is True
    send_force(gate, "forced-1", 1.05)
    for i in range(20):
        assert gate.model_audio(1.10 + i * 0.02, f"branch-{i}") is False
    assert gate.latched


def test_text_branches_are_suppressed_until_safe_listen_boundary():
    gate = TurnGate()
    assert barge(gate, 1.0) is True
    send_force(gate, "forced-1", 1.05)
    assert gate.model_text(1.10, "branch-old-1") is False
    assert gate.model_text(1.12, "branch-old-2") is False
    assert gate.model_listen(1.20, "forced-1") is True
    assert gate.model_text(1.0 + QUIET_S + 0.01, "branch-new") is True
    assert not gate.latched


def test_force_listen_stays_sticky_until_matching_ack():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.chunk_force_listen(1.1) is True
    assert gate.chunk_force_listen(1.2) is True
    gate.force_sent("forced-1", 1.2)
    assert gate.chunk_force_listen(1.3) is True
    assert gate.model_listen(1.4, "forced-1") is True
    assert gate.chunk_force_listen(1.5) is False


def test_new_user_speech_invalidates_an_earlier_listen_ack():
    gate = TurnGate()
    barge(gate, 1.0)
    send_force(gate, "forced-1", 1.05)
    assert gate.model_listen(1.10, "forced-1") is True
    gate.user_frame(True, False, 1.20)
    assert gate.model_audio(1.20 + QUIET_S + 0.1, "different-branch") is False
    send_force(gate, "forced-2", 1.20 + QUIET_S + 0.11)
    assert gate.model_listen(1.20 + QUIET_S + 0.12, "forced-1") is False
    assert gate.model_listen(1.20 + QUIET_S + 0.13, "forced-2") is True


def test_matching_force_then_listen_releases_new_answer_after_quiet():
    gate = TurnGate()
    barge(gate, 1.0)
    send_force(gate, "forced-1", 1.05)
    assert gate.model_listen(1.10, "forced-1") is True
    assert gate.model_audio(1.0 + QUIET_S - 0.01, "too-early") is False
    send_force(gate, "forced-2", 1.0 + QUIET_S + 0.02)
    assert gate.model_listen(1.0 + QUIET_S + 0.05, "forced-2") is True
    assert gate.model_audio(1.0 + QUIET_S + 0.10, "new-answer") is True
    assert not gate.latched


def test_listen_before_first_force_is_not_a_boundary():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.model_listen(1.01, "not-sent") is False
    assert gate.model_audio(1.0 + QUIET_S + 0.1) is False
    assert gate.latched


def test_reading_force_flag_does_not_fake_a_send():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.chunk_force_listen(1.05) is True  # packet may still be dropped
    assert gate.model_listen(1.10, "dropped-packet") is False
    assert gate.chunk_force_listen(1.20) is True


def test_wrong_input_id_cannot_ack_latest_force():
    gate = TurnGate()
    barge(gate, 1.0)
    send_force(gate, "forced-latest", 1.05)
    assert gate.model_listen(1.10, "older-input") is False
    assert gate.latched


def test_latch_timeout_never_auto_releases_stale_output():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.timed_out(1.0 + LATCH_CAP_S + 0.1)
    assert gate.model_audio(1.0 + LATCH_CAP_S + 0.2, "stale") is False
    assert gate.latched
