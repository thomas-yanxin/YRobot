"""Unit tests for the barge-in turn gate and duck verifier — the reliability core."""

from yrobot.turn import (
    LATCH_CAP_S,
    QUIET_S,
    USER_TAIL_S,
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


def test_different_response_ids_do_not_release_stale_monologue():
    gate = TurnGate()
    assert barge(gate, 1.0) is True
    assert gate.chunk_force_listen(1.05) is True
    for i in range(20):
        assert gate.model_audio(1.10 + i * 0.02, f"branch-{i}") is False
    assert gate.latched


def test_text_branches_are_suppressed_until_safe_listen_boundary():
    gate = TurnGate()
    assert barge(gate, 1.0) is True
    gate.chunk_force_listen(1.05)
    assert gate.model_text(1.10, "branch-old-1") is False
    assert gate.model_text(1.12, "branch-old-2") is False
    assert gate.model_listen(1.20) is True
    assert gate.model_text(1.0 + QUIET_S + 0.01, "branch-new") is True
    assert not gate.latched


def test_force_listen_rides_chunks_while_user_talks():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.chunk_force_listen(1.1) is True
    assert gate.chunk_force_listen(1.2) is False  # consumed until re-armed
    barge(gate, 1.4)
    assert gate.chunk_force_listen(1.5) is True


def test_new_user_speech_invalidates_an_earlier_listen_ack():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.chunk_force_listen(1.05) is True
    assert gate.model_listen(1.10) is True
    gate.user_frame(True, False, 1.20)
    assert gate.model_audio(1.20 + QUIET_S + 0.1, "different-branch") is False
    assert gate.chunk_force_listen(1.20 + QUIET_S + 0.11) is True


def test_final_user_tail_requires_force_then_listen_before_new_answer():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.chunk_force_listen(1.05) is True
    assert gate.model_listen(1.10) is True
    assert gate.should_flush_user_tail(1.0 + USER_TAIL_S + 0.01) is True
    assert gate.model_audio(1.0 + QUIET_S + 0.01, "stale") is False
    assert gate.chunk_force_listen(1.0 + QUIET_S + 0.02) is True
    assert gate.model_listen(1.0 + QUIET_S + 0.05) is True
    assert gate.model_audio(1.0 + QUIET_S + 0.10, "new-answer") is True
    assert not gate.latched


def test_listen_before_first_force_is_not_a_boundary():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.model_listen(1.01) is False
    assert gate.model_audio(1.0 + QUIET_S + 0.1) is False
    assert gate.latched


def test_user_tail_flushes_only_once_per_voice_run():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.should_flush_user_tail(1.0 + USER_TAIL_S + 0.01) is True
    assert gate.should_flush_user_tail(1.0 + USER_TAIL_S + 0.02) is False
    gate.user_frame(True, False, 2.0)
    assert gate.should_flush_user_tail(2.0 + USER_TAIL_S + 0.01) is True


def test_cancel_barge_releases_false_duck():
    gate = TurnGate()
    barge(gate, 1.0)
    gate.cancel_barge()
    assert gate.model_audio(1.1, "old-branch") is True
    assert not gate.latched


def test_latch_timeout_never_auto_releases_stale_output():
    gate = TurnGate()
    barge(gate, 1.0)
    assert gate.timed_out(1.0 + LATCH_CAP_S + 0.1)
    assert gate.model_audio(1.0 + LATCH_CAP_S + 0.2, "stale") is False
    assert gate.latched


def test_verifier_early_resumes_on_pure_silence_with_cooldown():
    v = DuckVerifier()
    v.start(0.0)
    t = 0.0
    while t < DuckVerifier.SETTLE_S:  # muted speech still in the air
        assert v.frame(True, t) is None
        t += 0.02
    verdict = None
    while verdict is None:  # echo died with the speaker: resume early
        verdict = v.frame(False, t)
        t += 0.02
    assert verdict == "resume"
    assert t <= DuckVerifier.SETTLE_S + DuckVerifier.EARLY_RESUME_S + 0.05
    assert not v.ready(t)  # the resumed tail echoes: cooldown blocks a re-duck
    assert v.ready(t + DuckVerifier.COOLDOWN_S)


def test_verifier_waits_out_the_window_after_a_voice_blip():
    v = DuckVerifier()
    v.start(0.0)
    t = DuckVerifier.SETTLE_S + 0.01
    v.frame(True, t)  # a single blip forfeits the early resume
    assert v.frame(False, t + 0.02) is None
    assert v.frame(False, DuckVerifier.SETTLE_S + DuckVerifier.EARLY_RESUME_S + 0.1) is None
    assert v.frame(False, DuckVerifier.WINDOW_S) == "resume"


def test_verifier_commits_on_post_settle_voice():
    v = DuckVerifier()
    v.start(0.0)
    t = DuckVerifier.SETTLE_S + 0.01
    assert v.frame(True, t) is None
    assert v.frame(True, t + 0.02) == "commit"
    assert not v.active


def test_verifier_tolerates_flickering_suppressed_speech():
    # XVF double-talk suppression makes real speech flicker: hits need not
    # be consecutive.
    v = DuckVerifier()
    v.start(0.0)
    t = DuckVerifier.SETTLE_S + 0.01
    assert v.frame(True, t) is None
    for i in range(10):  # suppressor swallows a stretch
        assert v.frame(False, t + 0.02 + i * 0.02) is None
    assert v.frame(True, t + 0.24) == "commit"


def test_verifier_inactive_returns_none():
    assert DuckVerifier().frame(True, 5.0) is None


def test_verifier_strong_voice_commits_during_settle():
    # A short interjection ends before the settle does — strong frames
    # (level above the in-flight echo prediction) must count immediately.
    v = DuckVerifier()
    v.start(0.0)
    assert v.frame(True, 0.10, strong=True) is None
    assert v.frame(True, 0.12, strong=True) == "commit"
    assert not v.active


def test_verifier_retry_window_opens_after_cooldown():
    v = DuckVerifier()
    v.start(0.0)
    t = DuckVerifier.SETTLE_S
    verdict = None
    while verdict is None:  # silence: early resume
        verdict = v.frame(False, t)
        t += 0.02
    assert verdict == "resume"
    assert not v.in_retry(t + 0.5)  # cooldown covers the resumed tail's echo
    assert v.in_retry(t + DuckVerifier.COOLDOWN_S + 0.1)  # the user insisting
    assert not v.in_retry(t + DuckVerifier.RETRY_S + 0.1)  # window closed


def test_verifier_no_retry_without_a_prior_resume():
    assert DuckVerifier().in_retry(5.0) is False


def test_verifier_weak_settle_voice_still_ignored():
    v = DuckVerifier()
    v.start(0.0)
    assert v.frame(True, 0.10) is None  # in-flight echo level: no evidence
    t = DuckVerifier.SETTLE_S
    verdict = None
    while verdict is None:
        verdict = v.frame(False, t)
        t += 0.02
    assert verdict == "resume"
