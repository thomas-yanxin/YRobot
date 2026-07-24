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


def test_response_identity_releases_new_reply_but_never_old_tail():
    gate = TurnGate()
    assert gate.model_audio(0.5, "response-old") is True
    assert barge(gate, 1.0) is True
    assert gate.model_audio(1.1, "response-old") is False
    assert gate.model_audio(1.2, "response-new") is True
    assert not gate.latched
    # The old identity stays invalid even after the new response starts.
    assert gate.model_audio(1.3, "response-old") is False


def test_text_identity_tracks_response_before_first_audio():
    gate = TurnGate()
    assert gate.model_text(0.5, "response-old") is True
    assert barge(gate, 1.0) is True
    assert gate.model_text(1.1, "response-old") is False
    assert gate.model_text(1.2, "response-new") is True
    assert gate.model_audio(1.3, "response-new") is True


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
