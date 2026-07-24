from yrobot.state import InteractionPhase, TurnCoordinator


def test_barge_in_fences_audio_until_explicit_listen_delta() -> None:
    turns = TurnCoordinator()
    first_epoch = turns.new_session()

    assert turns.accept_audio("response-1") == first_epoch
    interrupted_epoch = turns.interrupt()
    assert interrupted_epoch == first_epoch + 1
    assert turns.snapshot().phase is InteractionPhase.INTERRUPTED
    assert turns.snapshot().force_listen is True
    assert turns.accept_audio("response-1") is None

    # An already in-flight listen event cannot release the interruption.
    assert turns.model_listening() is None
    assert turns.snapshot().force_listen is True
    assert turns.force_listen_sent(interrupted_epoch)
    assert turns.model_listening() == interrupted_epoch
    assert turns.snapshot().phase is InteractionPhase.LISTENING
    assert turns.snapshot().force_listen is False
    # The ordered listen event starts a new output segment. Response IDs are
    # optional opaque diagnostics and may be reused by the server.
    assert turns.accept_audio("response-1") == interrupted_epoch


def test_session_loss_and_stop_are_hard_output_fences() -> None:
    turns = TurnCoordinator()
    turns.new_session()
    turns.accept_audio("response-1")

    lost_epoch = turns.session_lost()
    assert turns.snapshot().phase is InteractionPhase.CONNECTING
    assert turns.snapshot().drop_output is True
    assert turns.accept_audio("stale") is None

    assert turns.new_session() == lost_epoch + 1
    stopped_epoch = turns.stop()
    assert turns.snapshot().phase is InteractionPhase.STOPPED
    assert turns.interrupt() is None
    assert turns.accept_audio("late") is None
    assert stopped_epoch > lost_epoch


def test_validated_audible_response_can_fence_a_newly_queued_response() -> None:
    turns = TurnCoordinator()
    epoch = turns.new_session()
    assert turns.accept_audio("response-1") == epoch
    assert turns.model_listening() == epoch
    assert turns.accept_audio("response-2") == epoch

    interrupted_epoch = turns.interrupt_if_epoch(epoch)
    assert interrupted_epoch == epoch + 1
    assert turns.snapshot().phase is InteractionPhase.INTERRUPTED
    assert turns.force_listen_sent(interrupted_epoch)
    assert turns.model_listening() == interrupted_epoch
    assert turns.accept_audio("response-2") == interrupted_epoch


def test_audible_tail_can_be_interrupted_after_model_listen_boundary() -> None:
    turns = TurnCoordinator()
    epoch = turns.new_session()
    assert turns.accept_audio("response-1") == epoch
    assert turns.model_listening() == epoch

    interrupted_epoch = turns.interrupt_if_epoch(epoch)
    assert interrupted_epoch == epoch + 1
    assert turns.force_listen_sent(interrupted_epoch)
    assert turns.model_listening() == interrupted_epoch
    assert turns.accept_audio("response-1") == interrupted_epoch


def test_epoch_guard_rejects_vad_from_an_old_session() -> None:
    turns = TurnCoordinator()
    old_epoch = turns.new_session()
    assert turns.accept_audio(None) == old_epoch
    turns.session_lost()
    new_epoch = turns.new_session()
    assert turns.accept_audio(None) == new_epoch

    assert turns.interrupt_if_epoch(old_epoch) is None
    assert turns.snapshot().epoch == new_epoch
    assert turns.snapshot().phase is InteractionPhase.SPEAKING


def test_anonymous_audio_is_fenced_by_listen_segment_not_response_id() -> None:
    turns = TurnCoordinator()
    epoch = turns.new_session()
    assert turns.accept_audio(None) == epoch
    interrupted_epoch = turns.interrupt()
    assert interrupted_epoch == epoch + 1
    assert turns.accept_audio(None) is None

    assert turns.force_listen_sent(interrupted_epoch)
    assert turns.model_listening() == interrupted_epoch
    assert turns.accept_audio(None) == interrupted_epoch


def test_listen_is_accepted_once_force_write_is_in_flight() -> None:
    turns = TurnCoordinator()
    epoch = turns.new_session()
    assert turns.accept_audio("response-1") == epoch
    interrupted_epoch = turns.interrupt()
    assert interrupted_epoch == epoch + 1

    assert turns.force_listen_started(interrupted_epoch)
    assert turns.model_listening() == interrupted_epoch
    assert turns.snapshot().phase is InteractionPhase.LISTENING
    assert not turns.snapshot().drop_output
    assert not turns.force_listen_sent(interrupted_epoch)


def test_silent_old_gate_cannot_interrupt_a_listening_boundary() -> None:
    turns = TurnCoordinator()
    epoch = turns.new_session()
    assert turns.accept_audio("response-1") == epoch
    assert turns.model_listening() == epoch

    assert turns.interrupt_if_epoch(epoch, playback_audible=False) is None
    assert turns.snapshot().phase is InteractionPhase.LISTENING
    assert turns.interrupt_if_epoch(epoch, playback_audible=True) == epoch + 1
