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
    assert turns.accept_audio("response-1") is None
    assert turns.accept_audio("response-2") == interrupted_epoch


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
