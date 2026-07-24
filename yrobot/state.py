"""Thread-safe interaction state and playback generation fencing."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum


class InteractionPhase(StrEnum):
    CONNECTING = "connecting"
    LISTENING = "listening"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class TurnSnapshot:
    epoch: int
    phase: InteractionPhase
    force_listen: bool
    force_listen_sent: bool
    drop_output: bool
    response_id: str | None


class TurnCoordinator:
    """Atomically fences stale audio across barge-ins and session changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch = 0
        self._phase = InteractionPhase.CONNECTING
        self._force_listen = False
        self._force_listen_sent = False
        self._force_listen_in_flight = False
        self._drop_output = True
        self._response_id: str | None = None

    def snapshot(self) -> TurnSnapshot:
        with self._lock:
            return self._snapshot()

    def new_session(self) -> int:
        with self._lock:
            self._epoch += 1
            self._phase = InteractionPhase.LISTENING
            self._force_listen = False
            self._force_listen_sent = False
            self._force_listen_in_flight = False
            self._drop_output = False
            self._response_id = None
            return self._epoch

    def session_lost(self) -> int:
        with self._lock:
            self._epoch += 1
            self._phase = InteractionPhase.CONNECTING
            self._force_listen = False
            self._force_listen_sent = False
            self._force_listen_in_flight = False
            self._drop_output = True
            self._response_id = None
            return self._epoch

    def accept_audio(self, response_id: str | None) -> int | None:
        """Accept audio after the ordered server listen boundary."""
        with self._lock:
            if self._drop_output or self._phase is InteractionPhase.STOPPED:
                return None
            self._phase = InteractionPhase.SPEAKING
            self._response_id = response_id
            return self._epoch

    def interrupt(self) -> int | None:
        """Start one barge-in transaction and return its new epoch."""
        with self._lock:
            if self._drop_output or self._phase is InteractionPhase.STOPPED:
                return None
            return self._interrupt_locked()

    def interrupt_if_epoch(
        self,
        expected_epoch: int,
        *,
        playback_audible: bool = True,
    ) -> int | None:
        """Interrupt one validated playback without crossing a session fence."""

        with self._lock:
            if self._drop_output or self._epoch != expected_epoch:
                return None
            if self._phase is InteractionPhase.LISTENING and not playback_audible:
                return None
            if self._phase not in {InteractionPhase.SPEAKING, InteractionPhase.LISTENING}:
                return None
            return self._interrupt_locked()

    def force_listen_started(self, epoch: int) -> bool:
        """Mark the force-listen write as in flight on the ordered socket."""

        with self._lock:
            if (
                epoch != self._epoch
                or not self._force_listen
                or self._phase is not InteractionPhase.INTERRUPTED
            ):
                return False
            self._force_listen_in_flight = True
            return True

    def force_listen_sent(self, epoch: int) -> bool:
        """Record a force-listen input after its WebSocket write succeeds."""

        with self._lock:
            if (
                epoch != self._epoch
                or not self._force_listen
                or self._phase is not InteractionPhase.INTERRUPTED
            ):
                return False
            self._force_listen_sent = True
            self._force_listen_in_flight = False
            return True

    def model_listening(self) -> int | None:
        """Accept a listen boundary unless it predates the forced input."""

        with self._lock:
            if self._phase is InteractionPhase.STOPPED:
                return None
            if (
                self._force_listen
                and not self._force_listen_sent
                and not self._force_listen_in_flight
            ):
                return None
            return self._model_listening_locked()

    def stop(self) -> int:
        with self._lock:
            self._epoch += 1
            self._phase = InteractionPhase.STOPPED
            self._force_listen = False
            self._force_listen_sent = False
            self._force_listen_in_flight = False
            self._drop_output = True
            self._response_id = None
            return self._epoch

    def _snapshot(self) -> TurnSnapshot:
        return TurnSnapshot(
            epoch=self._epoch,
            phase=self._phase,
            force_listen=self._force_listen,
            force_listen_sent=self._force_listen_sent,
            drop_output=self._drop_output,
            response_id=self._response_id,
        )

    def _interrupt_locked(self) -> int:
        self._epoch += 1
        self._phase = InteractionPhase.INTERRUPTED
        self._force_listen = True
        self._force_listen_sent = False
        self._force_listen_in_flight = False
        self._drop_output = True
        self._response_id = None
        return self._epoch

    def _model_listening_locked(self) -> int:
        self._phase = InteractionPhase.LISTENING
        self._force_listen = False
        self._force_listen_sent = False
        self._force_listen_in_flight = False
        self._drop_output = False
        self._response_id = None
        return self._epoch
