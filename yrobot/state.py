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
        self._drop_output = True
        self._response_id: str | None = None
        self._blocked_response_id: str | None = None

    def snapshot(self) -> TurnSnapshot:
        with self._lock:
            return self._snapshot()

    def new_session(self) -> int:
        with self._lock:
            self._epoch += 1
            self._phase = InteractionPhase.LISTENING
            self._force_listen = False
            self._force_listen_sent = False
            self._drop_output = False
            self._response_id = None
            self._blocked_response_id = None
            return self._epoch

    def session_lost(self) -> int:
        with self._lock:
            self._epoch += 1
            self._phase = InteractionPhase.CONNECTING
            self._force_listen = False
            self._force_listen_sent = False
            self._drop_output = True
            self._response_id = None
            self._blocked_response_id = None
            return self._epoch

    def accept_audio(self, response_id: str | None) -> int | None:
        """Return the current epoch, or ``None`` while old output is fenced."""
        with self._lock:
            if self._drop_output or self._phase is InteractionPhase.STOPPED:
                return None
            if self._blocked_response_id is not None and (
                response_id is None or response_id == self._blocked_response_id
            ):
                return None
            if response_id is not None:
                self._blocked_response_id = None
            self._phase = InteractionPhase.SPEAKING
            self._response_id = response_id
            return self._epoch

    def interrupt(self) -> int | None:
        """Start one barge-in transaction and return its new epoch."""
        with self._lock:
            if self._drop_output or self._phase is InteractionPhase.STOPPED:
                return None
            self._epoch += 1
            self._phase = InteractionPhase.INTERRUPTED
            self._force_listen = True
            self._force_listen_sent = False
            self._drop_output = True
            self._blocked_response_id = self._response_id
            self._response_id = None
            return self._epoch

    def force_listen_sent(self, epoch: int) -> bool:
        """Record a force-listen input after its WebSocket send succeeds."""

        with self._lock:
            if (
                epoch != self._epoch
                or not self._force_listen
                or self._phase is not InteractionPhase.INTERRUPTED
            ):
                return False
            self._force_listen_sent = True
            return True

    def model_listening(self) -> int | None:
        """Accept a listen boundary unless it predates the forced input."""

        with self._lock:
            if self._phase is InteractionPhase.STOPPED:
                return None
            if self._force_listen and not self._force_listen_sent:
                return None
            self._phase = InteractionPhase.LISTENING
            self._force_listen = False
            self._force_listen_sent = False
            self._drop_output = False
            self._response_id = None
            return self._epoch

    def stop(self) -> int:
        with self._lock:
            self._epoch += 1
            self._phase = InteractionPhase.STOPPED
            self._force_listen = False
            self._force_listen_sent = False
            self._drop_output = True
            self._response_id = None
            self._blocked_response_id = None
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
