"""Barge-in turn gate for client-owned interruption boundaries.

``force_listen`` is a per-input override, not a sticky server state, while
``response_id`` identifies one output branch rather than a conversational
turn. Interruption is therefore client-owned:

* a qualified voice onset immediately latches and locally suppresses output;
* every complete one-second input remains forced until the matching
  ``input_id`` returns an explicit ``listen``;
* every newer user-voice frame invalidates that acknowledgement;
* output is admitted again only after the acknowledged force and user quiet.

Pure logic over injected timestamps — no I/O, fully unit-testable.
"""

from __future__ import annotations

QUIET_S = 0.45  # user silence before model output can be the new answer
LATCH_CAP_S = 12.0  # reconnect instead of ever replaying an uncertain old turn


class TurnGate:
    """Decides, per event, whether to barge, force-listen, or discard audio."""

    def __init__(self) -> None:
        self._latched = False
        self._latched_at = 0.0
        self._force_active = False
        self._last_force_at = -1e9
        self._last_forced_input_id = ""
        self._last_voice_at = -1e9
        self._last_listen_at = -1e9
        self._listen_after_force = False

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def force_pending(self) -> bool:
        return self._latched and self._force_active

    def user_frame(self, voiced: bool, robot_audible: bool, now: float) -> bool:
        """Register one VAD frame; return True when an interruption starts."""
        if not voiced:
            return False
        self._last_voice_at = now
        if robot_audible and not self._latched:
            self._latched = True
            self._latched_at = now
            self._force_active = True
            self._listen_after_force = False
            self._last_listen_at = -1e9
            return True
        if self._latched:
            self._force_active = True
            # A listen observed before this newer speech cannot delimit the
            # final user instruction.
            self._listen_after_force = False
        return False

    def model_audio(self, now: float, response_id: str = "") -> bool:
        """Return whether an audio branch belongs after the safe boundary.

        ``response_id`` is intentionally ignored: live traces and the public
        protocol define it at output-branch grain.
        """
        return self._model_output(now)

    def model_text(self, now: float, response_id: str = "") -> bool:
        """Return whether a text branch belongs after the safe boundary."""
        return self._model_output(now)

    def model_listen(self, now: float, input_id: str = "") -> bool:
        """Accept only the listen caused by the latest actually-sent force."""
        if not self._latched:
            return False
        if (
            not input_id
            or input_id != self._last_forced_input_id
            or self._last_force_at < self._latched_at
            or now < self._last_force_at
            or self._last_voice_at > self._last_force_at
        ):
            return False
        self._last_listen_at = now
        self._listen_after_force = True
        self._force_active = False
        return True

    def chunk_force_listen(self, now: float | None = None) -> bool:
        """Whether this complete inference unit must carry force_listen.

        Reading this flag does not consume it. The sender records transmission
        separately with :meth:`force_sent`, so a queued packet that is dropped
        cannot create a fictional acknowledgement boundary.
        """
        return self._latched and self._force_active

    def force_sent(self, input_id: str, now: float) -> None:
        """Record a forced unit at the point the network sender transmits it."""
        if not self._latched or not self._force_active:
            return
        self._last_force_at = now
        self._last_forced_input_id = input_id
        self._listen_after_force = False

    def timed_out(self, now: float) -> bool:
        """An uncertain old turn must trigger reconnect, never auto-release."""
        return self._latched and now - self._latched_at >= LATCH_CAP_S

    def _model_output(self, now: float) -> bool:
        if not self._latched:
            return True
        safe_boundary = (
            self._listen_after_force
            and not self._force_active
            and self._last_listen_at >= self._last_force_at
            and now - self._last_voice_at >= QUIET_S
        )
        if safe_boundary:
            self._unlatch()
            return True
        # Output before the causally-matched boundary is stale. Keep force
        # sticky on every later complete input, exactly like the official UI.
        self._force_active = True
        self._listen_after_force = False
        return False

    def _unlatch(self) -> None:
        self._latched = False
        self._force_active = False
        self._listen_after_force = False
        self._last_forced_input_id = ""
