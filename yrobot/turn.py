"""Barge-in turn gate and the duck-and-verify second stage.

The gateway's ``force_listen`` is advisory and ``response_id`` identifies one
output *branch*, not one conversational turn. A long stale monologue can
therefore contain many different response IDs. Interruption is client-owned:

* a qualified voice candidate immediately latches, requests ``force_listen``
  and locally suppresses every text/audio output;
* every new user-voice frame invalidates an earlier listen acknowledgement;
* output is admitted again only after the last forced uplink has been followed
  by an explicit ``listen`` event and the user has finished speaking.

Pure logic over injected timestamps — no I/O, fully unit-testable.
"""

from __future__ import annotations

QUIET_S = 0.45  # user silence before model output can be the new answer
USER_TAIL_S = 0.18  # flush the final partial uplink shortly after speech ends
REFORCE_S = 0.6  # min spacing of re-forces triggered by stale output
LATCH_CAP_S = 12.0  # reconnect instead of ever replaying an uncertain old turn


class TurnGate:
    """Decides, per event, whether to barge, force-listen, or discard audio."""

    def __init__(self) -> None:
        self._latched = False
        self._latched_at = 0.0
        self._pending_force = False
        self._last_force_at = -1e9
        self._last_voice_at = -1e9
        self._last_listen_at = -1e9
        self._listen_after_force = False
        self._tail_flushed_voice_at = -1e9

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def force_pending(self) -> bool:
        return self._latched and self._pending_force

    def user_frame(self, voiced: bool, robot_audible: bool, now: float) -> bool:
        """Register one VAD frame; return True when an interruption starts."""
        if not voiced:
            return False
        self._last_voice_at = now
        if robot_audible and not self._latched:
            self._latched = True
            self._latched_at = now
            self._pending_force = True
            self._listen_after_force = False
            self._last_listen_at = -1e9
            self._tail_flushed_voice_at = -1e9
            return True
        if self._latched:
            self._pending_force = True
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

    def model_listen(self, now: float) -> bool:
        """Record a semantic boundary; return True when it acknowledges force."""
        if not self._latched:
            return False
        if self._last_force_at < self._latched_at or now < self._last_force_at:
            return False
        self._last_listen_at = now
        self._listen_after_force = True
        return True

    def chunk_force_listen(self, now: float) -> bool:
        """Whether the uplink chunk being sent now must carry force_listen."""
        if self._latched and self._pending_force:
            self._pending_force = False
            self._last_force_at = now
            self._listen_after_force = False
            return True
        return False

    def should_flush_user_tail(self, now: float) -> bool:
        """Request one partial chunk after the latest user-voice run ends."""
        tail_waiting = self._last_voice_at > self._tail_flushed_voice_at
        if self._latched and tail_waiting and now - self._last_voice_at >= USER_TAIL_S:
            self._tail_flushed_voice_at = self._last_voice_at
            self._pending_force = True
            self._listen_after_force = False
            return True
        return False

    def cancel_barge(self) -> None:
        """A duck was verified as echo; return to the interrupted output."""
        self._unlatch()

    def timed_out(self, now: float) -> bool:
        """An uncertain old turn must trigger reconnect, never auto-release."""
        return self._latched and now - self._latched_at >= LATCH_CAP_S

    def _model_output(self, now: float) -> bool:
        if not self._latched:
            return True
        safe_boundary = (
            self._listen_after_force
            and not self._pending_force
            and self._last_listen_at >= self._last_force_at
            and now - self._last_voice_at >= QUIET_S
        )
        if safe_boundary:
            self._unlatch()
            return True
        if now - self._last_force_at >= REFORCE_S:
            self._pending_force = True
            self._listen_after_force = False
        return False

    def _unlatch(self) -> None:
        self._latched = False
        self._pending_force = False
        self._listen_after_force = False


class DuckVerifier:
    """Second barge-in stage: prove the voice survives the robot's silence.

    A candidate already ducks playback, provisionally forces the model to
    listen, and suppresses old output. This verifier decides only whether
    that provisional interruption becomes a destructive flush. The settle
    time covers the pipeline flush, sink residue, acoustic path and
    capture/XVF path back. The shared GStreamer pipeline reports
    min_latency ≈ 286 ms (hardware logs), and a 150 ms settle let in-flight
    echo of the just-muted speech confirm a false barge-in — hence 0.6 s.
    After the settle, sustained voice commits; a quiet window means the
    candidate was echo and playback resumes.

    Pure logic over injected timestamps — no I/O, fully unit-testable.
    """

    SETTLE_S = 0.6
    WINDOW_S = 1.5  # total, measured from the duck request
    # Evidence frames are already streak-confirmed upstream (60 ms of raw
    # VAD each). XVF double-talk suppression releases slowly and leaves
    # real interrupting speech flickering (hardware log 2026-07-24: a
    # genuine barge produced hits but never three *consecutive* confirmed
    # frames), so commit on accumulated hits, not on a streak.
    CONFIRM_HITS = 2
    EARLY_RESUME_S = 0.25  # pure post-settle silence: don't wait out the window
    COOLDOWN_S = 1.2  # the resumed tail echoes too — block back-to-back ducks
    # A medium-level short interjection is physically indistinguishable from
    # echo by one verify (below the envelope prediction, over before the
    # settle ends) — but a user who was wrongly resumed retries within
    # seconds. A second qualified candidate right after a resume commits
    # directly, no second verify. The cooldown still covers the resumed
    # tail's own onset echo, so the retry window opens after it.
    RETRY_S = 3.0

    def __init__(self) -> None:
        self.active = False
        self._settle_end = 0.0
        self._deadline = 0.0
        self._hits = 0
        self._cooldown_until = 0.0
        self._retry_until = 0.0

    def ready(self, now: float) -> bool:
        """Whether a new duck may start (not verifying, not cooling down)."""
        return not self.active and now >= self._cooldown_until

    def in_retry(self, now: float) -> bool:
        """A candidate arriving now is the user insisting after a wrong
        resume — commit it directly."""
        return self.ready(now) and now < self._retry_until

    def start(self, now: float) -> None:
        self.active = True
        self._settle_end = now + self.SETTLE_S
        self._deadline = now + self.WINDOW_S
        self._hits = 0

    def frame(self, voiced: bool, now: float, strong: bool = False) -> str | None:
        """Feed one VAD frame; returns "commit", "resume" or None.

        ``strong`` marks a frame whose level clearly exceeds the in-flight
        echo prediction. Such frames count as evidence even during the
        settle — a short interjection ("等一下…") often ends before the
        settle does, and discarding it made the robot pause briefly and
        then carry on (hardware log 2026-07-24, 8th run). Strong evidence
        also cuts commit latency for clear barges from 0.6 s+ to ~150 ms.
        """
        if not self.active:
            return None
        if voiced and (strong or now >= self._settle_end):
            self._hits += 1
            if self._hits >= self.CONFIRM_HITS:
                self.active = False
                return "commit"
        if now < self._settle_end:
            return None
        early = self._hits == 0 and now >= self._settle_end + self.EARLY_RESUME_S
        if early or now >= self._deadline:
            self.active = False
            self._cooldown_until = now + self.COOLDOWN_S
            self._retry_until = now + self.RETRY_S
            return "resume"
        return None
