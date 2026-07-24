"""Barge-in turn gate and the duck-and-verify second stage.

The gateway's ``force_listen`` is advisory: generation does not reliably stop
and, once the user goes quiet, the model often *resumes* the interrupted
monologue. Interruption therefore has to be client-owned:

* the instant the user speaks over robot audio, playback is flushed locally
  and every stale audio delta of that turn is discarded;
* every uplink chunk carries ``force_listen`` while the user keeps talking,
  and a rate-limited re-force fires when discarded audio arrives after the
  user went quiet;
* the discard latch releases only after **two consecutive clean listens** —
  a listen with the user quiet for ≥ ``QUIET_S`` that is not merely the ack
  of a force we sent within ``FORCE_ACK_S``. Stale model audio resets the
  streak (the resume signature is "listen, then old audio one slice later").

Pure logic over injected timestamps — no I/O, fully unit-testable.
"""

from __future__ import annotations

QUIET_S = 0.7  # user silence needed for a listen to count as clean
FORCE_ACK_S = 1.2  # a listen this close to our force is just its ack
REFORCE_S = 1.0  # min spacing of re-forces triggered by stale audio
CLEAN_LISTENS = 2  # consecutive clean listens required to unlatch
LATCH_CAP_S = 12.0  # absolute upper bound on discarding


class TurnGate:
    """Decides, per event, whether to barge, force-listen, or discard audio."""

    def __init__(self) -> None:
        self._latched = False
        self._latched_at = 0.0
        self._clean_streak = 0
        self._pending_force = False
        self._last_force_at = -1e9
        self._last_voice_at = -1e9

    @property
    def latched(self) -> bool:
        return self._latched

    def user_frame(self, voiced: bool, robot_audible: bool, now: float) -> bool:
        """Register one VAD frame. Returns True when a barge-in starts —
        the caller must flush local playback immediately."""
        if not voiced:
            return False
        self._last_voice_at = now
        if robot_audible and not self._latched:
            self._latched = True
            self._latched_at = now
            self._clean_streak = 0
            self._pending_force = True
            return True
        if self._latched:
            self._pending_force = True  # keep forcing while the user talks
        return False

    def model_audio(self, now: float) -> bool:
        """Audio delta arrived. Returns True if it may be played."""
        if not self._latched:
            return True
        self._clean_streak = 0  # the monologue is still coming
        if now - self._last_voice_at >= QUIET_S and now - self._last_force_at >= REFORCE_S:
            self._pending_force = True  # user is quiet yet audio persists: re-force
        self._maybe_expire(now)
        return not self._latched

    def model_listen(self, now: float) -> None:
        """Listen delta arrived — the only semantic utterance boundary."""
        if not self._latched:
            return
        quiet = now - self._last_voice_at >= QUIET_S
        genuine = now - self._last_force_at >= FORCE_ACK_S
        if quiet and genuine:
            self._clean_streak += 1
            if self._clean_streak >= CLEAN_LISTENS:
                self._unlatch()
        else:
            self._clean_streak = 0
        self._maybe_expire(now)

    def chunk_force_listen(self, now: float) -> bool:
        """Whether the uplink chunk being sent now must carry force_listen."""
        if self._latched and self._pending_force:
            self._pending_force = False
            self._last_force_at = now
            return True
        return False

    def _maybe_expire(self, now: float) -> None:
        if self._latched and now - self._latched_at >= LATCH_CAP_S:
            self._unlatch()

    def _unlatch(self) -> None:
        self._latched = False
        self._pending_force = False
        self._clean_streak = 0


class DuckVerifier:
    """Second barge-in stage: prove the voice survives the robot's silence.

    A candidate only *ducks* playback. The settle time covers everything
    between "we asked for silence" and "the microphone can prove it" — the
    pipeline flush, sink residue, the acoustic path and the capture/XVF
    path back. The shared GStreamer pipeline reports min_latency ≈ 286 ms
    (hardware logs), and a 150 ms settle let in-flight echo of the
    just-muted speech confirm a false barge-in — hence 0.6 s. After the
    settle, sustained voice commits the destructive flush; a quiet window
    means the candidate was echo and playback resumes.

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
