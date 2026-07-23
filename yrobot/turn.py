"""Turn gate — decides what model audio plays and when to force listening.

The server streams a reply many seconds ahead of playback and, after an
interruption, frequently resumes the interrupted monologue (verified against
the live gateway *and* on hardware): a `force_listen` is acked with a listen
delta within ~0.2 s, yet the old reply keeps arriving — typically one slice
after a listen. Playback control therefore lives entirely on the client:

- A user voice onset while the robot is speaking latches a discard: every
  model audio delta is dropped.
- Unlatching requires `unlatch_listens` *clean* consecutive listen deltas:
  the user quiet for `quiet_s`, no force_listen sent by us within
  `reforce_ack_s` (an ack of our own force proves nothing about the model's
  intent), and no model audio since the previous listen (audio resets the
  streak). A single quiet listen is exactly what precedes a monologue
  resumption — hardware logs show "listen, then the old story one slice
  later" — so one is never enough.
- While latched and the user is quiet, any model audio that still arrives is
  the old monologue resuming — it triggers a re-forced listen (rate-limited
  to one per `reforce_s`).
- `hold_max_s` caps the latch so a pathological server can't mute the robot
  forever.
"""

from __future__ import annotations

import threading

from yrobot.config import Config


class TurnGate:
    """Thread-safe: fed from the mic thread and the websocket thread."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._voice = False
        self._latched = False
        self._reforce_pending = False
        self._streak = 0  # consecutive clean listens while latched
        self._last_voice_t = -1e9
        self._hold_t0 = 0.0
        self._last_reforce_t = -1e9
        self._last_force_sent_t = -1e9

    def on_voice(self, active: bool, now: float, robot_speaking: bool) -> bool:
        """Update user-voice state; returns True when this onset is a barge-in."""
        with self._lock:
            onset = active and not self._voice
            if active or self._voice:  # speaking now, or this is the falling edge
                self._last_voice_t = now
            self._voice = active
            if onset and (robot_speaking or self._latched):
                self._latched = True
                self._hold_t0 = now
                self._streak = 0
                self._reforce_pending = False
                return True
            return False

    def on_listen(self, now: float) -> None:
        """A listen delta is the only utterance boundary the server exposes."""
        with self._lock:
            if not self._latched:
                return
            if now - self._hold_t0 > self._cfg.hold_max_s:
                self._unlatch()
                return
            if not self._user_quiet(now):
                self._streak = 0
                return
            if now - self._last_force_sent_t < self._cfg.reforce_ack_s:
                return  # merely acking a force we sent — neither counts nor resets
            self._streak += 1
            if self._streak >= self._cfg.unlatch_listens:
                self._unlatch()

    def on_model_audio(self, now: float) -> bool:
        """Returns True when the delta should be played, False to discard."""
        with self._lock:
            if not self._latched:
                return True
            if now - self._hold_t0 > self._cfg.hold_max_s:
                self._unlatch()
                return True
            self._streak = 0
            if self._user_quiet(now) and now - self._last_reforce_t >= self._cfg.reforce_s:
                # Old monologue resuming while the user waits: shove it back
                # into listening via the next uplink chunk.
                self._reforce_pending = True
                self._last_reforce_t = now
            return False

    def take_force_listen(self, now: float) -> bool:
        """Force-listen flag for the next uplink chunk (consumes any re-force)."""
        with self._lock:
            pending, self._reforce_pending = self._reforce_pending, False
            force = (self._latched and self._voice) or pending
            if force:
                self._last_force_sent_t = now
            return force

    def latched(self) -> bool:
        with self._lock:
            return self._latched

    def reset(self) -> None:
        with self._lock:
            self._unlatch()
            self._reforce_pending = False

    def _unlatch(self) -> None:
        self._latched = False
        self._streak = 0

    def _user_quiet(self, now: float) -> bool:
        return not self._voice and now - self._last_voice_t >= self._cfg.quiet_s
