"""Turn gate — decides what model audio plays and when to force listening.

The server streams a reply many seconds ahead of playback and, after an
interruption, frequently resumes the interrupted monologue (verified against
the live gateway: a `force_listen` is acked with a listen delta within ~0.2 s,
yet story audio keeps arriving for 10+ s). Playback control therefore lives
entirely on the client:

- A user voice onset while the robot is speaking latches a discard: every
  model audio delta is dropped until the model emits a `listen` delta *after*
  the user has been quiet for `quiet_s`.
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
        self._last_voice_t = -1e9
        self._hold_t0 = 0.0
        self._last_reforce_t = -1e9

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
                self._reforce_pending = False
                return True
            return False

    def on_listen(self, now: float) -> None:
        """A listen delta is the only utterance boundary the server exposes."""
        with self._lock:
            if self._latched and self._user_quiet(now):
                self._latched = False

    def on_model_audio(self, now: float) -> bool:
        """Returns True when the delta should be played, False to discard."""
        with self._lock:
            if not self._latched:
                return True
            if now - self._hold_t0 > self._cfg.hold_max_s:
                self._latched = False
                return True
            if self._user_quiet(now) and now - self._last_reforce_t >= self._cfg.reforce_s:
                # Old monologue resuming while the user waits: shove it back
                # into listening via the next uplink chunk.
                self._reforce_pending = True
                self._last_reforce_t = now
            return False

    def take_force_listen(self) -> bool:
        """Force-listen flag for the next uplink chunk (consumes any re-force)."""
        with self._lock:
            pending, self._reforce_pending = self._reforce_pending, False
            return (self._latched and self._voice) or pending

    def latched(self) -> bool:
        with self._lock:
            return self._latched

    def reset(self) -> None:
        with self._lock:
            self._latched = False
            self._reforce_pending = False

    def _user_quiet(self, now: float) -> bool:
        return not self._voice and now - self._last_voice_t >= self._cfg.quiet_s
