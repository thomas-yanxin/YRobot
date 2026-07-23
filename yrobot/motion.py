"""Puppeteer — one thread owns all set_target calls.

Layering (composed daemon-side, so everything blends instead of snapping):
- daemon head-wobble converts the actual speaker audio into speech motion;
- daemon face-tracking aims the head at the user, blended by a weight we ramp
  with the conversation state;
- this thread adds breath, sway, antenna moods and body orientation (DOA), all
  low-pass filtered so state changes glide.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np

from yrobot.config import Config
from yrobot.state import Shared

logger = logging.getLogger(__name__)


def _lp(current: float, target: float, dt: float, tau: float) -> float:
    return current + (target - current) * (1.0 - math.exp(-dt / tau))


class Puppeteer(threading.Thread):
    def __init__(self, mini, cfg: Config, state: Shared):
        super().__init__(name="yrobot-motion", daemon=True)
        self._mini = mini
        self._cfg = cfg
        self._state = state
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2.0)

    def run(self) -> None:
        from reachy_mini.utils import create_head_pose

        cfg, state = self._cfg, self._state
        dt = 1.0 / cfg.motion_hz
        t0 = time.monotonic()
        # filtered pose components
        z = pitch = roll = ant = boost = 0.0
        yaw = state.body_yaw
        track_weight = -1.0
        last_weight_change = 0.0

        while not self._stop.is_set():
            now = time.monotonic()
            t = now - t0
            speaking = state.robot_speaking()
            listening = state.voice_active
            ready = state.ready
            # "perk" impulse decaying from the last voice onset
            boost = math.exp(-(now - state.last_voice_onset) / 1.2)

            # -- face tracking weight (rate-limited daemon command) ----------
            want = (0.0 if not ready else
                    cfg.track_weight_listen if listening else
                    cfg.track_weight_speak if speaking else
                    cfg.track_weight_idle)
            if want != track_weight and now - last_weight_change > 0.5:
                try:
                    self._mini.start_head_tracking(weight=want)
                    track_weight, last_weight_change = want, now
                except Exception:
                    logger.debug("head tracking unavailable", exc_info=True)

            # -- posture targets ---------------------------------------------
            if not ready:  # visibly dozing while the session reconnects
                tz, tpitch, troll, tant = -0.012, math.radians(14), 0.0, 2.0
            else:
                breath_hz = 0.30 if (listening or speaking) else 0.20
                tz = 0.003 * math.sin(2 * math.pi * breath_hz * t)
                tpitch = math.radians(2.0 * math.sin(2 * math.pi * 0.13 * t + 0.7)
                                       - 4.0 * max(boost, 1.0 if listening else 0.0))
                troll = math.radians(1.5 * math.sin(2 * math.pi * 0.07 * t))
                tz += 0.004 * boost
                # antennas: 0.17 upright-neutral, →0 perked, →3 drooped
                perk = max(boost, 1.0 if listening else 0.0)
                wiggle = 0.10 * math.sin(2 * math.pi * 2.4 * t) if speaking else \
                         0.06 * math.sin(2 * math.pi * 0.09 * t)
                tant = 0.17 - 0.14 * perk + wiggle

            tau = 0.6 if not ready else (0.15 if boost > 0.3 else 0.35)
            z = _lp(z, tz, dt, tau)
            pitch = _lp(pitch, tpitch, dt, tau)
            roll = _lp(roll, troll, dt, tau)
            ant = _lp(ant, tant, dt, tau)

            # -- body yaw toward the speaker, slow recenter when alone -------
            if now - max(state.last_voice_onset, state.last_voice_end) > 45.0:
                state.yaw_target = _lp(state.yaw_target, 0.0, dt, 20.0)
            err = state.yaw_target - yaw
            yaw += float(np.clip(err, -1.6 * dt, 1.6 * dt))
            state.body_yaw = yaw

            try:
                self._mini.set_target(
                    head=create_head_pose(z=z, roll=roll, pitch=pitch, degrees=False),
                    antennas=[-ant, ant],
                    body_yaw=yaw,
                )
            except Exception:
                logger.debug("set_target failed", exc_info=True)

            sleep = dt - (time.monotonic() - now)
            if sleep > 0:
                time.sleep(sleep)
