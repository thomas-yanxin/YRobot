"""Sound-source tracking and the single motion owner.

Design rules that keep motion lifelike:

* exactly one thread commands the robot (``Choreographer``, 50 Hz
  ``set_target``) — every pose is the sum of smooth, phase-shifted
  oscillators plus a critically damped turn toward the speaker, so nothing
  ever steps or fights;
* speech articulation is delegated to the SDK's ``enable_wobbling()``
  (daemon-side, PTS-synced to the actual speaker output) and body rotation
  to ``set_automatic_body_yaw(True)`` — both compose with our target pose;
* the XVF3800 DoA angle (0 = left, π/2 = front/back-ambiguous, π = right,
  head-relative) is sampled **only while our own VAD hears the user** —
  the firmware speech flag fires pre-AEC, i.e. also on the robot's own
  voice, which is why naive DoA feels deaf. A one-second circular window
  plus a dead-band turns raw readings into stable gaze targets.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

IDLE, LISTEN, SPEAK = "idle", "listen", "speak"


def head_yaw_of(pose: np.ndarray) -> float:
    """Extract world yaw from a 4x4 head pose."""
    return math.atan2(pose[1, 0], pose[0, 0])


def rpy_pose(roll: float, pitch: float, yaw: float, z: float) -> np.ndarray:
    """Build a 4x4 head pose from roll/pitch/yaw (rad) and a z offset (m)."""
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll), math.sin(roll), math.cos(pitch),
        math.sin(pitch), math.cos(yaw), math.sin(yaw),
    )  # fmt: skip
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    pose[2, 3] = z
    return pose


def doa_to_yaw_delta(angle: float) -> float:
    """Map an XVF3800 DoA angle to a head-relative yaw turn (assume front)."""
    return math.pi / 2 - angle


class SoundCompass(threading.Thread):
    """Polls DoA at 12 Hz and publishes stable world-frame gaze targets."""

    WINDOW_S = 1.0
    MIN_SAMPLES = 3
    DEADBAND_RAD = 0.12  # ≈7°: don't chase noise around the current gaze

    def __init__(
        self,
        media,
        current_head_yaw: Callable[[], float],
        user_active: Callable[[], bool],
        on_target: Callable[[float], None],
    ) -> None:
        super().__init__(name="yrobot-doa", daemon=True)
        self._media = media
        self._head_yaw = current_head_yaw
        self._user_active = user_active
        self._on_target = on_target
        self._halt = threading.Event()

    def close(self) -> None:
        self._halt.set()

    def run(self) -> None:
        samples: list[tuple[float, float]] = []  # (time, world yaw)
        while not self._halt.wait(1 / 12):
            if not self._user_active():
                samples.clear()
                continue
            reading = self._media.get_DoA()
            if reading is None:
                continue
            angle, _pre_aec_speech = reading  # flag intentionally unused
            now = time.monotonic()
            try:
                yaw = self._head_yaw() + doa_to_yaw_delta(angle)
            except Exception:  # daemon read hiccup: skip this sample
                continue
            samples.append((now, yaw))
            samples = [(t, y) for t, y in samples if now - t <= self.WINDOW_S]
            if len(samples) < self.MIN_SAMPLES:
                continue
            target = circular_mean([y for _, y in samples])
            if abs(_wrap(target - self._head_yaw())) > self.DEADBAND_RAD:
                self._on_target(target)


def circular_mean(angles: list[float]) -> float:
    return math.atan2(
        sum(math.sin(a) for a in angles) / len(angles),
        sum(math.cos(a) for a in angles) / len(angles),
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class GazeSpring:
    """Critically damped 2nd-order tracker: fast, smooth, never overshoots."""

    def __init__(self, omega: float = 6.0, max_vel: float = 3.0) -> None:
        self._omega = omega
        self._max_vel = max_vel
        self.pos = 0.0
        self.vel = 0.0
        self.target = 0.0

    def step(self, dt: float) -> float:
        acc = self._omega * self._omega * (self.target - self.pos) - 2 * self._omega * self.vel
        self.vel = max(-self._max_vel, min(self._max_vel, self.vel + acc * dt))
        self.pos += self.vel * dt
        return self.pos


class Choreographer(threading.Thread):
    """The one and only writer of robot pose, ticking at 50 Hz.

    Layers, all continuous in position and velocity:
      breathing (slow multi-phase sway) + idle saccades (held glances)
      + conversation posture (listening lean / speaking lift)
      + gaze spring toward the current speaker.
    """

    RATE_HZ = 50
    YAW_LIMIT = 2.4  # rad, stay inside the ±160° body envelope
    ANTENNA_NEUTRAL = 0.17

    def __init__(self, mini) -> None:
        super().__init__(name="yrobot-motion", daemon=True)
        self._mini = mini
        self._halt = threading.Event()
        self._mode = IDLE
        self._mode_blend = {IDLE: 1.0, LISTEN: 0.0, SPEAK: 0.0}
        self._gaze = GazeSpring()
        self._last_voice_at = -1e9
        self._saccade = (0.0, 0.0)
        self._next_saccade_at = 0.0
        self._antennas = np.array([self.ANTENNA_NEUTRAL, self.ANTENNA_NEUTRAL])

    # -- thread-safe inputs -------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_gaze_target(self, world_yaw: float, now: float | None = None) -> None:
        self._gaze.target = max(-self.YAW_LIMIT, min(self.YAW_LIMIT, _wrap(world_yaw)))
        self._last_voice_at = time.monotonic() if now is None else now

    def current_yaw(self) -> float:
        return self._gaze.pos

    def close(self) -> None:
        self._halt.set()

    # -- 50 Hz loop -----------------------------------------------------------

    def run(self) -> None:
        try:
            self._mini.set_automatic_body_yaw(True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("automatic body yaw unavailable: %s", exc)
        dt = 1 / self.RATE_HZ
        t0 = time.monotonic()
        next_tick = t0
        while not self._halt.is_set():
            now = time.monotonic()
            t = now - t0
            self._blend_modes(dt)
            pose, antennas = self._compose(t, now, dt)
            try:
                self._mini.set_target(head=pose, antennas=antennas)
            except Exception as exc:  # noqa: BLE001
                logger.debug("set_target dropped: %s", exc)
            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()  # never try to catch up with a jump

    def _blend_modes(self, dt: float) -> None:
        """Cross-fade posture weights (~250 ms) so mode flips never step."""
        rate = dt / 0.25
        for mode in self._mode_blend:
            goal = 1.0 if mode == self._mode else 0.0
            blend = self._mode_blend[mode]
            self._mode_blend[mode] = blend + max(-rate, min(rate, goal - blend))

    def _compose(self, t: float, now: float, dt: float) -> tuple[np.ndarray, list[float]]:
        idle, listen, speak = (self._mode_blend[m] for m in (IDLE, LISTEN, SPEAK))

        # Breathing — quieter while listening (attention), fuller when idle.
        amp = 0.5 + 0.5 * idle
        z = 0.004 * amp * math.sin(2 * math.pi * 0.16 * t)
        pitch = 0.020 * amp * math.sin(2 * math.pi * 0.16 * t + 0.9)
        roll = 0.012 * amp * math.sin(2 * math.pi * 0.11 * t + 2.1)

        # Idle saccades: brief held glances, walked back when engaged.
        if now >= self._next_saccade_at:
            self._next_saccade_at = now + random.uniform(4.0, 9.0)
            self._saccade = (random.uniform(-0.25, 0.25), random.uniform(-0.10, 0.12))
        sac_yaw, sac_pitch = (s * idle for s in self._saccade)

        # Conversation posture.
        pitch += 0.06 * listen - 0.03 * speak  # lean in to listen, lift to speak
        roll += 0.05 * listen * math.sin(2 * math.pi * 0.05 * t)  # slow curious tilt

        # After long silence, drift the gaze home.
        if now - self._last_voice_at > 45.0:
            self._gaze.target *= 0.999

        yaw = self._gaze.step(dt) + sac_yaw
        pose = rpy_pose(roll, pitch + sac_pitch, yaw, z)

        # Antennas: perked and still when listening, dancing when speaking.
        target = self.ANTENNA_NEUTRAL * (1.0 - 0.6 * listen)
        sway = 0.05 * idle * math.sin(2 * math.pi * 0.3 * t) + 0.10 * speak * math.sin(
            2 * math.pi * 1.4 * t
        )
        goal = np.array([target + sway, target - sway])
        self._antennas += (goal - self._antennas) * min(dt / 0.12, 1.0)
        return pose, [float(self._antennas[0]), float(self._antennas[1])]
