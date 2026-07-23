"""One-owner, fixed-rate expressive motion for Reachy Mini.

Only :class:`MotionController` calls ``set_target``. Perception and interaction
workers publish state; this worker composes and rate-limits the final atomic head
and antenna command at 50 Hz.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from reachy_mini.utils import create_head_pose
from scipy.spatial.transform import Rotation

from .perception import DoASnapshot
from .state import InteractionPhase

LOGGER = logging.getLogger(__name__)
MOTION_HZ = 50.0


class MotionRobot(Protocol):
    """Reachy Mini methods used by the motion writer."""

    def get_current_head_pose(self) -> npt.NDArray[np.float64]: ...

    def get_present_antenna_joint_positions(self) -> list[float]: ...

    def set_target(
        self,
        head: npt.NDArray[np.float64] | None = None,
        antennas: list[float] | npt.NDArray[np.float64] | None = None,
        body_yaw: float | None = None,
    ) -> None: ...


@dataclass(slots=True)
class RateLimitedAxis:
    """Second-order scalar limiter with explicit velocity/acceleration bounds."""

    position: float
    max_velocity: float
    max_acceleration: float
    response_seconds: float
    velocity: float = 0.0

    def __post_init__(self) -> None:
        if self.max_velocity <= 0 or self.max_acceleration <= 0 or self.response_seconds <= 0:
            raise ValueError("axis limits must be positive")

    def reset(self, position: float) -> None:
        self.position = float(position)
        self.velocity = 0.0

    def advance(self, target: float, dt: float) -> float:
        if dt <= 0:
            return self.position
        elapsed = min(dt, 0.05)
        error = float(target) - self.position
        desired_velocity = float(
            np.clip(
                error / self.response_seconds,
                -self.max_velocity,
                self.max_velocity,
            )
        )
        velocity_delta = float(
            np.clip(
                desired_velocity - self.velocity,
                -self.max_acceleration * elapsed,
                self.max_acceleration * elapsed,
            )
        )
        self.velocity = float(
            np.clip(
                self.velocity + velocity_delta,
                -self.max_velocity,
                self.max_velocity,
            )
        )
        step = self.velocity * elapsed
        if error != 0.0 and abs(step) >= abs(error) and step * error > 0.0:
            self.position = float(target)
            self.velocity = 0.0
        else:
            self.position += step
        return self.position


@dataclass(frozen=True, slots=True)
class MotionTarget:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    right_antenna: float
    left_antenna: float


class MotionPlanner:
    """Pure composition of attention, breathing, and interaction expression."""

    def plan(
        self,
        now: float,
        phase: InteractionPhase,
        doa_yaw: float,
    ) -> MotionTarget:
        phase = _coerce_phase(phase)
        attention_gain = {
            InteractionPhase.CONNECTING: 0.25,
            InteractionPhase.LISTENING: 1.0,
            InteractionPhase.SPEAKING: 0.78,
            InteractionPhase.INTERRUPTED: 1.0,
            InteractionPhase.STOPPED: 0.0,
        }[phase]
        activity = {
            InteractionPhase.CONNECTING: 0.18,
            InteractionPhase.LISTENING: 0.38,
            InteractionPhase.SPEAKING: 1.0,
            InteractionPhase.INTERRUPTED: 0.55,
            InteractionPhase.STOPPED: 0.0,
        }[phase]

        # Keep the base motion subtle. Reachy's daemon speech wobble is additive.
        breath = math.sin(math.tau * 0.12 * now)
        drift = math.sin(math.tau * 0.085 * now + 1.1)
        speaking_pulse = math.sin(math.tau * 0.72 * now + 0.4)
        attention = float(np.clip(doa_yaw, -math.radians(75), math.radians(75)))

        pitch_bias = (
            math.radians(-1.7) if phase is InteractionPhase.INTERRUPTED else math.radians(0.35)
        )
        yaw = attention_gain * attention + math.radians(0.35) * drift
        pitch = pitch_bias + math.radians(0.65 + 0.55 * activity) * breath
        roll = math.radians(0.35 + 0.45 * activity) * drift
        x = 0.0006 * activity * drift
        y = 0.00045 * activity * math.sin(math.tau * 0.10 * now + 2.0)
        z = (0.0007 + 0.0005 * activity) * breath

        antenna_base = math.radians(10.0)
        antenna_slow = math.radians(2.0 + 2.0 * activity) * math.sin(math.tau * 0.24 * now + 0.7)
        antenna_speech = (
            math.radians(3.0) * speaking_pulse if phase is InteractionPhase.SPEAKING else 0.0
        )
        right_antenna = -antenna_base - antenna_slow - antenna_speech
        return MotionTarget(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            right_antenna=right_antenna,
            left_antenna=-right_antenna,
        )


@dataclass(frozen=True, slots=True)
class MotionStats:
    commands: int
    overruns: int
    errors: int


class MotionController:
    """The sole 50 Hz owner of Reachy's atomic motion target.

    Motors must already be enabled before :meth:`start`. This class deliberately
    does not call ``enable_motors``, ``disable_motors``, ``goto_target``, or any
    media API.
    """

    def __init__(
        self,
        robot: MotionRobot,
        *,
        phase_source: Callable[[], InteractionPhase | object] | None = None,
        doa_source: Callable[[], DoASnapshot | float] | None = None,
        hz: float = MOTION_HZ,
    ) -> None:
        if not math.isclose(hz, MOTION_HZ):
            raise ValueError("Reachy Mini Wireless motion writer must run at 50 Hz")
        self._robot = robot
        self._phase_source = phase_source or (lambda: InteractionPhase.LISTENING)
        self._doa_source = doa_source or (lambda: 0.0)
        self._period = 1.0 / hz
        self._planner = MotionPlanner()
        self._axes = self._new_axes()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._head_yaw = 0.0
        self._commands = 0
        self._overruns = 0
        self._errors = 0

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="yrobot-motion",
            daemon=True,
        )
        self._thread.start()

    def wait_ready(self, timeout: float = 1.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.is_alive

    def stats(self) -> MotionStats:
        with self._stats_lock:
            return MotionStats(self._commands, self._overruns, self._errors)

    def _new_axes(self) -> dict[str, RateLimitedAxis]:
        angular = {
            "roll": RateLimitedAxis(0.0, math.radians(50), math.radians(190), 0.16),
            "pitch": RateLimitedAxis(0.0, math.radians(50), math.radians(190), 0.16),
            "yaw": RateLimitedAxis(0.0, math.radians(65), math.radians(220), 0.14),
        }
        translation = {name: RateLimitedAxis(0.0, 0.025, 0.10, 0.18) for name in ("x", "y", "z")}
        antennas = {
            name: RateLimitedAxis(0.0, 2.2, 8.0, 0.10) for name in ("right_antenna", "left_antenna")
        }
        return angular | translation | antennas

    def _initialize_from_robot(self) -> None:
        try:
            pose = np.asarray(self._robot.get_current_head_pose(), dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"unexpected head pose shape: {pose.shape}")
            roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
            self._axes["x"].reset(float(pose[0, 3]))
            self._axes["y"].reset(float(pose[1, 3]))
            self._axes["z"].reset(float(pose[2, 3]))
            self._axes["roll"].reset(float(roll))
            self._axes["pitch"].reset(float(pitch))
            self._axes["yaw"].reset(float(yaw))
            with self._command_lock:
                self._head_yaw = float(yaw)
        except Exception:
            LOGGER.warning("could not read initial head pose; using neutral", exc_info=True)

        try:
            antennas = self._robot.get_present_antenna_joint_positions()
            if len(antennas) != 2:
                raise ValueError("expected two antenna positions")
            self._axes["right_antenna"].reset(float(antennas[0]))
            self._axes["left_antenna"].reset(float(antennas[1]))
        except Exception:
            LOGGER.warning(
                "could not read initial antenna positions; using neutral",
                exc_info=True,
            )

    def _read_phase(self) -> InteractionPhase:
        try:
            value = self._phase_source()
            value = getattr(value, "phase", value)
            return _coerce_phase(value)
        except Exception:
            LOGGER.debug("phase source failed", exc_info=True)
            return InteractionPhase.LISTENING

    def _read_doa_yaw(self) -> float:
        try:
            value = self._doa_source()
            if isinstance(value, DoASnapshot):
                if value.active:
                    with self._command_lock:
                        current_yaw = self._head_yaw
                    confidence = float(np.clip(value.confidence, 0.0, 1.0))
                    if confidence < 0.5:
                        return current_yaw
                    error = (value.yaw_radians - current_yaw + math.pi) % math.tau - math.pi
                    yaw = current_yaw + confidence * error
                else:
                    yaw = value.yaw_radians
            else:
                yaw = float(getattr(value, "yaw_radians", value))
            return yaw if math.isfinite(yaw) else 0.0
        except Exception:
            LOGGER.debug("DoA source failed", exc_info=True)
            return 0.0

    def _advance(self, target: MotionTarget, dt: float) -> MotionTarget:
        return MotionTarget(
            **{
                name: self._axes[name].advance(getattr(target, name), dt)
                for name in MotionTarget.__dataclass_fields__
            }
        )

    def _axis_state(self) -> dict[str, tuple[float, float]]:
        return {name: (axis.position, axis.velocity) for name, axis in self._axes.items()}

    def _restore_axis_state(self, state: dict[str, tuple[float, float]]) -> None:
        for name, (position, velocity) in state.items():
            self._axes[name].position = position
            self._axes[name].velocity = velocity

    def _write(self, target: MotionTarget) -> None:
        head = create_head_pose(
            x=target.x,
            y=target.y,
            z=target.z,
            roll=target.roll,
            pitch=target.pitch,
            yaw=target.yaw,
            degrees=False,
        )
        self._robot.set_target(
            head=head,
            antennas=[target.right_antenna, target.left_antenna],
        )

    def _run(self) -> None:
        self._initialize_from_robot()
        self._ready.set()
        deadline = time.monotonic()
        previous = deadline - self._period
        last_warning = -math.inf

        while not self._stop.is_set():
            now = time.monotonic()
            if now < deadline:
                self._stop.wait(deadline - now)
                continue

            dt = max(0.001, now - previous)
            previous = now
            target = self._planner.plan(
                now,
                self._read_phase(),
                self._read_doa_yaw(),
            )
            previous_axes = self._axis_state()
            command = self._advance(target, dt)
            try:
                self._write(command)
                with self._command_lock:
                    self._head_yaw = command.yaw
                with self._stats_lock:
                    self._commands += 1
            except Exception:
                self._restore_axis_state(previous_axes)
                with self._stats_lock:
                    self._errors += 1
                if now - last_warning >= 2.0:
                    LOGGER.warning("motion target write failed", exc_info=True)
                    last_warning = now

            deadline += self._period
            current = time.monotonic()
            if deadline <= current:
                with self._stats_lock:
                    self._overruns += 1
                skipped = math.floor((current - deadline) / self._period) + 1
                deadline += skipped * self._period


def _coerce_phase(value: object) -> InteractionPhase:
    try:
        return value if isinstance(value, InteractionPhase) else InteractionPhase(value)
    except (TypeError, ValueError):
        return InteractionPhase.LISTENING
