"""Deterministic, single-owner motion for conversational Reachy Mini apps.

The controller deliberately keeps sensing and actuation on different workers:
``get_DoA`` may block in USB control I/O, while the motion worker must continue
to publish smooth targets at a fixed cadence.  Other application components
only select a :class:`MotionState`; they never write motor targets directly.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_CONTROL_HZ = 50.0
DEFAULT_DOA_HZ = 10.0
DEFAULT_DOA_HOLD_SECONDS = 6.0
DEFAULT_JOIN_TIMEOUT = 2.0
COMMAND_ERROR_LOG_INTERVAL = 5.0

# Reachy's documented hard limits are pitch/roll +/-40 degrees, body +/-160
# degrees and at most 65 degrees between head and body yaw.  The application
# limits are intentionally more conservative for close-range conversation.
HEAD_ROLL_LIMIT = math.radians(24.0)
HEAD_PITCH_LIMIT = math.radians(24.0)
HEAD_YAW_LIMIT = math.radians(120.0)
BODY_YAW_LIMIT = math.radians(150.0)
HEAD_BODY_YAW_DELTA_LIMIT = math.radians(60.0)
DOA_GAZE_LIMIT = math.radians(105.0)
HEAD_Z_LIMIT = 0.006
ANTENNA_LIMIT = math.radians(35.0)
HEAD_BODY_SOFT_DELTA = math.radians(42.0)


class MotionState(StrEnum):
    """Conversation states understood by :class:`MotionController`."""

    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"

    # Lower-case aliases make event payloads and ``MotionState.listening`` both
    # convenient while preserving conventional upper-case enum members.
    idle = IDLE
    listening = LISTENING
    speaking = SPEAKING
    interrupted = INTERRUPTED


@dataclass(frozen=True, slots=True)
class MotionLimits:
    """Rate limits used by every commanded axis."""

    head_angular_speed: float = math.radians(45.0)
    head_angular_acceleration: float = math.radians(145.0)
    head_translation_speed: float = 0.018
    head_translation_acceleration: float = 0.065
    antenna_speed: float = math.radians(90.0)
    antenna_acceleration: float = math.radians(280.0)
    body_speed: float = math.radians(34.0)
    body_acceleration: float = math.radians(95.0)


DEFAULT_LIMITS = MotionLimits()


@dataclass(frozen=True, slots=True)
class MotionCommand:
    """Last command snapshot, useful for diagnostics and latency tests."""

    timestamp: float
    roll: float
    pitch: float
    yaw: float
    z: float
    antennas: tuple[float, float]
    body_yaw: float
    state: MotionState


@dataclass(frozen=True, slots=True)
class _Posture:
    roll: float
    pitch: float
    z: float
    antennas: tuple[float, float]


_POSTURES: dict[MotionState, _Posture] = {
    MotionState.IDLE: _Posture(
        roll=0.0,
        pitch=math.radians(-3.2),
        z=0.0,
        antennas=(math.radians(-10.0), math.radians(10.0)),
    ),
    MotionState.LISTENING: _Posture(
        roll=0.0,
        pitch=math.radians(-2.4),
        z=0.0008,
        antennas=(math.radians(-14.0), math.radians(14.0)),
    ),
    MotionState.SPEAKING: _Posture(
        roll=0.0,
        pitch=math.radians(-2.8),
        z=0.0004,
        antennas=(math.radians(-16.0), math.radians(16.0)),
    ),
    MotionState.INTERRUPTED: _Posture(
        roll=0.0,
        pitch=math.radians(-0.4),
        z=0.0017,
        antennas=(math.radians(-9.0), math.radians(9.0)),
    ),
}


@dataclass(frozen=True, slots=True)
class _DoASample:
    sequence: int
    angle: float
    head_world_yaw: float
    speech_detected: bool
    captured_at: float


@dataclass(slots=True)
class _LimitedCriticallyDampedAxis:
    """Second-order target follower with explicit velocity/acceleration caps."""

    position: float = 0.0
    velocity: float = 0.0

    def step(
        self,
        target: float,
        dt: float,
        *,
        omega: float,
        max_speed: float,
        max_acceleration: float,
        lower: float,
        upper: float,
    ) -> float:
        target = _clamp(target, lower, upper)
        dt = max(1e-4, dt)
        # x'' + 2*w*x' + w^2*(x-target) = 0 is critically damped.
        acceleration = omega * omega * (target - self.position) - 2.0 * omega * self.velocity
        acceleration = _clamp(acceleration, -max_acceleration, max_acceleration)
        previous_velocity = self.velocity
        self.velocity = _clamp(
            previous_velocity + acceleration * dt,
            -max_speed,
            max_speed,
        )
        # Trapezoidal integration keeps the finite-difference acceleration of
        # emitted positions aligned with the explicit acceleration bound.
        self.position += 0.5 * (previous_velocity + self.velocity) * dt

        if self.position < lower:
            self.position = lower
            self.velocity = max(0.0, self.velocity)
        elif self.position > upper:
            self.position = upper
            self.velocity = min(0.0, self.velocity)
        return self.position


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def doa_angle_to_yaw(angle: float, *, limit: float = DOA_GAZE_LIMIT) -> float:
    """Convert Reachy's DoA convention to a safe head-relative gaze yaw.

    XVF DoA uses 0 at the robot's left and pi/2 in front.  Reachy's positive
    head yaw turns left, hence ``yaw = pi/2 - angle``.
    """

    if not math.isfinite(angle):
        raise ValueError("DoA angle must be finite")
    if not 0.0 < limit <= math.pi:
        raise ValueError("DoA yaw limit must be in (0, pi]")
    return _clamp(_wrap_pi(math.pi / 2.0 - angle), -limit, limit)


def doa_angle_to_world_yaw(
    angle: float,
    head_world_yaw: float,
    *,
    limit: float = DOA_GAZE_LIMIT,
) -> float:
    """Transform one head-relative DoA bearing into the commanded world frame."""

    if not math.isfinite(head_world_yaw):
        raise ValueError("head world yaw must be finite")
    relative_yaw = doa_angle_to_yaw(angle, limit=math.pi)
    return _clamp(_wrap_pi(head_world_yaw + relative_yaw), -limit, limit)


def _default_pose_factory(**kwargs: float | bool) -> np.ndarray:
    # Delayed import keeps pure motion tests independent from the SDK runtime.
    from reachy_mini.utils import create_head_pose

    return np.asarray(create_head_pose(**kwargs), dtype=np.float64)


class MotionController:
    """Run bounded conversational motion with one robot-command owner.

    Parameters other than ``mini`` are primarily tuning and test seams.  In
    production, SDK wobbling supplies playback-synchronised speaking motion;
    this controller keeps the underlying posture quiet while speaking.
    """

    def __init__(
        self,
        mini: object,
        *,
        control_hz: float = DEFAULT_CONTROL_HZ,
        doa_hz: float = DEFAULT_DOA_HZ,
        doa_hold_seconds: float = DEFAULT_DOA_HOLD_SECONDS,
        limits: MotionLimits = DEFAULT_LIMITS,
        pose_factory: Callable[..., np.ndarray] | None = None,
        enable_wobbling: bool = True,
        neutral_transitions: bool = False,
        neutral_duration: float = 0.7,
    ) -> None:
        if control_hz <= 0.0 or doa_hz <= 0.0:
            raise ValueError("control_hz and doa_hz must be positive")
        if doa_hold_seconds < 0.0:
            raise ValueError("doa_hold_seconds must not be negative")
        if neutral_duration <= 0.0:
            raise ValueError("neutral_duration must be positive")

        self.mini = mini
        self.control_hz = float(control_hz)
        self.doa_hz = float(doa_hz)
        self.doa_hold_seconds = float(doa_hold_seconds)
        self.limits = limits
        self._pose_factory = pose_factory or _default_pose_factory
        self._enable_wobbling = enable_wobbling
        self._neutral_transitions = neutral_transitions
        self._neutral_duration = neutral_duration

        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = MotionState.IDLE
        self._state_entered_at = time.monotonic()
        self._doa_lock = threading.Lock()
        self._latest_doa: _DoASample | None = None
        self._doa_sequence = 0
        self._snapshot_lock = threading.Lock()
        self._last_command: MotionCommand | None = None

        self._motion_thread: threading.Thread | None = None
        self._doa_thread: threading.Thread | None = None
        self._wobbling_enabled = False
        self._started_at = time.monotonic()
        self._reset_axes()

    @property
    def state(self) -> MotionState:
        with self._state_lock:
            return self._state

    @property
    def last_command(self) -> MotionCommand | None:
        with self._snapshot_lock:
            return self._last_command

    @property
    def is_running(self) -> bool:
        thread = self._motion_thread
        return thread is not None and thread.is_alive() and not self._stop_event.is_set()

    @property
    def active_worker_names(self) -> tuple[str, ...]:
        return tuple(
            thread.name
            for thread in (self._motion_thread, self._doa_thread)
            if thread is not None and thread.is_alive()
        )

    def set_state(self, state: MotionState | str) -> None:
        """Select a posture; the motion worker applies it on its next tick."""

        selected = state if isinstance(state, MotionState) else MotionState(state)
        with self._state_lock:
            if selected != self._state:
                self._state = selected
                self._state_entered_at = time.monotonic()

    def start(self) -> None:
        """Start the isolated DoA sensor and single-owner motion workers."""

        with self._lifecycle_lock:
            if self.is_running:
                return
            if any(
                thread is not None and thread.is_alive()
                for thread in (self._motion_thread, self._doa_thread)
            ):
                raise RuntimeError("a previous motion worker is still stopping")

            self._stop_event.clear()
            self._started_at = time.monotonic()
            self._reset_axes()
            with self._doa_lock:
                self._latest_doa = None

            if self._neutral_transitions:
                self._goto_neutral("startup")
            self._start_wobbling()

            self._motion_thread = threading.Thread(
                target=self._motion_loop,
                name="yrobot-motion",
                daemon=True,
            )
            self._doa_thread = threading.Thread(
                target=self._doa_loop,
                name="yrobot-doa",
                daemon=True,
            )
            # Start actuation first: a blocking first USB read must not delay the
            # first safe target or the control cadence.
            self._motion_thread.start()
            self._doa_thread.start()

    def stop(self, *, join_timeout: float = DEFAULT_JOIN_TIMEOUT) -> None:
        """Stop workers, SDK wobbling and optionally return via min-jerk."""

        if join_timeout < 0.0:
            raise ValueError("join_timeout must not be negative")
        with self._lifecycle_lock:
            motion_thread = self._motion_thread
            doa_thread = self._doa_thread
            if motion_thread is None and doa_thread is None:
                self._stop_wobbling()
                return
            self._stop_event.set()

        deadline = time.monotonic() + join_timeout
        if motion_thread is not None:
            motion_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if doa_thread is not None:
            doa_thread.join(timeout=max(0.0, deadline - time.monotonic()))

        motion_alive = motion_thread is not None and motion_thread.is_alive()
        doa_alive = doa_thread is not None and doa_thread.is_alive()
        if motion_alive:
            log.warning("Motion worker did not stop within %.2f s", join_timeout)
        if doa_alive:
            # Python cannot cancel a USB control call already executing.  The
            # daemon thread is isolated from actuation and will exit as soon as
            # that SDK call returns.
            log.warning("DoA worker is still waiting for the SDK after %.2f s", join_timeout)

        self._stop_wobbling()
        if self._neutral_transitions and not motion_alive:
            self._goto_neutral("shutdown")

        with self._lifecycle_lock:
            self._motion_thread = motion_thread if motion_alive else None
            self._doa_thread = doa_thread if doa_alive else None

    def _reset_axes(self) -> None:
        self._roll_axis = _LimitedCriticallyDampedAxis()
        self._pitch_axis = _LimitedCriticallyDampedAxis()
        self._yaw_axis = _LimitedCriticallyDampedAxis()
        self._z_axis = _LimitedCriticallyDampedAxis()
        self._right_antenna_axis = _LimitedCriticallyDampedAxis()
        self._left_antenna_axis = _LimitedCriticallyDampedAxis()
        self._body_axis = _LimitedCriticallyDampedAxis()

    def _start_wobbling(self) -> None:
        if not self._enable_wobbling or self._wobbling_enabled:
            return
        enable = getattr(self.mini, "enable_wobbling", None)
        if not callable(enable):
            log.debug("Reachy SDK wobbling is unavailable")
            return
        try:
            enable()
        except Exception as exc:
            log.warning("Could not enable SDK speech wobbling: %s", exc)
        else:
            self._wobbling_enabled = True

    def _stop_wobbling(self) -> None:
        if not self._wobbling_enabled:
            return
        disable = getattr(self.mini, "disable_wobbling", None)
        try:
            if callable(disable):
                disable()
        except Exception as exc:
            log.debug("Could not disable SDK speech wobbling: %s", exc)
        finally:
            self._wobbling_enabled = False

    def _goto_neutral(self, phase: str) -> None:
        goto = getattr(self.mini, "goto_target", None)
        if not callable(goto):
            log.debug("Skipping %s neutral transition; goto_target is unavailable", phase)
            return
        pose = self._pose_factory(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            degrees=False,
        )
        try:
            goto(
                head=pose,
                antennas=np.radians((-10.0, 10.0)),
                body_yaw=0.0,
                duration=self._neutral_duration,
                method="minjerk",
            )
        except Exception as exc:
            log.warning("Could not complete %s neutral transition: %s", phase, exc)

    def _doa_loop(self) -> None:
        """Read potentially blocking USB DoA without sharing the motion worker."""

        getter = getattr(getattr(self.mini, "media", None), "get_DoA", None)
        if not callable(getter):
            log.debug("Reachy DoA is unavailable")
            return

        period = 1.0 / self.doa_hz
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            try:
                result = getter()
                captured_at = time.monotonic()
                if result is not None:
                    angle, speech_detected = result
                    angle = float(angle)
                    if math.isfinite(angle):
                        with self._snapshot_lock:
                            command = self._last_command
                        head_world_yaw = command.yaw if command is not None else 0.0
                        with self._doa_lock:
                            self._doa_sequence += 1
                            self._latest_doa = _DoASample(
                                sequence=self._doa_sequence,
                                angle=angle,
                                head_world_yaw=head_world_yaw,
                                speech_detected=bool(speech_detected),
                                captured_at=captured_at,
                            )
            except Exception as exc:
                log.debug("DoA sample failed: %s", exc)

            next_deadline += period
            now = time.monotonic()
            if next_deadline <= now:
                next_deadline = now + period
            self._stop_event.wait(max(0.0, next_deadline - now))

    def _motion_loop(self) -> None:
        """Only worker allowed to call ``mini.set_target``."""

        period = 1.0 / self.control_hz
        now = time.monotonic()
        last_command_at = now - period
        next_deadline = now
        last_doa_sequence = 0
        attention_yaw = 0.0
        last_confident_doa_at = -math.inf
        last_command_error = -math.inf

        while not self._stop_event.is_set():
            now = time.monotonic()
            with self._state_lock:
                state = self._state

            with self._doa_lock:
                doa = self._latest_doa
            if doa is not None and doa.sequence != last_doa_sequence:
                last_doa_sequence = doa.sequence
                # XVF's speech bit includes Reachy's own speaker.  Ignore new
                # bearings while speaking and let the last valid user target
                # decay instead of chasing self-voice.
                if doa.speech_detected and state not in {
                    MotionState.SPEAKING,
                    MotionState.INTERRUPTED,
                }:
                    attention_yaw = doa_angle_to_world_yaw(
                        doa.angle,
                        doa.head_world_yaw,
                    )
                    last_confident_doa_at = doa.captured_at

            keep_attention = state is MotionState.SPEAKING or (
                now - last_confident_doa_at <= self.doa_hold_seconds
            )
            gaze_target = attention_yaw if keep_attention else 0.0
            posture = _POSTURES[state]
            elapsed = now - self._started_at
            roll_target, pitch_target, z_target, antenna_targets = self._animated_posture(
                posture,
                state,
                elapsed,
            )

            body_target = math.copysign(
                max(0.0, abs(gaze_target) - HEAD_BODY_SOFT_DELTA),
                gaze_target,
            )
            body_target = _clamp(body_target, -BODY_YAW_LIMIT, BODY_YAW_LIMIT)
            head_yaw_target = _clamp(gaze_target, -HEAD_YAW_LIMIT, HEAD_YAW_LIMIT)
            # Interrupted is a quick, legible yield but still uses exactly the
            # same hard rate limits as every other state.
            omega = 9.0 if state is MotionState.INTERRUPTED else 6.2

            # Integrate against the most recent command issue time, not the
            # beginning of the previous loop. A scheduler pause between pose
            # calculation and set_target must not be followed by a large step
            # only a few milliseconds after the delayed command.
            integration_at = time.monotonic()
            dt = min(0.08, max(1e-4, integration_at - last_command_at))
            controlled_axes = (
                self._roll_axis,
                self._pitch_axis,
                self._yaw_axis,
                self._z_axis,
                self._right_antenna_axis,
                self._left_antenna_axis,
                self._body_axis,
            )
            previous_axis_states = tuple((axis.position, axis.velocity) for axis in controlled_axes)
            previous_positions = tuple(position for position, _ in previous_axis_states)

            roll = self._roll_axis.step(
                roll_target,
                dt,
                omega=omega,
                max_speed=self.limits.head_angular_speed,
                max_acceleration=self.limits.head_angular_acceleration,
                lower=-HEAD_ROLL_LIMIT,
                upper=HEAD_ROLL_LIMIT,
            )
            pitch = self._pitch_axis.step(
                pitch_target,
                dt,
                omega=omega,
                max_speed=self.limits.head_angular_speed,
                max_acceleration=self.limits.head_angular_acceleration,
                lower=-HEAD_PITCH_LIMIT,
                upper=HEAD_PITCH_LIMIT,
            )
            body_yaw = self._body_axis.step(
                body_target,
                dt,
                omega=5.0,
                max_speed=self.limits.body_speed,
                max_acceleration=self.limits.body_acceleration,
                lower=-BODY_YAW_LIMIT,
                upper=BODY_YAW_LIMIT,
            )
            yaw = self._yaw_axis.step(
                head_yaw_target,
                dt,
                omega=5.8,
                max_speed=self.limits.head_angular_speed,
                max_acceleration=self.limits.head_angular_acceleration,
                lower=-HEAD_YAW_LIMIT,
                upper=HEAD_YAW_LIMIT,
            )
            # Enforce the coupled safety bound on the command itself; choosing a
            # 42-degree soft target normally keeps this clamp inactive.
            safe_yaw = _clamp(
                yaw,
                body_yaw - HEAD_BODY_YAW_DELTA_LIMIT,
                body_yaw + HEAD_BODY_YAW_DELTA_LIMIT,
            )
            if safe_yaw != yaw:
                self._yaw_axis.position = safe_yaw
                self._yaw_axis.velocity = _clamp(
                    self._yaw_axis.velocity,
                    -self.limits.body_speed,
                    self.limits.body_speed,
                )
                yaw = safe_yaw

            z = self._z_axis.step(
                z_target,
                dt,
                omega=omega,
                max_speed=self.limits.head_translation_speed,
                max_acceleration=self.limits.head_translation_acceleration,
                lower=-HEAD_Z_LIMIT,
                upper=HEAD_Z_LIMIT,
            )
            right_antenna = self._right_antenna_axis.step(
                antenna_targets[0],
                dt,
                omega=omega,
                max_speed=self.limits.antenna_speed,
                max_acceleration=self.limits.antenna_acceleration,
                lower=-ANTENNA_LIMIT,
                upper=ANTENNA_LIMIT,
            )
            left_antenna = self._left_antenna_axis.step(
                antenna_targets[1],
                dt,
                omega=omega,
                max_speed=self.limits.antenna_speed,
                max_acceleration=self.limits.antenna_acceleration,
                lower=-ANTENNA_LIMIT,
                upper=ANTENNA_LIMIT,
            )

            head = self._pose_factory(
                x=0.0,
                y=0.0,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                degrees=False,
            )
            antennas = np.asarray((right_antenna, left_antenna), dtype=np.float64)
            try:
                # This is intentionally the only set_target call in the module.
                self.mini.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            except Exception as exc:
                # A failed SDK call must not advance the model of the physical
                # robot. Otherwise several failures accumulate into one large
                # recovery jump when transport resumes.
                for axis, (position, velocity) in zip(
                    controlled_axes,
                    previous_axis_states,
                    strict=True,
                ):
                    axis.position = position
                    axis.velocity = velocity
                if now - last_command_error >= COMMAND_ERROR_LOG_INTERVAL:
                    log.warning("Motion command failed: %s", exc)
                    last_command_error = now
            else:
                command_sent_at = time.monotonic()
                realized_dt = max(1e-4, command_sent_at - last_command_at)
                axes_and_commands = (
                    (self._roll_axis, roll, self.limits.head_angular_speed),
                    (self._pitch_axis, pitch, self.limits.head_angular_speed),
                    (self._yaw_axis, yaw, self.limits.head_angular_speed),
                    (self._z_axis, z, self.limits.head_translation_speed),
                    (self._right_antenna_axis, right_antenna, self.limits.antenna_speed),
                    (self._left_antenna_axis, left_antenna, self.limits.antenna_speed),
                    (self._body_axis, body_yaw, self.limits.body_speed),
                )
                # Reconcile the state velocity with the command timing the
                # daemon actually observed. If this worker was descheduled
                # after integration, the delivered velocity is lower; using
                # that realized value prevents a sharp catch-up next tick.
                for previous, (axis, commanded, speed_limit) in zip(
                    previous_positions,
                    axes_and_commands,
                    strict=True,
                ):
                    axis.velocity = _clamp(
                        (commanded - previous) / realized_dt,
                        -speed_limit,
                        speed_limit,
                    )
                with self._snapshot_lock:
                    self._last_command = MotionCommand(
                        timestamp=command_sent_at,
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        z=z,
                        antennas=(right_antenna, left_antenna),
                        body_yaw=body_yaw,
                        state=state,
                    )
            finally:
                # Treat an attempted SDK command as the pacing boundary. An
                # exception may occur after bytes reached the daemon, so a
                # conservative small next step is safer than catching up.
                last_command_at = time.monotonic()

            next_deadline += period
            after_command = time.monotonic()
            if next_deadline <= after_command:
                missed = math.floor((after_command - next_deadline) / period) + 1
                next_deadline += missed * period
            self._stop_event.wait(max(0.0, next_deadline - after_command))

    @staticmethod
    def _animated_posture(
        posture: _Posture,
        state: MotionState,
        elapsed: float,
    ) -> tuple[float, float, float, tuple[float, float]]:
        """Return continuous, deterministic targets with no random pose jumps."""

        roll = posture.roll
        pitch = posture.pitch
        z = posture.z
        right, left = posture.antennas

        if state is MotionState.IDLE:
            breath = math.sin(2.0 * math.pi * 0.11 * elapsed)
            secondary = math.sin(2.0 * math.pi * 0.073 * elapsed + 0.8)
            z += 0.00055 * (0.72 * breath + 0.28 * secondary)
            pitch += math.radians(0.24) * breath
            roll += math.radians(0.18) * secondary
            antenna_sway = math.radians(0.8) * math.sin(2.0 * math.pi * 0.16 * elapsed + 0.4)
            right -= antenna_sway
            left += antenna_sway
        elif state is MotionState.LISTENING:
            attention = math.sin(2.0 * math.pi * 0.09 * elapsed + 0.5)
            z += 0.00025 * attention
            pitch += math.radians(0.12) * attention
        # Speaking motion is deliberately absent here. SDK wobbling is driven
        # from actual playout PTS and composes its offsets on the daemon.

        return roll, pitch, z, (right, left)


__all__ = [
    "ANTENNA_LIMIT",
    "BODY_YAW_LIMIT",
    "DEFAULT_CONTROL_HZ",
    "DEFAULT_DOA_HZ",
    "DEFAULT_LIMITS",
    "DOA_GAZE_LIMIT",
    "HEAD_BODY_YAW_DELTA_LIMIT",
    "HEAD_PITCH_LIMIT",
    "HEAD_ROLL_LIMIT",
    "HEAD_YAW_LIMIT",
    "HEAD_Z_LIMIT",
    "MotionCommand",
    "MotionController",
    "MotionLimits",
    "MotionState",
    "doa_angle_to_world_yaw",
    "doa_angle_to_yaw",
]
