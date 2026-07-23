import math
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from yrobot.motion import (
    ANTENNA_LIMIT,
    BODY_YAW_LIMIT,
    DEFAULT_LIMITS,
    HEAD_BODY_YAW_DELTA_LIMIT,
    HEAD_PITCH_LIMIT,
    HEAD_ROLL_LIMIT,
    HEAD_YAW_LIMIT,
    HEAD_Z_LIMIT,
    MotionController,
    MotionState,
    _LimitedCriticallyDampedAxis,
    doa_angle_to_world_yaw,
    doa_angle_to_yaw,
)


def _test_pose_factory(**values: float | bool) -> np.ndarray:
    """Encode commanded axes in a matrix without importing Reachy's SDK."""

    pose = np.eye(4, dtype=np.float64)
    pose[0, 0] = float(values["roll"])
    pose[0, 1] = float(values["pitch"])
    pose[0, 2] = float(values["yaw"])
    pose[2, 3] = float(values["z"])
    return pose


class _FakeMedia:
    def __init__(self, provider: Callable[[], tuple[float, bool] | None]) -> None:
        self._provider = provider
        self.thread_names: list[str] = []

    def get_DoA(self) -> tuple[float, bool] | None:
        self.thread_names.append(threading.current_thread().name)
        return self._provider()


class _FakeMini:
    def __init__(self, provider: Callable[[], tuple[float, bool] | None]) -> None:
        self.media = _FakeMedia(provider)
        self.commands: list[tuple[float, str, np.ndarray, np.ndarray, float]] = []
        self.command_event = threading.Event()
        self.wobbling_enabled = 0
        self.wobbling_disabled = 0
        self._command_lock = threading.Lock()

    def set_target(
        self,
        *,
        head: np.ndarray,
        antennas: np.ndarray,
        body_yaw: float,
    ) -> None:
        with self._command_lock:
            self.commands.append(
                (
                    time.monotonic(),
                    threading.current_thread().name,
                    head.copy(),
                    antennas.copy(),
                    body_yaw,
                )
            )
        self.command_event.set()

    def enable_wobbling(self) -> None:
        self.wobbling_enabled += 1

    def disable_wobbling(self) -> None:
        self.wobbling_disabled += 1


def _command_axes(
    command: tuple[float, str, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    _, _, head, antennas, body_yaw = command
    return np.asarray(
        (
            head[0, 0],
            head[0, 1],
            head[0, 2],
            head[2, 3],
            antennas[0],
            antennas[1],
            body_yaw,
        ),
        dtype=np.float64,
    )


def test_doa_mapping_matches_reachy_coordinates_and_clamps() -> None:
    assert doa_angle_to_yaw(0.0) == pytest.approx(math.pi / 2.0)
    assert doa_angle_to_yaw(math.pi / 2.0) == pytest.approx(0.0)
    assert doa_angle_to_yaw(math.pi) == pytest.approx(-math.pi / 2.0)
    assert doa_angle_to_yaw(-0.5) == pytest.approx(math.radians(105.0))
    assert doa_angle_to_world_yaw(
        math.pi / 2.0,
        math.radians(35.0),
    ) == pytest.approx(math.radians(35.0))
    assert doa_angle_to_world_yaw(
        0.0,
        math.radians(30.0),
    ) == pytest.approx(math.radians(105.0))
    with pytest.raises(ValueError, match="finite"):
        doa_angle_to_yaw(float("nan"))


def test_blocked_doa_read_does_not_disturb_motion_cadence() -> None:
    def slow_doa() -> tuple[float, bool]:
        time.sleep(0.28)
        return math.pi / 2.0, False

    mini = _FakeMini(slow_doa)
    controller = MotionController(
        mini,
        control_hz=50.0,
        doa_hz=10.0,
        pose_factory=_test_pose_factory,
        enable_wobbling=False,
        neutral_transitions=False,
    )

    controller.start()
    try:
        assert mini.command_event.wait(0.5)
        time.sleep(0.34)
    finally:
        controller.stop(join_timeout=1.0)

    timestamps = np.asarray([command[0] for command in mini.commands])
    assert timestamps.size >= 14
    # A USB read blocks for 280 ms; the isolated actuator should still have no
    # correspondingly large hole. Keep a little CI scheduling headroom.
    assert np.max(np.diff(timestamps)) < 0.09
    assert set(command[1] for command in mini.commands) == {"yrobot-motion"}
    assert mini.media.thread_names
    assert set(mini.media.thread_names) == {"yrobot-doa"}


def test_motion_commands_have_one_owner_and_obey_rate_and_safety_limits() -> None:
    mini = _FakeMini(lambda: (0.0, True))  # far left: exercise body/head split
    controller = MotionController(
        mini,
        control_hz=50.0,
        doa_hz=20.0,
        doa_hold_seconds=0.3,
        pose_factory=_test_pose_factory,
        enable_wobbling=False,
        neutral_transitions=False,
    )

    controller.set_state(MotionState.LISTENING)
    controller.start()
    try:
        assert mini.command_event.wait(0.5)
        time.sleep(0.55)
        controller.set_state(MotionState.INTERRUPTED)
        time.sleep(0.45)
    finally:
        controller.stop(join_timeout=1.0)

    assert len(mini.commands) >= 35
    assert set(command[1] for command in mini.commands) == {"yrobot-motion"}
    timestamps = np.asarray([command[0] for command in mini.commands])
    values = np.stack([_command_axes(command) for command in mini.commands])

    roll, pitch, yaw, z, right, left, body = values.T
    assert np.max(np.abs(roll)) <= HEAD_ROLL_LIMIT + 1e-9
    assert np.max(np.abs(pitch)) <= HEAD_PITCH_LIMIT + 1e-9
    assert np.max(np.abs(yaw)) <= HEAD_YAW_LIMIT + 1e-9
    assert np.max(np.abs(z)) <= HEAD_Z_LIMIT + 1e-9
    assert np.max(np.abs(right)) <= ANTENNA_LIMIT + 1e-9
    assert np.max(np.abs(left)) <= ANTENNA_LIMIT + 1e-9
    assert np.max(np.abs(body)) <= BODY_YAW_LIMIT + 1e-9
    assert np.max(np.abs(yaw - body)) <= HEAD_BODY_YAW_DELTA_LIMIT + 1e-9

    dt = np.diff(timestamps)
    speeds = np.diff(values, axis=0) / dt[:, None]
    speed_limits = np.asarray(
        (
            DEFAULT_LIMITS.head_angular_speed,
            DEFAULT_LIMITS.head_angular_speed,
            DEFAULT_LIMITS.head_angular_speed,
            DEFAULT_LIMITS.head_translation_speed,
            DEFAULT_LIMITS.antenna_speed,
            DEFAULT_LIMITS.antenna_speed,
            DEFAULT_LIMITS.body_speed,
        )
    )
    assert np.all(np.max(np.abs(speeds), axis=0) <= speed_limits * 1.25 + 1e-6)


def test_axis_enforces_speed_and_acceleration_for_jittered_ticks() -> None:
    axis = _LimitedCriticallyDampedAxis()
    max_speed = 0.8
    max_acceleration = 1.7
    previous_velocity = axis.velocity

    for index, dt in enumerate((0.020, 0.011, 0.037, 0.016, 0.055, 0.020) * 20):
        target = 1.0 if index < 65 else -1.0
        position = axis.step(
            target,
            dt,
            omega=8.0,
            max_speed=max_speed,
            max_acceleration=max_acceleration,
            lower=-1.2,
            upper=1.2,
        )
        assert abs(axis.velocity) <= max_speed + 1e-12
        assert abs(axis.velocity - previous_velocity) <= max_acceleration * dt + 1e-12
        assert -1.2 <= position <= 1.2
        previous_velocity = axis.velocity


def test_failed_sdk_commands_do_not_accumulate_into_recovery_jump() -> None:
    class FlakyMini(_FakeMini):
        def __init__(self) -> None:
            super().__init__(lambda: (0.0, True))
            self.failures_remaining = 12

        def set_target(
            self,
            *,
            head: np.ndarray,
            antennas: np.ndarray,
            body_yaw: float,
        ) -> None:
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise ConnectionError("transient daemon transport failure")
            super().set_target(head=head, antennas=antennas, body_yaw=body_yaw)

    mini = FlakyMini()
    controller = MotionController(
        mini,
        control_hz=50.0,
        doa_hz=20.0,
        pose_factory=_test_pose_factory,
        enable_wobbling=False,
        neutral_transitions=False,
    )
    controller.set_state(MotionState.LISTENING)
    controller.start()
    try:
        assert mini.command_event.wait(1.0)
    finally:
        controller.stop(join_timeout=1.0)

    first = np.abs(_command_axes(mini.commands[0]))
    one_tick_limits = (
        np.asarray(
            (
                DEFAULT_LIMITS.head_angular_speed,
                DEFAULT_LIMITS.head_angular_speed,
                DEFAULT_LIMITS.head_angular_speed,
                DEFAULT_LIMITS.head_translation_speed,
                DEFAULT_LIMITS.antenna_speed,
                DEFAULT_LIMITS.antenna_speed,
                DEFAULT_LIMITS.body_speed,
            )
        )
        * 0.04
    )
    assert np.all(first <= one_tick_limits + 1e-6)


def test_low_confidence_doa_expires_and_returns_toward_neutral() -> None:
    confident = threading.Event()
    confident.set()

    def doa() -> tuple[float, bool]:
        return 0.0, confident.is_set()

    mini = _FakeMini(doa)
    controller = MotionController(
        mini,
        control_hz=50.0,
        doa_hz=20.0,
        doa_hold_seconds=0.15,
        pose_factory=_test_pose_factory,
        enable_wobbling=False,
        neutral_transitions=False,
    )
    controller.set_state("listening")
    controller.start()
    try:
        time.sleep(0.65)
        peak = abs(controller.last_command.yaw) if controller.last_command else 0.0
        confident.clear()
        deadline = time.monotonic() + 3.0
        final = math.inf
        while time.monotonic() < deadline:
            final = abs(controller.last_command.yaw) if controller.last_command else math.inf
            if final < peak * 0.45:
                break
            time.sleep(0.02)
    finally:
        controller.stop(join_timeout=1.0)

    assert peak > math.radians(8.0)
    assert final < peak * 0.45


def test_stop_joins_workers_and_sdk_wobbling_owns_speaking_motion() -> None:
    mini = _FakeMini(lambda: (math.pi / 2.0, False))
    controller = MotionController(
        mini,
        pose_factory=_test_pose_factory,
        enable_wobbling=True,
        neutral_transitions=False,
    )

    controller.set_state(MotionState.SPEAKING)
    controller.start()
    assert mini.command_event.wait(0.5)
    controller.stop(join_timeout=1.0)

    assert mini.wobbling_enabled == 1
    assert mini.wobbling_disabled == 1
    assert not controller.is_running
    assert controller.active_worker_names == ()
    assert not any(
        thread.name in {"yrobot-motion", "yrobot-doa"} and thread.is_alive()
        for thread in threading.enumerate()
    )
