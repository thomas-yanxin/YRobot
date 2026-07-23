from __future__ import annotations

import math
import threading
import time

import numpy as np

from yrobot.motion import MotionController, MotionPlanner, RateLimitedAxis
from yrobot.perception import DoASnapshot
from yrobot.state import InteractionPhase


def test_axis_limits_velocity_and_acceleration_without_overshoot() -> None:
    axis = RateLimitedAxis(
        position=0.0,
        max_velocity=1.0,
        max_acceleration=2.0,
        response_seconds=0.1,
    )
    positions = [axis.advance(1.0, 0.02) for _ in range(100)]

    assert positions == sorted(positions)
    assert positions[-1] == 1.0
    assert all(0.0 <= position <= 1.0 for position in positions)


def test_motion_planner_is_subtle_and_phase_dependent() -> None:
    planner = MotionPlanner()
    listening = planner.plan(2.0, InteractionPhase.LISTENING, math.radians(15))
    speaking = planner.plan(2.0, InteractionPhase.SPEAKING, math.radians(15))
    interrupted = planner.plan(2.0, InteractionPhase.INTERRUPTED, math.radians(15))

    assert abs(listening.yaw) > abs(speaking.yaw)
    assert interrupted.pitch < listening.pitch
    assert max(abs(listening.x), abs(listening.y), abs(listening.z)) < 0.003
    assert abs(speaking.right_antenna - listening.right_antenna) > 0.001
    assert listening.right_antenna < 0 < listening.left_antenna
    assert math.isclose(speaking.right_antenna, -speaking.left_antenna)
    side = planner.plan(2.0, InteractionPhase.LISTENING, math.radians(90))
    assert math.radians(70) < side.yaw < math.radians(80)


class FakeRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, list[float], str]] = []

    def get_current_head_pose(self) -> np.ndarray:
        return np.eye(4, dtype=np.float64)

    def get_present_antenna_joint_positions(self) -> list[float]:
        return [0.0, 0.0]

    def set_target(
        self,
        head: np.ndarray | None = None,
        antennas: list[float] | np.ndarray | None = None,
        body_yaw: float | None = None,
    ) -> None:
        assert head is not None
        assert antennas is not None
        assert body_yaw is None
        self.calls.append(
            (
                np.asarray(head).copy(),
                list(antennas),
                threading.current_thread().name,
            )
        )


def test_motion_controller_is_the_single_fixed_rate_atomic_writer() -> None:
    robot = FakeRobot()
    controller = MotionController(
        robot,
        phase_source=lambda: InteractionPhase.LISTENING,
        doa_source=lambda: DoASnapshot(math.radians(20), 1.0, True, 0.0),
    )

    controller.start()
    assert controller.wait_ready(1.0)
    time.sleep(0.13)
    assert controller.stop()

    assert 4 <= len(robot.calls) <= 10
    assert {name for _, _, name in robot.calls} == {"yrobot-motion"}
    assert all(head.shape == (4, 4) for head, _, _ in robot.calls)
    assert all(len(antennas) == 2 for _, antennas, _ in robot.calls)
    assert controller.stats().commands == len(robot.calls)
