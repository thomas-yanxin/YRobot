"""Unit tests for DoA mapping and motion primitives (no hardware)."""

import math

import numpy as np

from yrobot.motion import GazeSpring, circular_mean, doa_to_yaw_delta, head_yaw_of, rpy_pose


def test_doa_angle_convention():
    # XVF3800: 0 = left, pi/2 = front, pi = right (head-relative).
    assert math.isclose(doa_to_yaw_delta(0.0), math.pi / 2)  # turn left
    assert math.isclose(doa_to_yaw_delta(math.pi), -math.pi / 2)  # turn right
    assert math.isclose(doa_to_yaw_delta(math.pi / 2), 0.0)  # already facing


def test_circular_mean_handles_wraparound():
    mean = circular_mean([math.pi - 0.1, -math.pi + 0.1])
    assert math.isclose(abs(mean), math.pi, abs_tol=1e-6)


def test_rpy_pose_yaw_roundtrip():
    for yaw in (-1.2, 0.0, 0.7, 2.0):
        assert math.isclose(head_yaw_of(rpy_pose(0.05, -0.1, yaw, 0.002)), yaw, abs_tol=1e-9)


def test_rpy_pose_shape_and_z():
    pose = rpy_pose(0.0, 0.0, 0.0, 0.004)
    assert pose.shape == (4, 4)
    assert np.allclose(pose[:3, :3], np.eye(3))
    assert pose[2, 3] == 0.004


def test_gaze_spring_converges_without_overshoot():
    spring = GazeSpring()
    spring.target = 1.0
    peak = 0.0
    for _ in range(500):  # 10 s at 50 Hz
        peak = max(peak, spring.step(0.02))
    assert peak <= 1.0 + 1e-6
    assert math.isclose(spring.pos, 1.0, abs_tol=0.01)


def test_gaze_spring_velocity_clamp():
    spring = GazeSpring(max_vel=1.0)
    spring.target = 100.0
    spring.step(0.02)
    assert abs(spring.vel) <= 1.0
