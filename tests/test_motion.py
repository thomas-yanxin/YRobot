"""Unit tests for DoA mapping and motion primitives (no hardware)."""

import math
import time

import numpy as np

from yrobot.motion import (
    Choreographer,
    GazeSpring,
    SoundCompass,
    circular_mean,
    doa_to_yaw_delta,
    head_yaw_of,
    rpy_pose,
    weighted_circular_mean,
)


def test_doa_angle_convention():
    # XVF3800: 0 = left, pi/2 = front, pi = right (head-relative).
    assert math.isclose(doa_to_yaw_delta(0.0), math.pi / 2)  # turn left
    assert math.isclose(doa_to_yaw_delta(math.pi), -math.pi / 2)  # turn right
    assert math.isclose(doa_to_yaw_delta(math.pi / 2), 0.0)  # already facing


def test_circular_mean_handles_wraparound():
    mean = circular_mean([math.pi - 0.1, -math.pi + 0.1])
    assert math.isclose(abs(mean), math.pi, abs_tol=1e-6)


def test_weighted_circular_mean_prioritizes_device_confirmed_samples():
    mean = weighted_circular_mean([(0.0, 2.0), (math.pi / 2, 1.0)])
    assert 0.0 < mean < math.pi / 4


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


def test_sound_compass_survives_usb_errors():
    # XVF3800 control reads throw transient USB I/O errors under bus
    # contention; the thread must back off, not die (hardware 2026-07-24).
    class FlakyMedia:
        def get_DoA(self):
            raise OSError(5, "Input/Output Error")

    compass = SoundCompass(
        FlakyMedia(),
        current_head_yaw=lambda: 0.0,
        user_active=lambda: True,
        on_target=lambda yaw: None,
    )
    compass.start()
    time.sleep(0.5)
    try:
        assert compass.is_alive()  # backed off instead of crashing
    finally:
        compass.close()
        compass.join(timeout=2)
    assert not compass.is_alive()


def test_gaze_spring_freeze_brakes_smoothly_and_holds():
    spring = GazeSpring()
    spring.target = 2.0
    for _ in range(10):
        spring.step(0.02)  # mid-turn
    assert abs(spring.vel) > 0.1
    for _ in range(25):
        spring.step(0.02, freeze=1.0)  # 0.5 s of hold-still
    assert abs(spring.vel) < 0.01  # braked, no jerk, no drive
    held = spring.pos
    spring.step(0.02, freeze=1.0)
    assert abs(spring.pos - held) < 1e-3


def test_idle_saccade_target_is_trajectory_limited(monkeypatch):
    class FakeMini:
        pass

    choreo = Choreographer(FakeMini())

    def upper_bound(low, high):
        return high

    monkeypatch.setattr("yrobot.motion.random.uniform", upper_bound)
    pose, _ = choreo._compose(t=0.0, now=1.0, dt=0.02)
    yaw = head_yaw_of(pose)
    # The random target is +0.25 rad, but it must not appear in one 20 ms tick.
    assert 0.0 < yaw < 0.03
