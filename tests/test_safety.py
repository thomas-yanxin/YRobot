import math

import numpy as np

from reachy_mini_live_chat.motion import safety


def test_clamp_rpy_within_limits():
    assert safety.clamp_rpy_deg(10, -20, 90) == (10, -20, 90)


def test_clamp_rpy_exceeding():
    r, p, y = safety.clamp_rpy_deg(80, -75, 300)
    assert r == 40 and p == -40 and y == 180


def test_rpy_matrix_round_trip():
    # the numpy rpy<->matrix helpers (which replaced scipy) must invert cleanly
    for rpy in [(0, 0, 0), (10, -20, 35), (-40, 39, 179), (5, 0, -170)]:
        m = safety.rpy_to_matrix(*rpy, degrees=True)
        back = safety.matrix_to_rpy(m, degrees=True)
        assert np.allclose(back, rpy, atol=1e-6)


def test_clamp_head_pose_limits_pitch_roll():
    # request roll=70, pitch=-60 which exceed +/-40
    pose = np.eye(4)
    pose[:3, :3] = safety.rpy_to_matrix(70, -60, 10, degrees=True)
    clamped = safety.clamp_head_pose(pose)
    roll, pitch, yaw = safety.matrix_to_rpy(clamped[:3, :3], degrees=True)
    assert abs(roll) <= 40 + 1e-6
    assert abs(pitch) <= 40 + 1e-6
    assert abs(yaw - 10) < 1e-3


def test_body_yaw_range():
    # head near the body target so the delta limit doesn't bind; range clamp -> 160deg
    out = safety.clamp_body_yaw(math.radians(200), math.radians(120))
    assert abs(out - math.radians(160)) < 1e-6


def test_body_yaw_delta_binds_when_head_centered():
    # head at 0 -> body physically limited to +/-65deg regardless of the 160deg range
    out = safety.clamp_body_yaw(math.radians(200), 0.0)
    assert abs(out - math.radians(safety.YAW_DELTA_DEG)) < 1e-6


def test_body_yaw_delta_enforced():
    # head at +60deg, ask body for -60deg -> delta 120deg, must clamp to <=65deg
    head = math.radians(60)
    body = safety.clamp_body_yaw(math.radians(-60), head)
    delta = math.degrees(body - head)
    assert abs(delta) <= safety.YAW_DELTA_DEG + 1e-6


def test_clamp_antennas():
    out = safety.clamp_antennas([math.radians(200), math.radians(-200)])
    assert all(abs(a) <= math.radians(safety.ANTENNA_DEG) + 1e-9 for a in out)
