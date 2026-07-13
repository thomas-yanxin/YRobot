import math

from reachy_mini_live_chat.motion.doa import (
    DoaTracker,
    doa_to_head_yaw,
    doa_to_world_point,
)


def test_front_is_zero_yaw():
    assert abs(doa_to_head_yaw(math.pi / 2)) < 1e-9


def test_left_is_positive_yaw():
    # DOA 0 = left -> capped positive yaw
    assert doa_to_head_yaw(0.0) > 0


def test_right_is_negative_yaw():
    assert doa_to_head_yaw(math.pi) < 0


def test_yaw_capped():
    # far-left beyond cap stays within MAX_DOA_YAW
    from reachy_mini_live_chat.motion.doa import MAX_DOA_YAW

    assert abs(doa_to_head_yaw(-1.0)) <= MAX_DOA_YAW + 1e-9


def test_world_point_front():
    x, y, z = doa_to_world_point(math.pi / 2)
    assert x > 0.9 and abs(y) < 1e-6 and z == 0.0


def test_tracker_gates_small_changes():
    tr = DoaTracker(alpha=1.0, min_change_deg=12.0)
    first = tr.update(math.pi / 2)  # front
    assert first is not None
    # a tiny change should be gated out (returns None)
    assert tr.update(math.pi / 2 + math.radians(3)) is None


def test_tracker_none_on_none():
    assert DoaTracker().update(None) is None
