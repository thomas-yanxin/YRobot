import math

import numpy as np
import pytest
from reachy_mini.utils.interpolation import delta_angle_between_mat_rot
from scipy.spatial.transform import Rotation

from yrobot.robot import (
    DOA_GAZE_ELEVATION,
    MAX_HEAD_ANGULAR_STEP,
    MAX_HEAD_TRANSLATION_STEP,
    angular_distance,
    doa_world_direction,
    resample_audio,
    step_pose,
    to_mono,
)


def test_stereo_microphone_is_mixed_to_mono() -> None:
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(stereo), [0.0, 0.5])


def test_output_is_resampled_from_24k_to_16k() -> None:
    source = np.sin(np.linspace(0, 20 * math.pi, 2_400, endpoint=False)).astype(np.float32)
    converted = resample_audio(source)
    assert converted.dtype == np.float32
    assert len(converted) == 1_600
    assert np.max(np.abs(converted)) <= 1.0


def test_doa_uses_reachy_head_coordinates() -> None:
    pose = np.eye(4)
    np.testing.assert_allclose(doa_world_direction(0.0, pose), [0.0, 1.0, DOA_GAZE_ELEVATION])
    np.testing.assert_allclose(
        doa_world_direction(math.pi / 2, pose),
        [1.0, 0.0, DOA_GAZE_ELEVATION],
        atol=1e-12,
    )


def test_doa_ignores_head_pitch_but_preserves_yaw() -> None:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [15, 20, 90], degrees=True).as_matrix()

    direction = doa_world_direction(math.pi / 2, pose)

    np.testing.assert_allclose(direction, [0.0, 1.0, DOA_GAZE_ELEVATION], atol=1e-12)


def test_doa_rejects_invalid_pose() -> None:
    with pytest.raises(ValueError, match="4x4"):
        doa_world_direction(0.0, np.eye(3))


def test_angular_distance_wraps_at_pi() -> None:
    assert angular_distance(math.pi - 0.1, -math.pi + 0.1) == pytest.approx(0.2)


def test_pose_step_bounds_rotation_and_translation() -> None:
    current = np.eye(4)
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    target[0, 3] = 0.1

    stepped = step_pose(current, target)

    angular_step = delta_angle_between_mat_rot(current[:3, :3], stepped[:3, :3])
    translation_step = np.linalg.norm(stepped[:3, 3] - current[:3, 3])
    assert angular_step <= MAX_HEAD_ANGULAR_STEP + 1e-9
    assert translation_step <= MAX_HEAD_TRANSLATION_STEP + 1e-9


def test_pose_step_reaches_nearby_target() -> None:
    current = np.eye(4)
    target = np.eye(4)
    target[0, 3] = MAX_HEAD_TRANSLATION_STEP / 2
    np.testing.assert_allclose(step_pose(current, target), target)
