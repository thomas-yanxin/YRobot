import math

import numpy as np
import pytest

from yrobot.robot import angular_distance, doa_world_direction, resample_audio, to_mono


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
    np.testing.assert_allclose(doa_world_direction(0.0, pose), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(doa_world_direction(math.pi / 2, pose), [1.0, 0.0, 0.0], atol=1e-12)


def test_doa_rejects_invalid_pose() -> None:
    with pytest.raises(ValueError, match="4x4"):
        doa_world_direction(0.0, np.eye(3))


def test_angular_distance_wraps_at_pi() -> None:
    assert angular_distance(math.pi - 0.1, -math.pi + 0.1) == pytest.approx(0.2)
