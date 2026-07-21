import math
import threading
import time

import numpy as np
import pytest
from reachy_mini.utils.interpolation import delta_angle_between_mat_rot
from scipy.spatial.transform import Rotation

from yrobot.robot import (
    BARGE_IN_MIN_LEVEL_DB,
    BARGE_IN_RELEASE_SILENCE,
    DOA_GAZE_ELEVATION,
    MAX_HEAD_ANGULAR_STEP,
    MAX_HEAD_TRANSLATION_STEP,
    RobotIO,
    angular_distance,
    audio_level_db,
    doa_world_direction,
    is_near_end_speech,
    resample_audio,
    step_pose,
    to_mono,
)


def test_stereo_microphone_is_mixed_to_mono() -> None:
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(stereo), [0.0, 0.5])


def test_microphone_level_is_dbfs_rms() -> None:
    assert audio_level_db(np.ones(160, dtype=np.float32)) == pytest.approx(0.0)
    assert audio_level_db(np.full(160, 0.01, dtype=np.float32)) == pytest.approx(-40.0)
    assert audio_level_db(np.zeros(160, dtype=np.float32)) == pytest.approx(-120.0)


def test_doa_alone_cannot_trigger_barge_in() -> None:
    assert not is_near_end_speech(True, BARGE_IN_MIN_LEVEL_DB - 1.0)
    assert not is_near_end_speech(False, 0.0)
    assert is_near_end_speech(True, BARGE_IN_MIN_LEVEL_DB)


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


def test_omni_audio_is_played_by_dedicated_worker() -> None:
    pushed = threading.Event()

    class Media:
        def __init__(self) -> None:
            self.samples: list[np.ndarray] = []
            self.thread_name = ""

        def push_audio_sample(self, samples: np.ndarray) -> None:
            self.samples.append(samples)
            self.thread_name = threading.current_thread().name
            pushed.set()

    class Mini:
        def __init__(self) -> None:
            self.media = Media()

    mini = Mini()
    robot = RobotIO(mini)
    worker = threading.Thread(
        target=robot._playback_loop, name="yrobot-playback", daemon=True
    )
    worker.start()
    try:
        robot.play_omni_audio(np.zeros(2_400, dtype=np.float32))
        assert pushed.wait(1.0)
        assert mini.media.thread_name == "yrobot-playback"
        assert mini.media.samples[0].shape == (1_600,)
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)


def test_barge_in_stays_active_until_user_and_server_are_done() -> None:
    class Audio:
        def __init__(self) -> None:
            self.clear_count = 0

        def clear_player(self) -> None:
            self.clear_count += 1

    class Media:
        def __init__(self) -> None:
            self.audio = Audio()

    class Mini:
        def __init__(self) -> None:
            self.media = Media()

    mini = Mini()
    robot = RobotIO(mini)
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32))

    assert robot.interrupt_omni_audio()
    assert robot._playback_chunks.empty()
    assert mini.media.audio.clear_count == 1
    assert robot.force_listen_active()
    assert robot.force_listen_active()

    # A server listen event alone cannot release suppression while the user is
    # still speaking, and user silence alone cannot release before the server
    # has acknowledged a force_listen frame.
    robot.note_force_listen_sent("session_resp_7")
    robot.confirm_omni_listening("session_resp_6")
    assert robot.force_listen_active()
    robot.confirm_omni_listening("session_resp_7")
    assert robot.force_listen_active()
    robot._update_barge_in_release(
        False,
        time.monotonic() + BARGE_IN_RELEASE_SILENCE,
    )
    assert not robot.force_listen_active()

    # Late chunks from the interrupted response are ignored briefly after the
    # listen acknowledgement.
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32))
    assert robot._playback_chunks.empty()


def test_sustained_near_end_speech_interrupts_active_playback() -> None:
    interrupted = threading.Event()

    class Audio:
        def clear_player(self) -> None:
            interrupted.set()

    class Media:
        def __init__(self) -> None:
            self.audio = Audio()

        def get_DoA(self) -> tuple[float, bool]:
            return 0.0, True

    class Mini:
        def __init__(self) -> None:
            self.media = Media()

        def set_target(self, **kwargs: object) -> None:
            pass

        def get_current_head_pose(self) -> np.ndarray:
            return np.eye(4)

        def look_at_world(self, *args: float, **kwargs: object) -> np.ndarray:
            return np.eye(4)

    robot = RobotIO(Mini())
    with robot._state_lock:
        robot._speaking_until = float("inf")
        robot._barge_in_armed_at = 0.0
        robot._microphone_level_db = BARGE_IN_MIN_LEVEL_DB + 6.0

    worker = threading.Thread(target=robot._motion_loop, daemon=True)
    worker.start()
    try:
        assert interrupted.wait(1.0)
        assert robot.force_listen_active()
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)
