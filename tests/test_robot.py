import logging
import math
import threading
import time

import numpy as np
import pytest
from reachy_mini.utils.interpolation import delta_angle_between_mat_rot
from scipy.spatial.transform import Rotation

from yrobot.robot import (
    ANTENNA_POSES,
    DOA_GAZE_ELEVATION,
    MAX_HEAD_ANGULAR_SPEED,
    MAX_HEAD_ANGULAR_STEP,
    MAX_HEAD_TRANSLATION_SPEED,
    MAX_HEAD_TRANSLATION_STEP,
    RobotIO,
    StreamingAudioResampler,
    angular_distance,
    doa_world_direction,
    effective_conversation_state,
    gesture_pulse,
    lifelike_motion_overlay,
    resample_audio,
    smooth_pose_step,
    step_pose,
    to_mono,
)


def test_stereo_microphone_uses_xvf_processed_channel_zero() -> None:
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(stereo), [1.0, 0.5])

    channels_first = np.array([[1.0, 0.5, 0.25], [-1.0, 0.5, -0.25]], dtype=np.float32)
    np.testing.assert_allclose(to_mono(channels_first), [1.0, 0.5, 0.25])


def test_post_aec_microphone_is_uploaded_unchanged_during_playback() -> None:
    robot = RobotIO(object())
    microphone = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    robot._audio_chunks.put(microphone)
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 1.0

    uploaded = robot.next_audio_chunk(0.01)

    assert uploaded is not None
    np.testing.assert_array_equal(uploaded, microphone)


def test_xvf_configuration_uses_verified_settled_writes() -> None:
    calls: list[tuple[object, bool, float]] = []

    class Audio:
        def apply_audio_config(
            self,
            config: object,
            *,
            verify: bool,
            write_settle_seconds: float,
        ) -> bool:
            calls.append((config, verify, write_settle_seconds))
            return True

    class Media:
        audio = Audio()

    class Mini:
        media = Media()

    RobotIO(Mini())._apply_audio_startup_config()

    assert len(calls) == 1
    assert calls[0][1:] == (True, 0.1)


def test_model_listen_flushes_playback_after_recent_near_end_activity() -> None:
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
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0
        robot._last_near_end_activity_at = time.monotonic()

    robot.handle_omni_listen("r2")

    assert mini.media.audio.clear_count == 1
    assert robot._playback_chunks.empty()
    assert not robot.force_listen_active()


def test_quiet_model_listen_allows_buffered_sentence_tail_to_drain() -> None:
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
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0

    robot.handle_omni_listen("r2")

    assert mini.media.audio.clear_count == 0
    assert not robot._playback_chunks.empty()


def test_high_confidence_double_talk_forces_listen_until_acknowledged() -> None:
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
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0

    assert robot._request_user_interrupt(-24.0, -30.0)
    assert robot.force_listen_active()
    assert mini.media.audio.clear_count == 1
    assert robot._playback_chunks.empty()

    robot.handle_omni_listen("r2")

    assert not robot.force_listen_active()
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "late") is False


def test_output_is_resampled_from_24k_to_16k() -> None:
    source = np.sin(np.linspace(0, 20 * math.pi, 2_400, endpoint=False)).astype(np.float32)
    converted = resample_audio(source)
    assert converted.dtype == np.float32
    assert len(converted) == 1_600
    assert np.max(np.abs(converted)) <= 1.0


def test_streaming_resampler_is_continuous_across_arbitrary_deltas() -> None:
    source = np.sin(np.linspace(0, 200 * math.pi, 24_017, endpoint=False)).astype(np.float32)
    whole = StreamingAudioResampler().process(source)
    chunked_resampler = StreamingAudioResampler()
    sizes = (317, 911, 2_003, 79, 4_097)
    chunks: list[np.ndarray] = []
    offset = 0
    index = 0
    while offset < source.size:
        size = sizes[index % len(sizes)]
        chunks.append(chunked_resampler.process(source[offset : offset + size]))
        offset += size
        index += 1

    chunked = np.concatenate(chunks)
    np.testing.assert_allclose(chunked, whole, atol=1e-6)
    assert abs(chunked.size - source.size * 2 / 3) < 1


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


def test_reactive_pose_servo_is_rate_independent_and_eased() -> None:
    current = np.eye(4)
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    target[0, 3] = 0.1
    elapsed = 0.02

    stepped = smooth_pose_step(current, target, elapsed)

    angular_step = delta_angle_between_mat_rot(current[:3, :3], stepped[:3, :3])
    translation_step = np.linalg.norm(stepped[:3, 3] - current[:3, 3])
    assert 0.0 < angular_step <= MAX_HEAD_ANGULAR_SPEED * elapsed + 1e-9
    assert 0.0 < translation_step <= MAX_HEAD_TRANSLATION_SPEED * elapsed + 1e-9
    np.testing.assert_allclose(smooth_pose_step(current, target, 0.0), current)


def test_gesture_pulse_has_minimum_jerk_return_to_rest() -> None:
    assert gesture_pulse(-0.1) == 0.0
    assert gesture_pulse(0.0) == 0.0
    assert gesture_pulse(0.25) == pytest.approx(0.5)
    assert gesture_pulse(0.5) == 1.0
    assert gesture_pulse(0.75) == pytest.approx(0.5)
    assert gesture_pulse(1.0) == 0.0
    assert gesture_pulse(1.1) == 0.0


def test_lifelike_overlay_is_restrained_and_listening_antennas_can_hold() -> None:
    base_listening = np.deg2rad(ANTENNA_POSES["listening"])
    quiet_head, quiet_antennas = lifelike_motion_overlay(
        3.7,
        "listening",
        user_speaking=False,
        nod_pulse=1.0,
        glance_pulse=1.0,
        glance_yaw_degrees=6.0,
        glance_pitch_degrees=1.4,
    )
    _, held_antennas = lifelike_motion_overlay(
        3.7,
        "listening",
        user_speaking=True,
    )

    angular_offset = delta_angle_between_mat_rot(np.eye(3), quiet_head[:3, :3])
    assert angular_offset < math.radians(10.0)
    assert abs(quiet_head[2, 3]) < 0.002
    assert np.max(np.abs(quiet_antennas)) < math.radians(25.0)
    np.testing.assert_allclose(held_antennas, base_listening)


def test_playback_deadline_clears_a_late_stale_speaking_state() -> None:
    assert effective_conversation_state("speaking", speaking=True) == "speaking"
    assert effective_conversation_state("listening", speaking=True) == "speaking"
    assert effective_conversation_state("speaking", speaking=False) == "listening"
    assert effective_conversation_state("idle", speaking=False) == "idle"


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
    worker = threading.Thread(target=robot._playback_loop, name="yrobot-playback", daemon=True)
    worker.start()
    try:
        robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "response-1")
        assert pushed.wait(1.0)
        assert mini.media.thread_name == "yrobot-playback"
        assert mini.media.samples[0].shape == (1_600,)
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)


def test_camera_jpeg_is_cached_off_the_sender_path() -> None:
    captured = threading.Event()

    class Media:
        def get_frame_jpeg(self) -> bytes:
            captured.set()
            return b"latest-jpeg"

    class Mini:
        def __init__(self) -> None:
            self.media = Media()

    robot = RobotIO(Mini())
    worker = threading.Thread(target=robot._camera_loop, daemon=True)
    worker.start()
    try:
        assert captured.wait(1.0)
        assert robot.get_frame_jpeg() == b"latest-jpeg"
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)


def test_brief_sdk_liveness_miss_is_retried_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recovered = threading.Event()

    class Media:
        def get_DoA(self) -> tuple[float, bool]:
            return 0.0, False

    class Mini:
        def __init__(self) -> None:
            self.media = Media()
            self.command_count = 0

        def set_target(self, **kwargs: object) -> None:
            del kwargs
            self.command_count += 1
            if self.command_count <= 3:
                raise ConnectionError("Lost connection with the server.")
            recovered.set()

    mini = Mini()
    robot = RobotIO(mini)
    worker = threading.Thread(target=robot._motion_loop, daemon=True)
    with caplog.at_level(logging.WARNING, logger="yrobot.robot"):
        worker.start()
        try:
            assert recovered.wait(1.0)
        finally:
            robot._stop_event.set()
            worker.join(timeout=1.0)

    assert mini.command_count >= 4
    assert "Motion command failed" not in caplog.text
