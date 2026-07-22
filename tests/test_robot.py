import io
import logging
import math
import threading
import time

import numpy as np
import pytest
from reachy_mini.utils.interpolation import delta_angle_between_mat_rot
from scipy.spatial.transform import Rotation

from yrobot import robot as robot_module
from yrobot.config import CHUNK_SAMPLES
from yrobot.robot import (
    ANTENNA_POSES,
    DOA_GAZE_ELEVATION,
    INTERRUPT_ACK_TIMEOUT,
    INTERRUPT_PREROLL_SAMPLES,
    MAX_HEAD_ANGULAR_SPEED,
    MAX_HEAD_ANGULAR_STEP,
    MAX_HEAD_TRANSLATION_SPEED,
    MAX_HEAD_TRANSLATION_STEP,
    PLAYBACK_FRAME_SAMPLES,
    PLAYBACK_PREROLL_DECAY,
    PLAYBACK_PREROLL_MARGIN,
    PLAYBACK_PREROLL_SECONDS,
    AudioSampleBuffer,
    RobotIO,
    StreamingAudioResampler,
    UplinkAGC,
    angular_distance,
    doa_world_direction,
    downscale_jpeg,
    effective_conversation_state,
    gesture_pulse,
    lifelike_body_yaw,
    lifelike_motion_overlay,
    smooth_pose_step,
    step_pose,
    to_mono,
)


def test_audio_sample_buffer_preserves_fifo_and_bounded_tail() -> None:
    audio = AudioSampleBuffer(max_samples=5)
    audio.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    audio.append(np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))

    np.testing.assert_array_equal(audio.snapshot(), [3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_array_equal(audio.pop(2), [3.0, 4.0])
    np.testing.assert_array_equal(audio.snapshot(), [5.0, 6.0, 7.0])


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

    assert robot._player_clear_pending is True
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

    assert robot._player_clear_pending is False
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
    assert robot._player_clear_pending is True
    assert robot._playback_chunks.empty()
    # Burst audio of the interrupted turn is discarded until a turn boundary.
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1") is False

    robot.handle_omni_listen("r2")

    assert not robot.force_listen_active()
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "late") is False


def test_force_listen_timeout_frees_input_but_keeps_turn_discarded() -> None:
    robot = RobotIO(object())
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0

    assert robot._request_user_interrupt(-24.0, -30.0)
    robot._force_requested_at = time.monotonic() - INTERRUPT_ACK_TIMEOUT - 0.1

    # The control flag expires so real microphone slices flow again, but the
    # interrupted turn's burst audio must never resume mid-sentence: only a
    # listen boundary (the model actually stopped speaking) ends the discard.
    assert not robot.force_listen_active()
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1") is False

    # The user finished speaking a while ago, so this listen ends the hold.
    with robot._state_lock:
        robot._last_near_end_activity_at = time.monotonic() - 2.0
    robot.handle_omni_listen("r1")

    assert robot._discard_turn_active is False


def test_listen_ack_holds_playback_while_user_still_speaking() -> None:
    robot = RobotIO(object())
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0
    assert robot._request_user_interrupt(-24.0, -30.0)

    # The ack arrives while the user is still mid-utterance: the model's next
    # utterance would talk straight over them, so playback stays muted.
    with robot._state_lock:
        robot._last_user_speech_at = time.monotonic()
    robot.handle_omni_listen("r2")

    assert not robot.force_listen_active()
    assert robot._discard_turn_active is True
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r3") is False

    # The first listen after the user goes quiet reopens playback.
    with robot._state_lock:
        robot._last_user_speech_at = time.monotonic() - 2.0
        robot._last_near_end_activity_at = time.monotonic() - 2.0
    robot.handle_omni_listen("r4")

    assert robot._discard_turn_active is False


def test_talk_over_audio_reforces_listen_without_redetection() -> None:
    robot = RobotIO(object())
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0
    assert robot._request_user_interrupt(-24.0, -30.0)
    with robot._state_lock:
        robot._last_user_speech_at = time.monotonic()
    robot.handle_omni_listen("r2")
    assert not robot.force_listen_active()
    robot._emit_partial_event.clear()

    # Model audio arriving during the hold re-forces listen at once (the
    # energy detector's arm/confirm cycle would leak a second of talk-over).
    robot._force_requested_at = time.monotonic() - 1.5
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r3") is False

    assert robot.force_listen_active()
    assert robot._emit_partial_event.is_set()


def _pushable_mini(pushed: list[np.ndarray] | None = None):
    class Audio:
        def __init__(self) -> None:
            self.clear_count = 0

        def clear_player(self) -> None:
            self.clear_count += 1

    class Media:
        def __init__(self) -> None:
            self.audio = Audio()

        def push_audio_sample(self, frame: np.ndarray) -> None:
            if pushed is not None:
                pushed.append(np.asarray(frame, dtype=np.float32).copy())

    class Mini:
        def __init__(self) -> None:
            self.media = Media()

    return Mini()


def test_double_talk_candidate_ducks_playback_without_discarding_turn() -> None:
    robot = RobotIO(_pushable_mini())
    status, _ = robot._push_playback_frame(0, np.full(6_400, 0.1, dtype=np.float32))
    assert status == "pushed"

    now = time.monotonic()
    assert robot._begin_playback_hold(now)

    # The device queue is cycled (by the worker) but the un-played tail is
    # kept and the turn is not discarded: nothing destructive happened yet.
    assert robot._player_duck_pending is True
    assert robot._playback_resume_tail is not None
    assert robot._playback_resume_tail.size > 0
    with robot._state_lock:
        assert robot._speaking_until <= now
    assert not robot.force_listen_active()
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1") is True
    # A push racing the duck reports "held" so the worker preserves the frame.
    status, _ = robot._push_playback_frame(0, np.zeros(640, dtype=np.float32))
    assert status == "held"


def test_cancelled_verify_resumes_the_held_device_tail() -> None:
    pushed: list[np.ndarray] = []
    resumed = threading.Event()
    target_size = [1 << 30]

    mini = _pushable_mini(pushed)
    push_frame = mini.media.push_audio_sample

    def watched_push(frame: np.ndarray) -> None:
        push_frame(frame)
        if len(pushed) > 1 and sum(f.size for f in pushed[1:]) >= target_size[0]:
            resumed.set()

    mini.media.push_audio_sample = watched_push

    robot = RobotIO(mini)
    tail_audio = np.linspace(-0.5, 0.5, 6_400, dtype=np.float32)
    status, _ = robot._push_playback_frame(0, tail_audio)
    assert status == "pushed"

    assert robot._begin_playback_hold(time.monotonic())
    held_tail = robot._playback_resume_tail.copy()
    target_size[0] = held_tail.size
    robot._release_playback_hold(resume=True)

    worker = threading.Thread(target=robot._playback_loop, name="yrobot-playback", daemon=True)
    worker.start()
    try:
        assert resumed.wait(2.0)
    finally:
        robot._stop_event.set()
        robot._playback_wakeup.set()
        worker.join(timeout=1.0)

    # The duck cycled the shared pipeline exactly once, and the resumed
    # audio replays the un-played tail sample-for-sample, in order.
    assert mini.media.audio.clear_count == 1
    replayed = np.concatenate(pushed[1:])[: held_tail.size]
    assert np.array_equal(replayed, held_tail)


def test_confirmed_interrupt_during_hold_discards_turn_and_tail() -> None:
    robot = RobotIO(_pushable_mini())
    robot._push_playback_frame(0, np.full(6_400, 0.1, dtype=np.float32))
    assert robot._begin_playback_hold(time.monotonic())

    # _speaking_until was reset by the duck, but the held turn is still live
    # and must remain interruptible.
    assert robot._request_user_interrupt(-24.0, -38.0)

    assert robot.force_listen_active()
    assert robot._playback_hold_active is False
    assert robot._playback_resume_tail is None
    assert robot._discard_turn_active is True
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1") is False


def test_far_end_envelope_tracks_recent_playout_level() -> None:
    robot = RobotIO(_pushable_mini())
    assert robot._recent_far_end_db(time.monotonic()) == -120.0

    robot._push_playback_frame(0, np.full(1_600, 0.5, dtype=np.float32))

    level = robot._recent_far_end_db(time.monotonic())
    assert level == pytest.approx(20.0 * math.log10(0.5), abs=0.5)
    # Frames scheduled outside the lookback window stop contributing.
    assert robot._recent_far_end_db(time.monotonic() + 10.0) == -120.0


def test_playback_worker_owns_the_shared_pipeline_flush() -> None:
    cleared = threading.Event()

    class Audio:
        def __init__(self) -> None:
            self.thread_name = ""

        def clear_player(self) -> None:
            self.thread_name = threading.current_thread().name
            cleared.set()

    class Media:
        audio = Audio()

    class Mini:
        media = Media()

    mini = Mini()
    robot = RobotIO(mini)
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0
    assert robot._request_user_interrupt(-24.0, -30.0)

    worker = threading.Thread(target=robot._playback_loop, name="yrobot-playback", daemon=True)
    worker.start()
    try:
        assert cleared.wait(1.0)
        assert mini.media.audio.thread_name == "yrobot-playback"
        assert robot._player_clear_pending is False
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)


def test_uplink_agc_lifts_quiet_speech_and_leaves_loud_speech_alone() -> None:
    agc = UplinkAGC()
    quiet = np.full(16_000, 0.02, dtype=np.float32)
    lifted = agc.process(quiet)
    assert np.sqrt(np.mean(np.square(lifted))) == pytest.approx(0.12, rel=0.05)

    loud = np.full(16_000, 0.5, dtype=np.float32)
    np.testing.assert_array_equal(agc.process(loud), loud)

    silence = np.zeros(16_000, dtype=np.float32)
    np.testing.assert_array_equal(UplinkAGC().process(silence), silence)


def test_uplink_agc_freezes_its_estimate_while_the_robot_speaks() -> None:
    agc = UplinkAGC()
    agc.process(np.full(16_000, 0.05, dtype=np.float32))
    residual_echo = np.full(16_000, 0.01, dtype=np.float32)

    bypassed = agc.process(residual_echo, adapt=False)

    assert agc._speech_rms == pytest.approx(0.05)
    np.testing.assert_array_equal(bypassed, residual_echo)


def test_new_session_clears_stale_interruption_state() -> None:
    class Audio:
        def clear_player(self) -> None:
            pass

    class Media:
        audio = Audio()

    class Mini:
        media = Media()

    robot = RobotIO(Mini())
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0
    assert robot._request_user_interrupt(-24.0, -30.0)

    robot.reset_interruption()

    assert not robot.force_listen_active()
    assert robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r2") is True


def test_interrupt_ships_the_partial_microphone_slice_immediately() -> None:
    class Media:
        def get_audio_sample(self) -> np.ndarray:
            time.sleep(0.005)
            return np.full(160, 0.05, dtype=np.float32)  # 10 ms of speech

    class Mini:
        def __init__(self) -> None:
            self.media = Media()

    robot = RobotIO(Mini())
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r1")
    with robot._state_lock:
        robot._speaking_until = time.monotonic() + 2.0

    worker = threading.Thread(target=robot._capture_loop, daemon=True)
    worker.start()
    try:
        time.sleep(0.1)
        assert robot._request_user_interrupt(-24.0, -30.0)
        chunk = robot.next_audio_chunk(0.5)
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)

    assert chunk is not None
    assert 0 < chunk.size < CHUNK_SAMPLES
    assert chunk.size <= INTERRUPT_PREROLL_SAMPLES


def test_interrupt_preroll_overtakes_normal_uplink_audio() -> None:
    robot = RobotIO(object())
    normal = np.full(16_000, 0.1, dtype=np.float32)
    interruption = np.full(5_600, 0.2, dtype=np.float32)
    robot._put_latest(normal)
    robot._put_interrupt_audio(interruption)

    np.testing.assert_array_equal(robot.next_audio_chunk(0.01), interruption)
    np.testing.assert_array_equal(robot.next_audio_chunk(0.01), normal)


def test_slice_boundaries_within_one_utterance_do_not_restart_playback() -> None:
    robot = RobotIO(object())
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "resp_39")
    robot.set_conversation_state("speaking")
    # The per-slice response.done between resp_39 and resp_40 leaves the model
    # state alone, so the next slice continues the utterance without the
    # start-of-utterance preroll or a resampler reset.
    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "resp_40")

    assert robot._playback_chunks.get_nowait()[3] is True
    assert robot._playback_chunks.get_nowait()[3] is False


def test_playback_preroll_grows_with_supply_gaps_and_decays_per_utterance() -> None:
    pushed = threading.Event()

    class Media:
        def push_audio_sample(self, samples: np.ndarray) -> None:
            pushed.set()

    class Mini:
        def __init__(self) -> None:
            self.media = Media()

    robot = RobotIO(Mini())
    assert robot._playback_preroll == PLAYBACK_PREROLL_SECONDS

    robot.note_tts_supply_gap(0.3)
    assert robot._playback_preroll == pytest.approx(0.3 + PLAYBACK_PREROLL_MARGIN)

    robot.note_tts_supply_gap(0.06)
    assert robot._playback_preroll == pytest.approx(0.3 + PLAYBACK_PREROLL_MARGIN)

    # Listen slices arrive once per second while idle; they must not erode
    # the jitter buffer the way a started utterance does.
    robot.handle_omni_listen("r1")
    assert robot._playback_preroll == pytest.approx(0.3 + PLAYBACK_PREROLL_MARGIN)

    robot.play_omni_audio(np.zeros(2_400, dtype=np.float32), "r2")
    worker = threading.Thread(target=robot._playback_loop, name="yrobot-playback", daemon=True)
    worker.start()
    try:
        assert pushed.wait(2.0)
        deadline = time.monotonic() + 1.0
        while (
            robot._playback_preroll > 0.3 + PLAYBACK_PREROLL_MARGIN - 1e-9
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)
    assert robot._playback_preroll == pytest.approx(
        0.3 + PLAYBACK_PREROLL_MARGIN - PLAYBACK_PREROLL_DECAY
    )


def test_output_is_resampled_from_24k_to_16k() -> None:
    source = np.sin(np.linspace(0, 20 * math.pi, 2_400, endpoint=False)).astype(np.float32)
    converted = StreamingAudioResampler().process(source)
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


def test_interrupted_motion_yields_and_whole_body_freezes_for_user_speech() -> None:
    interrupted_head, interrupted_antennas = lifelike_motion_overlay(
        2.0,
        "interrupted",
        user_speaking=True,
    )

    assert delta_angle_between_mat_rot(np.eye(3), interrupted_head[:3, :3]) > 0.0
    np.testing.assert_allclose(
        interrupted_antennas,
        np.deg2rad(ANTENNA_POSES["interrupted"]),
    )
    assert lifelike_body_yaw(3.0, "speaking", user_speaking=False) != 0.0
    assert lifelike_body_yaw(3.0, "speaking", user_speaking=True) == 0.0


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


def test_playback_worker_paces_a_burst_instead_of_dumping_it_to_appsrc() -> None:
    pushed_twice = threading.Event()

    class Media:
        def __init__(self) -> None:
            self.samples: list[np.ndarray] = []

        def push_audio_sample(self, samples: np.ndarray) -> None:
            self.samples.append(samples.copy())
            if len(self.samples) >= 2:
                pushed_twice.set()

    class Mini:
        def __init__(self) -> None:
            self.media = Media()

    mini = Mini()
    robot = RobotIO(mini)
    worker = threading.Thread(target=robot._playback_loop, name="yrobot-playback", daemon=True)
    worker.start()
    try:
        robot.play_omni_audio(np.zeros(24_000, dtype=np.float32), "burst")
        assert pushed_twice.wait(1.0)
    finally:
        robot._stop_event.set()
        robot._playback_wakeup.set()
        worker.join(timeout=1.0)

    assert mini.media.samples[0].size == round(PLAYBACK_PREROLL_SECONDS * 16_000)
    assert mini.media.samples[1].size == PLAYBACK_FRAME_SAMPLES
    assert sum(part.size for part in mini.media.samples[:2]) < 16_000


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
        deadline = time.monotonic() + 1.0
        frame = robot.get_frame_jpeg()
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.01)
            frame = robot.get_frame_jpeg()
        assert frame == b"latest-jpeg"
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


def test_stalled_microphone_substitutes_silent_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_module, "CAPTURE_STALL_TIMEOUT", 0.05)

    class Media:
        def get_audio_sample(self) -> None:
            return None

    class Mini:
        media = Media()

    robot = RobotIO(Mini())
    worker = threading.Thread(target=robot._capture_loop, daemon=True)
    worker.start()
    try:
        chunk = robot.next_audio_chunk(1.0)
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)

    assert chunk is not None
    assert chunk.size == CHUNK_SAMPLES
    assert not chunk.any()


def test_capture_stall_pads_the_partial_slice_with_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_module, "CAPTURE_STALL_TIMEOUT", 0.05)
    speech = np.full(1_000, 0.25, dtype=np.float32)
    samples = [speech]

    class Media:
        def get_audio_sample(self) -> np.ndarray | None:
            return samples.pop() if samples else None

    class Mini:
        media = Media()

    robot = RobotIO(Mini())
    worker = threading.Thread(target=robot._capture_loop, daemon=True)
    worker.start()
    try:
        chunk = robot.next_audio_chunk(1.0)
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)

    assert chunk is not None
    assert chunk.size == CHUNK_SAMPLES
    np.testing.assert_array_equal(chunk[: speech.size], speech)
    assert not chunk[speech.size :].any()


def test_microphone_failure_keeps_the_slice_clock_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot_module, "CAPTURE_STALL_TIMEOUT", 0.05)

    class Media:
        def get_audio_sample(self) -> np.ndarray:
            raise RuntimeError("usb reset")

    class Mini:
        media = Media()

    robot = RobotIO(Mini())
    worker = threading.Thread(target=robot._capture_loop, daemon=True)
    worker.start()
    try:
        # The failure path retries at a one-second cadence before padding.
        chunk = robot.next_audio_chunk(2.5)
    finally:
        robot._stop_event.set()
        worker.join(timeout=2.0)

    assert chunk is not None
    assert not chunk.any()


def test_camera_frames_are_downscaled_to_the_server_vision_size() -> None:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (1280, 720), (90, 120, 40)).save(buffer, format="JPEG")
    original = buffer.getvalue()

    shrunk = downscale_jpeg(original)

    with pil_image.open(io.BytesIO(shrunk)) as image:
        assert image.size == (448, 252)
    assert len(shrunk) < len(original)


def test_small_camera_frames_pass_through_unchanged() -> None:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (320, 240)).save(buffer, format="JPEG")
    original = buffer.getvalue()

    assert downscale_jpeg(original) is original


def test_unparseable_camera_frame_is_sent_unchanged() -> None:
    assert downscale_jpeg(b"not-a-jpeg") == b"not-a-jpeg"


def test_camera_loop_caches_downscaled_frames() -> None:
    pil_image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    pil_image.new("RGB", (1280, 720)).save(buffer, format="JPEG")
    native_frame = buffer.getvalue()
    captured = threading.Event()

    class Media:
        def get_frame_jpeg(self) -> bytes:
            captured.set()
            return native_frame

    class Mini:
        media = Media()

    robot = RobotIO(Mini())
    worker = threading.Thread(target=robot._camera_loop, daemon=True)
    worker.start()
    try:
        assert captured.wait(1.0)
        deadline = time.monotonic() + 1.0
        frame = robot.get_frame_jpeg()
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.01)
            frame = robot.get_frame_jpeg()
    finally:
        robot._stop_event.set()
        worker.join(timeout=1.0)

    assert frame is not None
    with pil_image.open(io.BytesIO(frame)) as image:
        assert max(image.size) == 448
