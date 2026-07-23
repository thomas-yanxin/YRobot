import queue
import threading
import time

import numpy as np

from yrobot.audio import (
    AUDIO_STARTUP_CONFIG,
    AudioEngine,
    PlayerClearError,
    StreamingResampler24To16,
    _FarEndEchoGuard,
)


def wait_until(predicate: object, timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


class EnergyVad:
    def is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        assert sample_rate == 16_000
        samples = np.frombuffer(pcm, dtype="<i2")
        return bool(np.max(np.abs(samples)) > 1_000)


class FakePlayer:
    def __init__(self, media: "FakeMedia", *, fail_clear: bool = False) -> None:
        self.media = media
        self.fail_clear = fail_clear
        self.clear_calls = 0
        self.config_calls: list[tuple[object, bool, float]] = []
        self.max_output_buffers: list[int] = []

    def clear_player(self) -> None:
        self.media.events.append("clear")
        self.clear_calls += 1
        if self.fail_clear:
            raise OSError("fake flush failure")
        with self.media.output_lock:
            self.media.audible.clear()

    def apply_audio_config(
        self,
        config: object,
        *,
        verify: bool,
        write_settle_seconds: float,
    ) -> bool:
        self.media.events.append("configure")
        self.config_calls.append((config, verify, write_settle_seconds))
        return True

    def set_max_output_buffers(self, value: int) -> None:
        self.media.events.append("bound_output")
        self.max_output_buffers.append(value)


class FakeMedia:
    def __init__(
        self,
        *,
        fail_clear: bool = False,
        camera_delay: float = 0.0,
        push_delay: float = 0.0,
    ) -> None:
        self.input: queue.Queue[np.ndarray] = queue.Queue()
        self.output_lock = threading.Lock()
        self.audible: list[np.ndarray] = []
        self.push_history: list[np.ndarray] = []
        self.push_timestamps: list[float] = []
        self.audio = FakePlayer(self, fail_clear=fail_clear)
        self.camera_delay = camera_delay
        self.push_delay = push_delay
        self.recording = False
        self.playing = False
        self.events: list[str] = []

    def start_recording(self) -> None:
        self.events.append("start_recording")
        self.recording = True

    def stop_recording(self) -> None:
        self.recording = False

    def start_playing(self) -> None:
        self.events.append("start_playing")
        self.playing = True

    def stop_playing(self) -> None:
        self.playing = False

    def get_audio_sample(self) -> np.ndarray | None:
        try:
            return self.input.get_nowait()
        except queue.Empty:
            return None

    def push_audio_sample(self, samples: np.ndarray) -> None:
        self.push_timestamps.append(time.monotonic())
        if self.push_delay:
            time.sleep(self.push_delay)
        copied = np.asarray(samples, dtype=np.float32).copy()
        with self.output_lock:
            self.audible.append(copied)
            self.push_history.append(copied)

    def get_frame_jpeg(self) -> bytes:
        if self.camera_delay:
            time.sleep(self.camera_delay)
        return b"jpeg"

    def feed_mono(self, samples: np.ndarray, channel_one: float = 0.9) -> None:
        mono = np.asarray(samples, dtype=np.float32)
        stereo = np.column_stack((mono, np.full(mono.size, channel_one, dtype=np.float32)))
        self.input.put(stereo)


class FakeMini:
    def __init__(self, media: FakeMedia) -> None:
        self.media = media


def make_engine(media: FakeMedia, **kwargs: object) -> AudioEngine:
    return AudioEngine(
        FakeMini(media),
        vad=EnergyVad(),
        capture_video=False,
        **kwargs,
    )


def test_exact_channel_zero_uplink_units_drop_oldest_when_bounded() -> None:
    media = FakeMedia()
    engine = make_engine(media, uplink_queue_size=2)
    source = np.linspace(-0.4, 0.4, 48_000, dtype=np.float32)
    engine.start()
    try:
        offsets = (0, 173, 9_111, 16_731, 35_007, source.size)
        for start, end in zip(offsets, offsets[1:], strict=False):
            media.feed_mono(source[start:end])
        wait_until(lambda: engine.metrics["uplink_units"] == 3)

        second = engine.next_audio_unit(0.1)
        third = engine.next_audio_unit(0.1)
        assert second is not None
        assert third is not None
        np.testing.assert_array_equal(second[0], source[16_000:32_000])
        np.testing.assert_array_equal(third[0], source[32_000:48_000])
        assert second[0].shape == (16_000,)
        assert second[0].dtype == np.float32
        assert second[1] is False
        assert third[1] is False
        assert engine.metrics["uplink_dropped_units"] == 1
    finally:
        engine.stop()


def test_microphone_epoch_opens_only_after_session_ready() -> None:
    media = FakeMedia()
    engine = make_engine(media)
    engine.start(session_ready=False)
    try:
        assert engine.state == "idle"
        media.feed_mono(np.full(16_000, 0.05, dtype=np.float32))
        wait_until(media.input.empty)
        assert engine.next_audio_unit(0.05) is None

        engine.handle_session_ready()
        media.feed_mono(np.full(16_000, 0.05, dtype=np.float32))
        unit = engine.next_audio_unit(0.5)
        assert unit is not None
        np.testing.assert_allclose(unit[0], 0.05, rtol=0.0, atol=1e-7)
    finally:
        engine.stop()


def test_barge_in_flushes_and_rejects_late_audio_from_old_response() -> None:
    media = FakeMedia()
    states: list[str] = []
    engine = AudioEngine(
        FakeMini(media),
        vad=EnergyVad(),
        capture_video=False,
        state_callback=states.append,
        playback_lead_seconds=0.100,
    )
    engine.start()
    try:
        wait_until(lambda: media.audio.clear_calls >= 1)
        initial_clears = media.audio.clear_calls
        media.feed_mono(np.zeros(16_000, dtype=np.float32))
        wait_until(lambda: engine.metrics["uplink_queue_depth"] == 1)
        engine.handle_audio_delta(np.full(24_000, 0.2, dtype=np.float32), "old")
        media.feed_mono(np.full(3 * 320, 0.12, dtype=np.float32))

        wait_until(lambda: engine.state == "interrupted")
        wait_until(lambda: media.audio.clear_calls > initial_clears)
        interrupted_clears = media.audio.clear_calls
        with media.output_lock:
            assert media.audible == []

        # A network delta already in flight after epoch invalidation is never
        # admitted, even after enough time for normal playback to start.
        engine.handle_audio_delta(np.full(4_800, 0.3, dtype=np.float32), "old")
        time.sleep(0.15)
        with media.output_lock:
            assert media.audible == []
        assert engine.metrics["interruptions"] == 1

        # A padded control unit dispatches force_listen immediately instead of
        # waiting up to one second for the normal capture unit to fill.
        control = engine.next_audio_unit(0.2)
        assert control is not None and control[1] is True
        np.testing.assert_allclose(control[0][: 2 * 320], 0.12, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(control[0][2 * 320 :], 0.0)
        assert engine.metrics["force_control_units"] == 1
        assert engine.metrics["last_force_dispatch_latency_ms"] >= 0.0

        # Subsequent near-end speech remains continuous in ordinary one-second
        # model units while the force latch awaits listen acknowledgement.
        media.feed_mono(np.full(16_000, 0.12, dtype=np.float32))
        speech = engine.next_audio_unit(1.0)
        assert speech is not None and speech[1] is True
        np.testing.assert_allclose(speech[0], 0.12, rtol=0.0, atol=1e-6)

        engine.handle_listen("listen")
        wait_until(lambda: media.audio.clear_calls > interrupted_clears)
        assert engine.state == "listening"
        assert engine.metrics["last_interrupt_to_listen_ms"] >= 0.0
        assert states[-3:] == ["speaking", "interrupted", "listening"]
    finally:
        engine.stop()


def test_one_self_speech_frame_does_not_interrupt_model() -> None:
    media = FakeMedia()
    states: list[str] = []
    engine = AudioEngine(
        FakeMini(media),
        vad=EnergyVad(),
        capture_video=False,
        state_callback=states.append,
        vad_onset_frames=3,
        playback_lead_seconds=0.060,
    )
    engine.start()
    try:
        engine.handle_audio_delta(np.full(4_800, 0.2, dtype=np.float32), "response")
        media.feed_mono(np.full(320, 0.12, dtype=np.float32))
        media.feed_mono(np.zeros(4 * 320, dtype=np.float32))
        wait_until(lambda: len(media.push_history) > 0)
        assert engine.state == "speaking"
        assert engine.metrics["interruptions"] == 0
        assert engine.metrics["last_delta_to_speaker_ms"] >= 0.0
        assert "interrupted" not in states
    finally:
        engine.stop()


def test_recent_playback_echo_is_not_mistaken_for_barge_in() -> None:
    media = FakeMedia()
    engine = make_engine(media, playback_lead_seconds=0.060)
    source = (0.25 * np.sin(2.0 * np.pi * 613.0 * np.arange(24_000) / 24_000)).astype(np.float32)
    engine.start()
    try:
        engine.handle_audio_delta(source, "response")
        wait_until(lambda: len(media.push_history) >= 4)
        echoed = np.concatenate(media.push_history[-3:])
        media.feed_mono(echoed)
        wait_until(lambda: media.input.empty())
        time.sleep(0.05)

        assert engine.state == "speaking"
        assert engine.metrics["interruptions"] == 0
        assert engine.metrics["self_echo_frames_suppressed"] >= 3

        near_end = (0.2 * np.sin(2.0 * np.pi * 173.0 * np.arange(3 * 320) / 16_000)).astype(
            np.float32
        )
        media.feed_mono(near_end)
        wait_until(lambda: engine.state == "interrupted")
    finally:
        engine.stop()


def test_echo_guard_matches_non_quantized_device_delay() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(0.0, 0.15, 8 * 320 + 47).astype(np.float32)
    reference = np.convolve(raw, np.ones(9, dtype=np.float32) / 9.0, mode="valid")
    guard = _FarEndEchoGuard()
    for offset in range(0, 8 * 320, 320):
        guard.remember(reference[offset : offset + 320])

    assert guard.matches(reference[47 : 47 + 320])
    assert not guard.matches(rng.normal(0.0, 0.15, 320).astype(np.float32))


def test_correlated_far_end_audio_is_removed_from_model_uplink() -> None:
    media = FakeMedia()
    engine = make_engine(media, playback_lead_seconds=0.020)
    source = (0.25 * np.sin(2.0 * np.pi * 613.0 * np.arange(2_400) / 24_000)).astype(np.float32)
    engine.start()
    try:
        engine.handle_audio_delta(source, "response")
        wait_until(lambda: len(media.push_history) >= 5)
        echo_frame = media.push_history[-2]
        media.feed_mono(np.tile(echo_frame, 50))

        unit = engine.next_audio_unit(1.0)
        assert unit is not None
        np.testing.assert_array_equal(unit[0], np.zeros(16_000, dtype=np.float32))
        assert engine.metrics["self_echo_frames_suppressed"] >= 50
        assert engine.metrics["interruptions"] == 0
    finally:
        engine.stop()


def test_playback_uses_absolute_sample_clock_despite_push_overhead() -> None:
    media = FakeMedia(push_delay=0.005)
    engine = make_engine(media, playback_lead_seconds=0.020)
    source = (0.2 * np.sin(2.0 * np.pi * 440.0 * np.arange(24_000) / 24_000)).astype(np.float32)
    engine.start()
    try:
        engine.handle_audio_delta(source, "response")
        engine.handle_listen("listen")
        wait_until(lambda: len(media.push_timestamps) >= 50, timeout=2.0)

        elapsed = media.push_timestamps[49] - media.push_timestamps[0]
        assert 0.85 <= elapsed <= 1.10
    finally:
        engine.stop()


def test_streaming_resampler_is_continuous_across_arbitrary_deltas() -> None:
    source = np.sin(2.0 * np.pi * 733.0 * np.arange(2_403) / 24_000).astype(np.float32)
    whole_resampler = StreamingResampler24To16()
    split_resampler = StreamingResampler24To16()

    whole = whole_resampler.process(source)
    offsets = (0, 1, 19, 240, 241, 1_777, source.size)
    pieces = [
        split_resampler.process(source[start:end])
        for start, end in zip(offsets, offsets[1:], strict=False)
    ]
    split = np.concatenate(pieces)

    assert whole.shape == split.shape
    np.testing.assert_allclose(split, whole, rtol=0.0, atol=1e-7)


def test_slow_camera_never_blocks_audio_and_latest_frame_is_a_snapshot() -> None:
    media = FakeMedia(camera_delay=0.12)
    engine = AudioEngine(
        FakeMini(media),
        vad=EnergyVad(),
        camera_fps=10.0,
        playback_lead_seconds=0.060,
    )
    engine.start()
    try:
        media.feed_mono(np.zeros(16_000, dtype=np.float32))
        unit = engine.next_audio_unit(0.5)
        assert unit is not None

        started = time.perf_counter()
        snapshot = engine.latest_frame_jpeg()
        elapsed = time.perf_counter() - started
        assert elapsed < 0.02
        if snapshot is None:
            wait_until(lambda: engine.latest_frame_jpeg() == b"jpeg")
        else:
            assert snapshot == b"jpeg"
    finally:
        engine.stop()


def test_clear_failure_is_observable_and_stop_joins_every_worker() -> None:
    media = FakeMedia(fail_clear=True)
    errors: list[BaseException] = []
    engine = AudioEngine(
        FakeMini(media),
        vad=EnergyVad(),
        capture_video=False,
        error_callback=errors.append,
    )
    engine.start()
    wait_until(lambda: engine.metrics["clear_player_failures"] >= 1)
    wait_until(lambda: bool(errors))
    threads = engine.worker_threads

    assert isinstance(engine.last_error, PlayerClearError)
    assert any(isinstance(error, PlayerClearError) for error in errors)
    engine.stop()

    assert not media.recording
    assert not media.playing
    assert all(not thread.is_alive() for thread in threads)


def test_response_ids_are_slices_and_ordinary_listen_preserves_tail() -> None:
    media = FakeMedia()
    engine = make_engine(media, playback_lead_seconds=0.060)
    engine.start()
    try:
        wait_until(lambda: media.audio.clear_calls >= 1)
        initial_clears = media.audio.clear_calls

        engine.handle_audio_delta(np.full(2_400, 0.15, dtype=np.float32), "slice-1")
        engine.handle_audio_delta(np.full(120, 0.15, dtype=np.float32), "slice-2")
        engine.handle_listen("listen-boundary")

        wait_until(lambda: len(media.push_history) >= 6)
        assert media.audio.clear_calls == initial_clears
        assert engine.metrics["playback_frames_pushed"] >= 6
        # 2,520 samples at 24 kHz become 1,680 at 16 kHz: five complete
        # frames plus an 80-sample tail padded only at the natural boundary.
        assert np.count_nonzero(media.push_history[5][80:]) == 0
    finally:
        engine.stop()


def test_xvf_config_is_applied_after_both_media_directions_start() -> None:
    media = FakeMedia()
    engine = make_engine(media)
    engine.start()
    try:
        assert media.events[:4] == [
            "start_recording",
            "start_playing",
            "bound_output",
            "configure",
        ]
        assert media.audio.max_output_buffers == [6]
        assert media.audio.config_calls == [(AUDIO_STARTUP_CONFIG, True, 0.1)]
        assert engine.metrics["audio_config_successes"] == 1
    finally:
        engine.stop()


def test_server_wall_clock_metrics_are_retained_for_field_diagnosis() -> None:
    media = FakeMedia()
    engine = make_engine(media, playback_lead_seconds=0.060)
    engine.start()
    try:
        engine.handle_audio_delta(
            np.full(2_400, 0.1, dtype=np.float32),
            "slice",
            {"wall_clock_ms": 1_250.0},
        )
        wait_until(lambda: engine.metrics["playback_frames_pushed"] > 0)
        snapshot = engine.metrics
        assert snapshot["last_server_wall_clock_ms"] == 1_250.0
        assert snapshot["max_server_wall_clock_ms"] == 1_250.0
        assert snapshot["slow_server_events"] == 1
    finally:
        engine.stop()


def test_session_invalidation_discards_capture_partial_before_reconnect() -> None:
    media = FakeMedia()
    engine = make_engine(media)
    engine.start()
    try:
        media.feed_mono(np.full(8_000, 0.001, dtype=np.float32))
        wait_until(media.input.empty)
        time.sleep(0.02)

        engine.invalidate_session("reconnect")
        media.feed_mono(np.full(8_000, 0.002, dtype=np.float32))
        assert engine.next_audio_unit(0.05) is None
        engine.handle_session_ready()
        media.feed_mono(np.full(8_000, 0.002, dtype=np.float32))
        media.feed_mono(np.full(8_000, 0.002, dtype=np.float32))

        unit = engine.next_audio_unit(0.5)
        assert unit is not None
        np.testing.assert_allclose(unit[0], 0.002, rtol=0.0, atol=1e-7)
    finally:
        engine.stop()


def test_natural_listen_keeps_playout_interruptible_until_speaker_drains() -> None:
    media = FakeMedia()
    engine = make_engine(
        media,
        playback_lead_seconds=0.060,
        playback_queue_seconds=3.0,
    )
    engine.start()
    try:
        wait_until(lambda: media.audio.clear_calls >= 1)
        initial_clears = media.audio.clear_calls
        engine.handle_audio_delta(np.full(48_000, 0.2, dtype=np.float32), "response")
        engine.handle_listen("listen-boundary")

        wait_until(lambda: len(media.push_history) > 0)
        assert engine.state == "speaking"
        media.feed_mono(np.full(3 * 320, 0.12, dtype=np.float32))

        wait_until(lambda: engine.state == "interrupted")
        wait_until(lambda: media.audio.clear_calls > initial_clears)
        with media.output_lock:
            assert media.audible == []
    finally:
        engine.stop()


def test_playback_overflow_preserves_the_start_of_a_response() -> None:
    media = FakeMedia()
    engine = make_engine(
        media,
        playback_lead_seconds=0.150,
        playback_queue_seconds=1.0,
    )
    source = np.linspace(-0.7, 0.7, 48_000, dtype=np.float32)
    expected = StreamingResampler24To16().process(source)[:320]
    engine.start()
    try:
        engine.handle_audio_delta(source, "long-response")
        enqueued_at_overflow = engine.metrics["playback_frames_enqueued"]
        engine.handle_audio_delta(np.full(4_800, 0.6, dtype=np.float32), "later-slice")
        wait_until(lambda: len(media.push_history) > 0)
        np.testing.assert_allclose(media.push_history[0], expected, rtol=0.0, atol=1e-7)
        assert engine.metrics["playback_dropped_frames"] > 0
        assert engine.metrics["playback_overflowed_turns"] == 1
        assert engine.metrics["playback_frames_enqueued"] == enqueued_at_overflow
    finally:
        engine.stop()
