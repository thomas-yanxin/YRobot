from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from yrobot.audio import (
    AudioCaptureWorker,
    AudioUnitizer,
    EchoReference,
    FrameSplitter,
    NearEndDetector,
    PlaybackEngine,
    PlaybackPacket,
    StreamingResampler,
    mono_capture,
)


def wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("condition was not reached before timeout")


class AlwaysSpeech:
    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        return True


class NeverSpeech:
    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        return False


class FakeAudioBackend:
    def __init__(self, operations: list[str] | None = None) -> None:
        self.max_buffers: int | None = None
        self.clears = 0
        self.operations = operations

    def set_max_output_buffers(self, value: int) -> None:
        self.max_buffers = value

    def clear_player(self) -> None:
        self.clears += 1
        if self.operations is not None:
            self.operations.append("clear")


class FakeMedia:
    def __init__(self) -> None:
        self.audio = FakeAudioBackend()
        self.pushed: list[np.ndarray] = []

    def push_audio_sample(self, samples: np.ndarray) -> None:
        self.pushed.append(samples.copy())


class BlockingMedia:
    def __init__(self) -> None:
        self.operations: list[str] = []
        self.audio = FakeAudioBackend(self.operations)
        self.push_started = threading.Event()
        self.release_push = threading.Event()

    def push_audio_sample(self, samples: np.ndarray) -> None:
        self.operations.append("push-start")
        self.push_started.set()
        assert self.release_push.wait(1.0)
        self.operations.append("push-end")


def test_capture_normalization_splitter_and_exact_f32le_unit() -> None:
    channels_last = np.column_stack(
        (
            np.linspace(-0.5, 0.5, 16_000, dtype=np.float32),
            np.ones(16_000, dtype=np.float32),
        )
    )
    channel_zero = mono_capture(channels_last, channel=0)
    channels_first = mono_capture(channels_last.T, channel=-1)

    assert channel_zero.shape == (16_000,)
    np.testing.assert_allclose(channels_first, channels_last.mean(axis=1))

    splitter = FrameSplitter()
    assert splitter.push(channel_zero[:319]) == []
    frames = splitter.push(channel_zero[319:961])
    assert [frame.shape for frame in frames] == [(320,), (320,), (320,)]
    assert splitter.pending_samples == 1

    unitizer = AudioUnitizer()
    assert unitizer.push(channel_zero[:7_999], captured_at=1.0) == []
    units = unitizer.push(channel_zero[7_999:], captured_at=2.0)
    assert len(units) == 1
    assert units[0].sequence == 0
    assert units[0].captured_at == 2.0
    assert units[0].samples.shape == (16_000,)
    assert len(units[0].f32le) == 16_000 * 4
    np.testing.assert_array_equal(np.frombuffer(units[0].f32le, dtype="<f4"), channel_zero)


def test_echo_is_suppressed_and_barge_in_requires_80_ms_near_end() -> None:
    rng = np.random.default_rng(9)
    played = rng.normal(0.0, 0.08, 16_000).astype(np.float32)
    echo_frames = [
        played[start : start + 320] * np.float32(0.2) for start in range(4_000, 5_600, 320)
    ]
    human_frame = rng.normal(0.0, 0.08, 320).astype(np.float32)
    reference = EchoReference()
    reference.append_played(played)
    detector = NearEndDetector(
        min_rms=0.001,
        echo_correlation=0.7,
        echo_reference=reference,
        vad=AlwaysSpeech(),
    )

    echo_decisions = [
        detector.process(
            echo_frame,
            output_active=True,
            timestamp=index * 0.02,
        )
        for index, echo_frame in enumerate(echo_frames)
    ]
    assert all(decision.echo_like for decision in echo_decisions)
    assert not any(decision.near_end for decision in echo_decisions)
    assert not any(decision.barge_in for decision in echo_decisions)

    detector.reset()
    human_decisions = [
        detector.process(
            human_frame,
            output_active=True,
            timestamp=0.1 + index * 0.02,
        )
        for index in range(6)
    ]
    assert not any(decision.current_near_end for decision in human_decisions[:3])
    assert human_decisions[3].current_near_end
    assert human_decisions[3].near_end
    assert not any(decision.barge_in for decision in human_decisions[:3])
    assert human_decisions[3].barge_in
    assert not any(decision.barge_in for decision in human_decisions[4:])
    held = detector.process(
        np.zeros(320, dtype=np.float32),
        output_active=True,
        timestamp=0.24,
    )
    assert held.current_near_end is False
    assert held.near_end is True


def test_echo_tail_filters_reference_without_triggering_barge_in() -> None:
    detector = NearEndDetector(
        min_rms=0.001,
        vad=AlwaysSpeech(),
    )
    human_frame = np.full(320, 0.08, dtype=np.float32)

    decisions = [
        detector.process(
            human_frame,
            output_active=False,
            echo_guard_active=True,
            timestamp=index * 0.02,
        )
        for index in range(6)
    ]

    assert decisions[-1].current_near_end is True
    assert decisions[-1].echo_guard_active is True
    assert not any(decision.barge_in for decision in decisions)


def test_filtered_residual_echo_near_point_five_does_not_cut_off_playback() -> None:
    rng = np.random.default_rng(22)
    played = rng.normal(0.0, 0.08, 16_000).astype(np.float32)
    segment = played[4_000:5_280]
    room_filter = np.array(
        [
            0.085,
            0.122,
            0.102,
            0.093,
            0.102,
            0.052,
            -0.020,
            -0.023,
            -0.072,
            -0.084,
            -0.163,
            -0.082,
        ],
        dtype=np.float32,
    )
    filtered = np.convolve(
        segment,
        room_filter,
        mode="full",
    )[:1_280].astype(np.float32)
    residual_echo = filtered * np.float32(0.3)
    reference = EchoReference()
    reference.append_played(played)
    detector = NearEndDetector(
        min_rms=0.001,
        echo_reference=reference,
        vad=AlwaysSpeech(),
    )

    decisions = [
        detector.process(
            residual_echo[start : start + 320],
            output_active=True,
            timestamp=index * 0.02,
        )
        for index, start in enumerate(range(0, 1_280, 320))
    ]

    assert 0.48 <= decisions[-1].echo_similarity <= 0.55
    assert decisions[-1].echo_fit >= 0.82
    assert decisions[-1].echo_like
    assert not any(decision.barge_in for decision in decisions)


def test_human_double_talk_is_not_swallowed_by_echo_guard() -> None:
    rng = np.random.default_rng(99)
    played = rng.normal(0.0, 0.08, 16_000).astype(np.float32)
    echo = played[4_000:5_280] * np.float32(0.2)
    human = (
        np.random.default_rng(100)
        .normal(
            0.0,
            float(np.std(echo)),
            echo.size,
        )
        .astype(np.float32)
    )
    mixed = echo + human
    reference = EchoReference()
    reference.append_played(played)
    detector = NearEndDetector(
        min_rms=0.001,
        echo_reference=reference,
        vad=AlwaysSpeech(),
    )

    decisions = [
        detector.process(
            mixed[start : start + 320],
            output_active=True,
            timestamp=index * 0.02,
        )
        for index, start in enumerate(range(0, 1_280, 320))
    ]

    assert decisions[-1].echo_similarity >= 0.65
    assert decisions[-1].echo_fit < 0.82
    assert not decisions[-1].echo_like
    assert decisions[-1].barge_in


def test_playback_resamples_mono_and_rejects_stale_epoch_after_flush() -> None:
    media = FakeMedia()
    epoch = [4]
    reference = EchoReference()
    engine = PlaybackEngine(
        media,
        lambda: epoch[0],
        reference,
        preroll_ms=0,
        output_chunk_ms=1_000,
        max_ahead_ms=1_000,
    )
    engine.start()
    assert media.audio.max_buffers == 0

    packet = PlaybackPacket(4, np.linspace(-0.2, 0.2, 2_400, dtype=np.float32))
    assert engine.enqueue(packet)
    wait_for(lambda: len(media.pushed) == 1)
    assert media.pushed[0].shape == (1_600,)
    assert media.pushed[0].ndim == 1
    assert media.pushed[0].dtype == np.float32
    assert engine.output_active()

    epoch[0] = 5
    assert engine.interrupt(5)
    assert media.audio.clears == 1
    assert not engine.output_active()
    assert engine.echo_guard_active()
    assert not engine.enqueue(packet)
    assert reference.similarity(media.pushed[0][:320]) > 0.7
    assert engine.stop(flush=False)


def test_streaming_resampler_is_continuous_across_server_delta_boundaries() -> None:
    samples = np.sin(2 * np.pi * 1_000 * np.arange(7_201, dtype=np.float32) / 24_000).astype(
        np.float32
    )
    whole = StreamingResampler(24_000, 16_000).process(samples)
    split_resampler = StreamingResampler(24_000, 16_000)
    split = np.concatenate(
        [
            split_resampler.process(samples[:997]),
            split_resampler.process(samples[997:3_511]),
            split_resampler.process(samples[3_511:]),
        ]
    )

    np.testing.assert_allclose(split, whole, atol=1e-6)


def test_playback_logs_receive_to_first_push_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [5.0]
    media = FakeMedia()
    engine = PlaybackEngine(
        media,
        lambda: 1,
        EchoReference(),
        input_sample_rate=16_000,
        output_sample_rate=16_000,
        clock=lambda: now[0],
    )
    caplog.set_level(logging.INFO, logger="yrobot.audio")
    engine.start()

    packet = PlaybackPacket(
        1,
        np.ones(320, dtype=np.float32),
        "response-1",
        received_at=4.975,
    )
    assert engine.enqueue(packet)
    assert engine.enqueue(packet)
    wait_for(lambda: len(media.pushed) == 2)
    assert engine.stop(flush=False)

    logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Reachy first audio push:")
    ]
    assert len(logs) == 1
    assert "raw_to_enqueue_ms=25.0" in logs[0]
    assert "enqueue_to_push_ms=0.0" in logs[0]


def test_preroll_is_applied_again_after_each_listen_boundary() -> None:
    now = [1.0]
    media = FakeMedia()
    engine = PlaybackEngine(
        media,
        lambda: 1,
        EchoReference(),
        input_sample_rate=1_000,
        output_sample_rate=1_000,
        preroll_ms=100,
        output_chunk_ms=100,
        max_ahead_ms=100,
        clock=lambda: now[0],
    )
    engine.start()

    first = PlaybackPacket(1, np.ones(50, dtype=np.float32), "response-1")
    assert engine.enqueue(first)
    time.sleep(0.02)
    assert media.pushed == []
    now[0] = 1.1
    wait_for(lambda: len(media.pushed) == 1)

    engine.mark_response_boundary()
    second = PlaybackPacket(1, np.ones(50, dtype=np.float32), "response-2")
    assert engine.enqueue(second)
    time.sleep(0.02)
    assert len(media.pushed) == 1
    now[0] = 1.201
    wait_for(lambda: len(media.pushed) == 2)
    assert engine.stop(flush=False)


def test_playback_resampler_resets_at_response_and_listen_boundaries() -> None:
    media = FakeMedia()
    engine = PlaybackEngine(
        media,
        lambda: 1,
        EchoReference(),
        preroll_ms=0,
        output_chunk_ms=1_000,
        max_ahead_ms=1_000,
    )
    first = np.full(2_400, 0.25, dtype=np.float32)
    second = np.linspace(-0.2, 0.2, 2_400, dtype=np.float32)
    expected = StreamingResampler(24_000, 16_000).process(second)

    engine.start()
    assert engine.enqueue(PlaybackPacket(1, first, "response-1"))
    wait_for(lambda: len(media.pushed) == 1)
    assert engine.enqueue(PlaybackPacket(1, second, "response-2"))
    wait_for(lambda: len(media.pushed) == 2)
    np.testing.assert_allclose(media.pushed[1], expected, atol=1e-6)

    engine.mark_response_boundary()
    assert engine.enqueue(PlaybackPacket(1, second, "response-2"))
    wait_for(lambda: len(media.pushed) == 3)
    np.testing.assert_allclose(media.pushed[2], expected, atol=1e-6)
    assert engine.stop(flush=False)


def test_playback_fifo_preserves_every_server_delta_in_order() -> None:
    media = FakeMedia()
    reference = EchoReference()
    engine = PlaybackEngine(
        media,
        lambda: 1,
        reference,
        input_sample_rate=1_000,
        output_sample_rate=1_000,
        preroll_ms=0,
        output_chunk_ms=1_000,
        max_ahead_ms=10_000,
    )
    values = (0.1, 0.2, 0.3, 0.4, 0.5)
    for value in values:
        assert engine.enqueue(PlaybackPacket(1, np.full(1_000, value, dtype=np.float32)))

    assert engine.stats().dropped == 0
    engine.start()
    wait_for(lambda: len(media.pushed) == len(values))
    assert media.audio.max_buffers == 0
    assert [round(float(chunk.mean()), 1) for chunk in media.pushed] == list(values)
    assert engine.stats().dropped == 0
    assert engine.stop(flush=False)


def test_default_playout_pacing_preserves_every_sample() -> None:
    media = FakeMedia()
    engine = PlaybackEngine(
        media,
        lambda: 1,
        EchoReference(sample_rate=1_000),
        input_sample_rate=1_000,
        output_sample_rate=1_000,
    )
    expected = np.concatenate(
        [
            np.full(100, 0.1, dtype=np.float32),
            np.full(100, 0.2, dtype=np.float32),
            np.full(100, 0.3, dtype=np.float32),
        ]
    )
    for packet in np.split(expected, 3):
        assert engine.enqueue(PlaybackPacket(1, packet))

    engine.start()
    wait_for(lambda: sum(chunk.size for chunk in media.pushed) == expected.size)
    np.testing.assert_array_equal(np.concatenate(media.pushed), expected)
    assert engine.stats().dropped == 0
    assert engine.stop(flush=False)


def test_interrupt_is_ordered_after_inflight_push_and_old_audio_cannot_revive() -> None:
    media = BlockingMedia()
    epoch = [7]
    engine = PlaybackEngine(
        media,
        lambda: epoch[0],
        EchoReference(),
        input_sample_rate=16_000,
        output_sample_rate=16_000,
        preroll_ms=0,
    )
    engine.start()
    assert engine.enqueue(PlaybackPacket(7, np.ones(320, dtype=np.float32)))
    assert media.push_started.wait(1.0)

    epoch[0] = 8
    interrupt = threading.Thread(target=engine.interrupt, args=(8,))
    interrupt.start()
    time.sleep(0.01)
    assert "clear" not in media.operations
    media.release_push.set()
    interrupt.join(1.0)
    assert not interrupt.is_alive()
    assert media.operations == ["push-start", "push-end", "clear"]
    assert not engine.enqueue(PlaybackPacket(7, np.ones(320, dtype=np.float32)))
    assert engine.stop(flush=False)


def test_capture_worker_emits_units_and_frames_without_owning_media_lifecycle() -> None:
    class CaptureMedia:
        def __init__(self) -> None:
            half = np.zeros((8_000, 2), dtype=np.float32)
            self.chunks: list[np.ndarray | None] = [half, half]

        def get_audio_sample(self) -> np.ndarray | None:
            return self.chunks.pop(0) if self.chunks else None

    units = []
    voices = []
    got_unit = threading.Event()

    def on_unit(unit: object) -> None:
        units.append(unit)
        got_unit.set()

    worker = AudioCaptureWorker(
        CaptureMedia(),
        channel=0,
        detector=NearEndDetector(vad=NeverSpeech()),
        output_active=lambda: False,
        on_unit=on_unit,
        on_voice=voices.append,
    )
    worker.start()
    assert got_unit.wait(1.0)
    assert worker.stop()
    assert len(units) == 1
    assert len(voices) == 50
