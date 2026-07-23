from __future__ import annotations

import io
import logging
import math
import threading
import time

import numpy as np
import pytest
import requests
from PIL import Image

from yrobot.perception import (
    DaemonDoASource,
    DoATracker,
    DoAWorker,
    LatestFrame,
    doa_to_world_yaw,
    doa_to_yaw,
    encode_bgr_jpeg,
)


class _Response:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True

    def json(self) -> object:
        return self._payload


class _Session:
    def __init__(self, payloads: list[object]) -> None:
        self._payloads = iter(payloads)
        self.requests: list[tuple[str, float]] = []
        self.responses: list[_Response] = []
        self.closed = False

    def get(self, url: str, *, timeout: float) -> _Response:
        self.requests.append((url, timeout))
        response = _Response(next(self._payloads))
        self.responses.append(response)
        return response

    def close(self) -> None:
        self.closed = True


def test_camera_frame_is_rgb_640px_jpeg_and_latest_only() -> None:
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    jpeg = encode_bgr_jpeg(bgr, width=640, quality=80)
    decoded = Image.open(io.BytesIO(jpeg))

    assert decoded.size == (640, 320)
    red, green, blue = decoded.getpixel((320, 160))
    assert red > 240 and green < 10 and blue < 10

    latest = LatestFrame()
    latest.publish(b"old", captured_at=1.0)
    latest.publish(jpeg, captured_at=2.0)
    assert latest.snapshot(now=2.5, max_age_seconds=1.0).jpeg == jpeg
    assert latest.snapshot(now=4.0, max_age_seconds=1.0) is None


def test_doa_mapping_and_world_yaw_smoothing() -> None:
    assert doa_to_yaw(0.0) == math.pi / 2
    assert doa_to_yaw(math.pi / 2) == 0.0
    assert doa_to_yaw(math.pi) == -math.pi / 2
    pose = np.eye(4)
    pose[:2, :2] = [
        [math.cos(0.2), -math.sin(0.2)],
        [math.sin(0.2), math.cos(0.2)],
    ]
    assert math.isclose(
        doa_to_world_yaw(math.pi / 2 - 0.3, pose),
        0.5,
    )

    tracker = DoATracker(
        hold_seconds=1.0,
        smoothing_seconds=0.1,
        release_seconds=1.0,
    )
    for index in range(5):
        assert tracker.update(
            math.pi / 2 - 0.3,
            hardware_speech=False,
            near_end_speech=True,
            head_pose=pose,
            now=index * 0.1,
        )

    active = tracker.snapshot(now=0.5)
    assert active.active
    assert active.confidence == 1.0
    assert math.isclose(active.yaw_radians, 0.5, abs_tol=0.02)

    released = tracker.snapshot(now=2.0)
    assert not released.active
    assert 0.0 < released.yaw_radians < active.yaw_radians


def test_doa_rejects_unvoiced_and_invalid_samples() -> None:
    tracker = DoATracker()

    assert not tracker.update(
        math.pi / 2,
        hardware_speech=False,
        near_end_speech=False,
        head_pose=np.eye(4),
    )
    assert not tracker.update(
        math.nan,
        hardware_speech=True,
        near_end_speech=False,
        head_pose=np.eye(4),
    )
    assert tracker.snapshot().active is False


def test_daemon_doa_source_reuses_session_and_parses_endpoint_payload() -> None:
    session = _Session(
        [
            {"angle": math.pi / 2, "speech_detected": True},
            None,
        ]
    )
    source = DaemonDoASource(
        "http://localhost:8000/api/state/doa",
        timeout_seconds=0.2,
        session=session,
    )

    assert source.read() == (math.pi / 2, True)
    assert source.read() is None
    assert session.requests == [
        ("http://localhost:8000/api/state/doa", 0.2),
        ("http://localhost:8000/api/state/doa", 0.2),
    ]
    assert all(response.raised for response in session.responses)

    source.close()
    source.close()
    assert session.closed
    with pytest.raises(RuntimeError, match="closed"):
        source.read()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"angle": "bad", "speech_detected": True},
        {"angle": "nan", "speech_detected": True},
        {"angle": 1.0, "speech_detected": 1},
    ],
)
def test_daemon_doa_source_rejects_invalid_payload(payload: object) -> None:
    source = DaemonDoASource(
        "http://localhost:8000/api/state/doa",
        session=_Session([payload]),
    )

    with pytest.raises(ValueError, match="DoA|invalid"):
        source.read()
    source.close()


def test_doa_worker_skips_self_playback_then_wakes_for_near_end_speech() -> None:
    class Source:
        def __init__(self) -> None:
            self.polled = threading.Event()
            self.closed = False

        def read(self) -> tuple[float, bool]:
            self.polled.set()
            return (0.0, True)

        def close(self) -> None:
            self.closed = True

    near_end_speech = threading.Event()
    source = Source()
    tracker = DoATracker()
    worker = DoAWorker(
        source,
        tracker,
        near_end_speech.is_set,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: True,
        hz=20,
    )
    worker.start()
    assert not source.polled.wait(0.15)
    near_end_speech.set()
    assert source.polled.wait(0.2)
    assert worker.stop()
    assert source.closed
    assert tracker.snapshot().active is True


def test_doa_worker_rechecks_playback_after_daemon_read() -> None:
    class Source:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def read(self) -> tuple[float, bool]:
            self.started.set()
            assert self.release.wait(1.0)
            return (0.0, True)

        def close(self) -> None: ...

    source = Source()
    playback = threading.Event()
    gate_rechecked = threading.Event()
    playback_checks = 0

    def playback_active() -> bool:
        nonlocal playback_checks
        playback_checks += 1
        if playback_checks >= 2:
            gate_rechecked.set()
        return playback.is_set()

    tracker = DoATracker()
    worker = DoAWorker(
        source,
        tracker,
        lambda: False,
        head_pose=lambda: np.eye(4),
        playback_active=playback_active,
    )

    worker.start()
    assert source.started.wait(1.0)
    playback.set()
    source.release.set()
    assert gate_rechecked.wait(1.0)
    assert worker.stop()

    assert tracker.snapshot().active is False


def test_doa_worker_uses_near_end_that_starts_during_daemon_read() -> None:
    class Source:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()

        def read(self) -> tuple[float, bool]:
            self.started.set()
            assert self.release.wait(1.0)
            return (math.pi / 2, False)

        def close(self) -> None: ...

    source = Source()
    near_end = threading.Event()
    head_pose_used = threading.Event()

    def head_pose() -> np.ndarray:
        head_pose_used.set()
        return np.eye(4)

    tracker = DoATracker()
    worker = DoAWorker(
        source,
        tracker,
        near_end.is_set,
        head_pose=head_pose,
        playback_active=lambda: False,
    )

    worker.start()
    assert source.started.wait(1.0)
    near_end.set()
    source.release.set()
    assert head_pose_used.wait(1.0)
    assert worker.stop()

    assert tracker.snapshot().active is True


def test_doa_worker_active_cadence_is_anchored_to_request_start() -> None:
    class Source:
        def __init__(self) -> None:
            self.calls = 0
            self.first_started = threading.Event()
            self.release_first = threading.Event()
            self.second_started = threading.Event()

        def read(self) -> tuple[float, bool]:
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
                assert self.release_first.wait(1.0)
            elif self.calls == 2:
                self.second_started.set()
            return (math.pi / 2, True)

        def close(self) -> None: ...

    source = Source()
    worker = DoAWorker(
        source,
        DoATracker(),
        lambda: True,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: False,
        hz=10,
        slow_request_seconds=1.0,
    )

    worker.start()
    assert source.first_started.wait(1.0)
    time.sleep(0.06)
    source.release_first.set()
    assert source.second_started.wait(0.07)
    assert worker.stop()


def test_doa_worker_backs_off_recovers_and_closes_source(caplog: pytest.LogCaptureFixture) -> None:
    class RecoveringSource:
        def __init__(self) -> None:
            self.calls: list[float] = []
            self.recovered = threading.Event()
            self.closed = False

        def read(self) -> tuple[float, bool]:
            self.calls.append(time.monotonic())
            if len(self.calls) <= 2:
                raise OSError("daemon temporarily unavailable")
            self.recovered.set()
            return (math.pi / 2, True)

        def close(self) -> None:
            self.closed = True

    caplog.set_level(logging.INFO)
    source = RecoveringSource()
    tracker = DoATracker()
    worker = DoAWorker(
        source,
        tracker,
        lambda: False,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: False,
        hz=20,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.02,
        warning_interval_seconds=1.0,
    )

    worker.start()
    assert source.recovered.wait(1.0)
    assert worker.stop()

    assert source.closed
    assert tracker.snapshot().active
    assert source.calls[1] - source.calls[0] >= 0.008
    assert source.calls[2] - source.calls[1] >= 0.018
    assert (
        caplog.messages.count(
            "DoA daemon polling failed (daemon temporarily unavailable); retrying in 0.01 s"
        )
        == 1
    )
    assert "DoA polling recovered after 2 failure(s)" in caplog.messages


def test_doa_worker_treats_daemon_null_as_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Source:
        def __init__(self) -> None:
            self.calls = 0
            self.recovered = threading.Event()

        def read(self) -> tuple[float, bool] | None:
            self.calls += 1
            if self.calls == 1:
                return None
            self.recovered.set()
            return (math.pi / 2, True)

        def close(self) -> None: ...

    caplog.set_level(logging.INFO)
    source = Source()
    worker = DoAWorker(
        source,
        DoATracker(),
        lambda: False,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: False,
        retry_initial_seconds=0.01,
        retry_max_seconds=0.01,
    )

    worker.start()
    assert source.recovered.wait(1.0)
    assert worker.stop()

    assert any("daemon returned null" in message for message in caplog.messages)
    assert "DoA polling recovered after 1 failure(s)" in caplog.messages


def test_doa_worker_read_timeout_opens_circuit_for_this_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Source:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = threading.Event()

        def read(self) -> tuple[float, bool]:
            self.calls += 1
            raise requests.exceptions.ReadTimeout("USB control transfer stalled")

        def close(self) -> None:
            self.closed.set()

    caplog.set_level(logging.INFO)
    source = Source()
    worker = DoAWorker(
        source,
        DoATracker(),
        lambda: False,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: False,
    )

    worker.start()
    assert source.closed.wait(1.0)
    assert worker.stop()

    assert source.calls == 1
    assert any("DoA disabled for this run" in message for message in caplog.messages)


def test_doa_worker_repeated_slow_reads_open_circuit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Source:
        def __init__(self) -> None:
            self.calls = 0
            self.closed = threading.Event()

        def read(self) -> tuple[float, bool]:
            self.calls += 1
            time.sleep(0.005)
            return (math.pi / 2, True)

        def close(self) -> None:
            self.closed.set()

    caplog.set_level(logging.INFO)
    source = Source()
    worker = DoAWorker(
        source,
        DoATracker(),
        lambda: False,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: False,
        hz=20,
        idle_hz=20,
        slow_request_seconds=0.001,
        slow_request_limit=3,
    )

    worker.start()
    assert source.closed.wait(1.0)
    assert worker.stop()

    assert source.calls == 3
    assert any("consecutive daemon reads exceeded" in message for message in caplog.messages)


def test_doa_worker_stop_before_start_closes_source() -> None:
    class Source:
        def __init__(self) -> None:
            self.closed = False

        def read(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    source = Source()
    worker = DoAWorker(
        source,
        DoATracker(),
        lambda: False,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: False,
    )

    assert worker.stop()
    assert source.closed
