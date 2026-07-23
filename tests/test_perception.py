from __future__ import annotations

import io
import math
import threading

import numpy as np
from PIL import Image

from yrobot.perception import (
    DoATracker,
    DoAWorker,
    LatestFrame,
    doa_to_world_yaw,
    doa_to_yaw,
    encode_bgr_jpeg,
)


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


def test_doa_worker_rejects_hardware_only_self_voice_during_playback() -> None:
    class Media:
        def __init__(self) -> None:
            self.polled = threading.Event()

        def get_DoA(self) -> tuple[float, bool]:  # noqa: N802
            self.polled.set()
            return (0.0, True)

    self_voice_media = Media()
    rejected = DoATracker()
    worker = DoAWorker(
        self_voice_media,
        rejected,
        lambda: False,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: True,
        hz=20,
    )
    worker.start()
    assert self_voice_media.polled.wait(1.0)
    assert worker.stop()
    assert rejected.snapshot().active is False

    human_media = Media()
    accepted = DoATracker()
    worker = DoAWorker(
        human_media,
        accepted,
        lambda: True,
        head_pose=lambda: np.eye(4),
        playback_active=lambda: True,
        hz=20,
    )
    worker.start()
    assert human_media.polled.wait(1.0)
    assert worker.stop()
    assert accepted.snapshot().active is True
