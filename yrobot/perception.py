"""Low-latency camera and sound-direction workers.

The workers in this module never command motors and never share unbounded queues.
Camera consumers always receive the newest JPEG; DoA consumers receive a smoothed
attention yaw.
"""

from __future__ import annotations

import io
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from PIL import Image

LOGGER = logging.getLogger(__name__)


class PerceptionMedia(Protocol):
    """Subset of Reachy Mini's local ``MediaManager`` used here."""

    def get_frame(self) -> npt.NDArray[np.uint8] | None: ...

    def get_DoA(self) -> tuple[float, bool] | None: ...  # noqa: N802


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    jpeg: bytes
    captured_at: float
    sequence: int


class LatestFrame:
    """Single-slot, thread-safe JPEG store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: FrameSnapshot | None = None
        self._sequence = 0

    def publish(self, jpeg: bytes, *, captured_at: float | None = None) -> None:
        if not jpeg:
            return
        timestamp = time.monotonic() if captured_at is None else captured_at
        with self._lock:
            self._sequence += 1
            self._snapshot = FrameSnapshot(bytes(jpeg), timestamp, self._sequence)

    def snapshot(
        self,
        *,
        max_age_seconds: float | None = None,
        now: float | None = None,
    ) -> FrameSnapshot | None:
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None or max_age_seconds is None:
            return snapshot
        current = time.monotonic() if now is None else now
        return snapshot if current - snapshot.captured_at <= max_age_seconds else None


def encode_bgr_jpeg(
    frame: npt.NDArray[np.uint8],
    *,
    width: int = 640,
    quality: int = 72,
) -> bytes:
    """Convert an SDK BGR frame to a width-bounded RGB JPEG."""

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("camera frame must have shape (height, width, 3)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("camera frame must not be empty")
    if width <= 0:
        raise ValueError("JPEG width must be positive")
    if not 1 <= quality <= 95:
        raise ValueError("JPEG quality must be in 1..95")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(array[:, :, ::-1])
    image = Image.fromarray(rgb, mode="RGB")
    if image.width != width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=False, subsampling=2)
    return output.getvalue()


class CameraWorker:
    """Capture and encode at a low fixed rate without blocking audio."""

    def __init__(
        self,
        media: PerceptionMedia,
        latest: LatestFrame,
        *,
        width: int = 640,
        quality: int = 72,
        fps: float = 1.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("camera fps must be positive")
        self._media = media
        self._latest = latest
        self._width = width
        self._quality = quality
        self._period = 1.0 / fps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="yrobot-camera",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.is_alive

    def _run(self) -> None:
        deadline = time.monotonic()
        last_warning = -math.inf
        while not self._stop.is_set():
            now = time.monotonic()
            if now < deadline:
                self._stop.wait(deadline - now)
                continue
            try:
                frame = self._media.get_frame()
                if frame is not None:
                    captured_at = time.monotonic()
                    jpeg = encode_bgr_jpeg(
                        frame,
                        width=self._width,
                        quality=self._quality,
                    )
                    self._latest.publish(jpeg, captured_at=captured_at)
            except Exception:
                if now - last_warning >= 5.0:
                    LOGGER.warning("camera capture failed", exc_info=True)
                    last_warning = now

            deadline += self._period
            current = time.monotonic()
            if deadline <= current:
                skipped = math.floor((current - deadline) / self._period) + 1
                deadline += skipped * self._period


def _wrap_radians(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


def _circular_median(values: tuple[float, ...]) -> float:
    return min(
        values,
        key=lambda candidate: sum(abs(_wrap_radians(value - candidate)) for value in values),
    )


def doa_to_yaw(angle_radians: float) -> float:
    """Map XVF3800 DoA to Reachy yaw: left=+π/2, center=0, right=-π/2.

    The microphone array is linear, so its π/2 reading cannot distinguish sound
    arriving from the front from sound arriving from the back.
    """

    if not math.isfinite(angle_radians):
        raise ValueError("DoA angle must be finite")
    if not 0.0 <= angle_radians <= math.pi:
        raise ValueError("DoA angle must be within [0, pi]")
    return math.pi / 2.0 - angle_radians


def doa_to_world_yaw(
    angle_radians: float,
    head_pose: npt.ArrayLike,
) -> float:
    """Transform the head-relative DoA vector with Reachy's measured pose."""

    relative_yaw = doa_to_yaw(angle_radians)
    pose = np.asarray(head_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("head pose must be a finite 4x4 transform")
    direction_head = np.array(
        [math.cos(relative_yaw), math.sin(relative_yaw), 0.0],
        dtype=np.float64,
    )
    direction_world = pose[:3, :3] @ direction_head
    if math.hypot(float(direction_world[0]), float(direction_world[1])) < 1e-6:
        raise ValueError("DoA direction has no stable horizontal projection")
    return _wrap_radians(
        math.atan2(
            float(direction_world[1]),
            float(direction_world[0]),
        )
    )


@dataclass(frozen=True, slots=True)
class DoASnapshot:
    yaw_radians: float
    confidence: float
    active: bool
    age_seconds: float | None


class DoATracker:
    """Robust world-yaw smoothing with a short post-speech attention hold."""

    def __init__(
        self,
        *,
        hold_seconds: float = 3.0,
        smoothing_seconds: float = 0.18,
        release_seconds: float = 1.0,
        median_window: int = 3,
    ) -> None:
        if hold_seconds < 0 or smoothing_seconds <= 0 or release_seconds <= 0:
            raise ValueError("DoA timing values must be positive")
        if median_window < 1 or median_window % 2 == 0:
            raise ValueError("DoA median window must be a positive odd number")

        self._hold_seconds = hold_seconds
        self._smoothing_seconds = smoothing_seconds
        self._release_seconds = release_seconds
        self._samples: deque[float] = deque(maxlen=median_window)
        self._lock = threading.Lock()
        self._yaw: float | None = None
        self._last_update: float | None = None
        self._last_voice: float | None = None
        self._confidence = 0.0

    def update(
        self,
        angle_radians: float,
        *,
        hardware_speech: bool,
        near_end_speech: bool,
        head_pose: npt.ArrayLike,
        now: float | None = None,
    ) -> bool:
        """Accept a head-relative DoA sample and transform it to world yaw."""

        if not (hardware_speech or near_end_speech):
            return False
        if not math.isfinite(angle_radians):
            return False
        # Small firmware boundary noise is harmless; grossly invalid readings are not.
        if angle_radians < -0.05 or angle_radians > math.pi + 0.05:
            return False

        timestamp = time.monotonic() if now is None else now
        try:
            yaw = doa_to_world_yaw(
                min(math.pi, max(0.0, angle_radians)),
                head_pose,
            )
        except (TypeError, ValueError):
            return False
        with self._lock:
            if self._last_voice is not None and timestamp - self._last_voice > self._hold_seconds:
                self._samples.clear()
                self._confidence = 0.0
            self._samples.append(yaw)
            robust_yaw = _circular_median(tuple(self._samples))
            if self._yaw is None:
                self._yaw = robust_yaw
            else:
                previous = self._last_update
                elapsed = 0.1 if previous is None else max(0.001, timestamp - previous)
                alpha = 1.0 - math.exp(-elapsed / self._smoothing_seconds)
                self._yaw = _wrap_radians(self._yaw + alpha * _wrap_radians(robust_yaw - self._yaw))
            self._last_update = timestamp
            self._last_voice = timestamp
            self._confidence = min(1.0, self._confidence + 0.25)
        return True

    def snapshot(self, *, now: float | None = None) -> DoASnapshot:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            yaw = self._yaw
            last_voice = self._last_voice
            confidence = self._confidence

        if yaw is None or last_voice is None:
            return DoASnapshot(0.0, 0.0, False, None)
        age = max(0.0, timestamp - last_voice)
        if age <= self._hold_seconds:
            return DoASnapshot(yaw, confidence, True, age)

        decay = math.exp(-(age - self._hold_seconds) / self._release_seconds)
        return DoASnapshot(yaw * decay, confidence * decay, False, age)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._yaw = None
            self._last_update = None
            self._last_voice = None
            self._confidence = 0.0


class DoAWorker:
    """Poll the XVF3800 at 10–20 Hz; publish state, never motor commands."""

    def __init__(
        self,
        media: PerceptionMedia,
        tracker: DoATracker,
        near_end_speech: Callable[[], bool],
        *,
        head_pose: Callable[[], npt.ArrayLike],
        playback_active: Callable[[], bool],
        hz: float = 10.0,
    ) -> None:
        if not 10.0 <= hz <= 20.0:
            raise ValueError("DoA polling rate must be within 10..20 Hz")
        self._media = media
        self._tracker = tracker
        self._near_end_speech = near_end_speech
        self._head_pose = head_pose
        self._playback_active = playback_active
        self._period = 1.0 / hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="yrobot-doa",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.is_alive

    def _run(self) -> None:
        deadline = time.monotonic()
        last_warning = -math.inf
        while not self._stop.is_set():
            now = time.monotonic()
            if now < deadline:
                self._stop.wait(deadline - now)
                continue
            try:
                result = self._media.get_DoA()
                if result is not None:
                    angle, hardware_speech = result
                    near_end_speech = bool(self._near_end_speech())
                    # The XVF speech bit can still react to residual loudspeaker
                    # energy. During playback, only the echo-guarded near-end VAD
                    # is trusted.
                    hardware_gate = bool(hardware_speech) and not bool(self._playback_active())
                    if hardware_gate or near_end_speech:
                        self._tracker.update(
                            float(angle),
                            hardware_speech=hardware_gate,
                            near_end_speech=near_end_speech,
                            head_pose=self._head_pose(),
                            now=now,
                        )
            except Exception:
                if now - last_warning >= 5.0:
                    LOGGER.warning("DoA polling failed", exc_info=True)
                    last_warning = now

            deadline += self._period
            current = time.monotonic()
            if deadline <= current:
                skipped = math.floor((current - deadline) / self._period) + 1
                deadline += skipped * self._period
