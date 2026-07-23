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
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image

LOGGER = logging.getLogger(__name__)


class PerceptionMedia(Protocol):
    """Subset of Reachy Mini's local ``MediaManager`` used for the camera."""

    def get_frame(self) -> npt.NDArray[np.uint8] | None: ...


class DoASource(Protocol):
    """Bounded, closeable source of daemon-owned DoA samples."""

    def read(self) -> tuple[float, bool] | None: ...

    def close(self) -> None: ...


class DaemonDoASource:
    """Read DoA through the Reachy Mini daemon's persistent HTTP session."""

    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 0.25,
        session: Any | None = None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("DoA URL must use http:// or https://")
        if timeout_seconds <= 0:
            raise ValueError("DoA request timeout must be positive")
        self._url = url
        self._timeout_seconds = timeout_seconds
        if session is None:
            session = requests.Session()
            session.trust_env = False
        self._session = session
        self._closed = False

    def read(self) -> tuple[float, bool] | None:
        if self._closed:
            raise RuntimeError("DoA source is closed")
        response = self._session.get(self._url, timeout=self._timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("DoA response must be an object or null")
        try:
            angle = float(payload["angle"])
            speech_detected = payload["speech_detected"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid DoA response") from error
        if not math.isfinite(angle):
            raise ValueError("DoA angle must be finite")
        if not isinstance(speech_detected, bool):
            raise ValueError("DoA speech_detected must be a boolean")
        return angle, speech_detected

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.close()


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
    """Adaptively poll daemon-owned DoA; publish state, never motor commands."""

    def __init__(
        self,
        source: DoASource | str,
        tracker: DoATracker,
        near_end_speech: Callable[[], bool],
        *,
        head_pose: Callable[[], npt.ArrayLike],
        playback_active: Callable[[], bool],
        hz: float = 10.0,
        idle_hz: float = 2.0,
        request_timeout_seconds: float = 0.25,
        retry_initial_seconds: float = 0.25,
        retry_max_seconds: float = 2.0,
        warning_interval_seconds: float = 5.0,
        slow_request_seconds: float = 0.03,
        slow_request_limit: int = 3,
    ) -> None:
        if not math.isfinite(hz) or not 10.0 <= hz <= 20.0:
            raise ValueError("DoA polling rate must be within 10..20 Hz")
        if not math.isfinite(idle_hz) or not 0 < idle_hz <= hz:
            raise ValueError("DoA idle polling rate must be within 0..active Hz")
        if (
            not math.isfinite(retry_initial_seconds)
            or not math.isfinite(retry_max_seconds)
            or retry_initial_seconds <= 0
            or retry_max_seconds < retry_initial_seconds
        ):
            raise ValueError("DoA retry timings are invalid")
        if not math.isfinite(warning_interval_seconds) or warning_interval_seconds <= 0:
            raise ValueError("DoA warning interval must be positive")
        if not math.isfinite(slow_request_seconds) or slow_request_seconds <= 0:
            raise ValueError("DoA slow-request threshold must be positive")
        if slow_request_limit <= 0:
            raise ValueError("DoA slow-request limit must be positive")
        self._source = (
            DaemonDoASource(source, timeout_seconds=request_timeout_seconds)
            if isinstance(source, str)
            else source
        )
        self._tracker = tracker
        self._near_end_speech = near_end_speech
        self._head_pose = head_pose
        self._playback_active = playback_active
        self._active_period = 1.0 / hz
        self._idle_period = 1.0 / idle_hz
        self._retry_initial = retry_initial_seconds
        self._retry_max = retry_max_seconds
        self._warning_interval = warning_interval_seconds
        self._slow_request_seconds = slow_request_seconds
        self._slow_request_limit = slow_request_limit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        if self._closed:
            raise RuntimeError("DoA worker cannot be restarted after it is closed")
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
        stopped = not self.is_alive
        if stopped:
            self._close_source()
        return stopped

    def _close_source(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._source.close()
        except Exception:
            LOGGER.warning("failed to close DoA source", exc_info=True)

    def _run(self) -> None:
        last_warning = -math.inf
        last_processing_warning = -math.inf
        last_poll = -math.inf
        failures = 0
        slow_requests = 0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                try:
                    near_end_speech = bool(self._near_end_speech())
                    playback_active = bool(self._playback_active())
                except Exception:
                    if now - last_processing_warning >= self._warning_interval:
                        LOGGER.warning("DoA gate processing failed", exc_info=True)
                        last_processing_warning = now
                    self._stop.wait(self._active_period)
                    continue

                # Hardware speech detection can react to the robot's own speaker.
                # Echo-guarded near-end VAD wakes polling within one active period.
                if playback_active and not near_end_speech:
                    self._stop.wait(self._active_period)
                    continue

                period = self._active_period if near_end_speech else self._idle_period
                until_poll = period - (now - last_poll)
                if until_poll > 0:
                    self._stop.wait(min(self._active_period, until_poll))
                    continue

                request_started = time.monotonic()
                try:
                    result = self._source.read()
                    if result is None:
                        raise RuntimeError("daemon returned null (audio/DoA unavailable)")
                except requests.exceptions.ReadTimeout:
                    LOGGER.error(
                        "DoA disabled for this run after a daemon read timeout; "
                        "restart the Reachy daemon or device before retrying"
                    )
                    LOGGER.debug("DoA daemon read timeout", exc_info=True)
                    break
                except Exception as error:
                    slow_requests = 0
                    failures += 1
                    retry = min(
                        self._retry_initial * (2 ** min(failures - 1, 30)),
                        self._retry_max,
                    )
                    if now - last_warning >= self._warning_interval:
                        LOGGER.warning(
                            "DoA daemon polling failed (%s); retrying in %.2f s",
                            error,
                            retry,
                        )
                        last_warning = now
                    LOGGER.debug("DoA daemon polling failure", exc_info=True)
                    if self._stop.wait(retry):
                        break
                    continue

                request_finished = time.monotonic()
                request_seconds = request_finished - request_started
                # Healthy reads are paced start-to-start. If a synchronous
                # daemon read overruns the period, add one cooldown instead of
                # immediately hammering the already slow control endpoint.
                last_poll = request_finished if request_seconds >= period else request_started
                if request_seconds > self._slow_request_seconds:
                    slow_requests += 1
                    if slow_requests == 1:
                        LOGGER.warning(
                            "DoA daemon read is slow (%.1f ms); disabling after "
                            "%d consecutive slow reads",
                            request_seconds * 1_000,
                            self._slow_request_limit,
                        )
                    if slow_requests >= self._slow_request_limit:
                        LOGGER.error(
                            "DoA disabled for this run after %d consecutive "
                            "daemon reads exceeded %.1f ms",
                            slow_requests,
                            self._slow_request_seconds * 1_000,
                        )
                        break
                else:
                    slow_requests = 0

                if failures:
                    LOGGER.info("DoA polling recovered after %d failure(s)", failures)
                    failures = 0
                try:
                    current_near_end = bool(self._near_end_speech())
                    current_playback = bool(self._playback_active())
                    self._publish(
                        result,
                        request_finished,
                        near_end_speech=current_near_end,
                        # If playback crossed either edge during this request,
                        # conservatively reject the hardware speech bit.
                        playback_active=playback_active or current_playback,
                    )
                except Exception:
                    if request_finished - last_processing_warning >= self._warning_interval:
                        LOGGER.warning("DoA sample processing failed", exc_info=True)
                        last_processing_warning = request_finished
        finally:
            self._close_source()

    def _publish(
        self,
        result: tuple[float, bool],
        now: float,
        *,
        near_end_speech: bool,
        playback_active: bool,
    ) -> None:
        angle, hardware_speech = result
        # The XVF speech bit can still react to residual loudspeaker energy.
        # During playback, only the echo-guarded near-end VAD is trusted.
        hardware_gate = bool(hardware_speech) and not playback_active
        if hardware_gate or near_end_speech:
            self._tracker.update(
                float(angle),
                hardware_speech=hardware_gate,
                near_end_speech=near_end_speech,
                head_pose=self._head_pose(),
                now=now,
            )
